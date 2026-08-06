#include "rbe_tip4p.h"
#include <mpi.h>
#include <cmath>
#include <immintrin.h>
#include "atom.h"
#include "angle.h"
#include "bond.h"
#include "comm.h"
#include "force.h" 
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>

#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef> 
#include <vector>
#include<numeric>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

// Compatibility: _mm512_sincos_ps requires Intel SVML. Provide scalar fallback.
#ifndef __SVML__
static inline __m512 _mm512_sincos_ps(__m512 *pcos, __m512 x) {
  alignas(64) float x_arr[16], sin_arr[16], cos_arr[16];
  _mm512_store_ps(x_arr, x);
  for (int ii = 0; ii < 16; ++ii) {
    sin_arr[ii] = std::sin(x_arr[ii]);
    cos_arr[ii] = std::cos(x_arr[ii]);
  }
  *pcos = _mm512_load_ps(cos_arr);
  return _mm512_load_ps(sin_arr);
}
#endif

#ifdef LMP_GPU
int rbe_gpu_compute_rho(const int nlocal, const int pcount, const float *x,
                        const float *y, const float *z, const float *q,
                        const float *kx, const float *ky, const float *kz,
                        float *rho_cos, float *rho_sin);
int rbe_gpu_compute_force(const int nlocal, const int pcount, const float *x,
                          const float *y, const float *z, const float *q,
                          const float *kx, const float *ky, const float *kz,
                          const float *fac, const float *midterm_arr,
                          const float *rho_all_cos, const float *rho_all_sin,
                          const float midterm_scalar, float *fx, float *fy, float *fz);
#endif


struct t_complex
{
	float im,re;
};

static t_complex cmul(t_complex a, t_complex b)
{
	t_complex c;

	c.re = a.re * b.re - a.im * b.im;
	c.im = a.re * b.im + a.im * b.re;

	return c;
}

static t_complex conjugate(t_complex c)
{
	t_complex d;

	d.re = c.re;
	d.im = -c.im;

	return d;
}

/**
 * random number generator
 * 
 * 
 * 
*/
double RBETIP4P::randn_box_muller_linear_congruential1(const double Mean, const double SquareMargin)
{
	const double epsilon = 1.17549e-038; 
	const double two_pi = 2.0 * 3.14159265358979323846;
	static double z0, z1;
	double u1, u2;
	do {
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * SquareMargin + Mean;
}

inline double RBETIP4P::Prob(double m, double alpha, double L, double S1, double pi)
{
	double k = 2 * pi * m / L;
	return exp(-(k) * (k) / (4 * alpha)) / S1;
}

double RBETIP4P::MH_D(double m, double alpha, double L, double pi)
{
	double a;
	if (m == 0 || fabs(m) < 1.0e-13)
	{
		a = erf(0.5 / sqrt(alpha * L * L / (pi * pi)));
	}
	else
	{
		a = 0.5 * (erf((abs(m) + 0.5) / sqrt(alpha * L * L / (pi * pi))) - erf((abs(m) - 0.5) / sqrt(alpha * L * L / (pi * pi))));
	}
	return a;
}

#define SMALL 0.00001


RBETIP4P::RBETIP4P(LAMMPS* lmp) : KSpace(lmp)
{
    // triclinic_support = 1;
    tip4pflag = 1;
	ewaldflag = 1;
	pppmflag = 1;  // accept PPPM-compatible pair styles (lj/cut/tip4p/long)
	use_gpu_accel = false;
	TimeSet = new double[1000];

	// triclinic = domain->triclinic;
}
 
void RBETIP4P::init()
{
	// MPI_Comm_rank(world,&me);
  	// MPI_Comm_size(world,&RankID);
    // TIP4P RBE requires newton on, b/c it computes forces on ghost atoms

    // if (force->newton == 0)
    // error->all(FLERR,"Kspace style pppm/tip4p requires newton on");

    if (!comm->me) utils::logmesg(lmp,"RBE initialization ...\n");

    // error check

	triclinic_check();
	if (domain->dimension == 2)
		error->all(FLERR, "Cannot use Ewald with 2d simulation");

	if (!atom->q_flag) error->all(FLERR, "Kspace style requires atom attribute q");

	if (slabflag == 0 && domain->nonperiodic > 0)
		error->all(FLERR, "Cannot use non-periodic boundaries with RBE");
	if (slabflag) {
		if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
			domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
			error->all(FLERR, "Incorrect boundaries with slab Ewald");
		if (domain->triclinic)
			error->all(FLERR, "Cannot (yet) use Ewald with triclinic box "
				"and slab correction");
	}

    // compute two charge force

    two_charge();


	// extract short-range Coulombic cutoff from pair style

	int triclinic = domain->triclinic;
	pair_check();

	// set accuracy (force units) from accuracy_relative or accuracy_absolute

	int itmp;
	double* p_cutoff = (double*)force->pair->extract("cut_coul", itmp);
	if (p_cutoff == NULL)
		error->all(FLERR, "KSpace style is incompatible with Pair style");
	cutoff = *p_cutoff;

    // if kspace is TIP4P, extract TIP4P params from pair style
    // bond/angle are not yet init(), so ensure equilibrium request is valid

    qdist = 0.0;

    if (tip4pflag) {
        if (me == 0) utils::logmesg(lmp,"  extracting TIP4P info from pair style\n");

        auto p_qdist = (double *) force->pair->extract("qdist",itmp);
        int *p_typeO = (int *) force->pair->extract("typeO",itmp);
        int *p_typeH = (int *) force->pair->extract("typeH",itmp);
        int *p_typeA = (int *) force->pair->extract("typeA",itmp);
        int *p_typeB = (int *) force->pair->extract("typeB",itmp);
        if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
        error->all(FLERR,"Pair style is incompatible with TIP4P KSpace style");
        qdist = *p_qdist;
        typeO = *p_typeO;
        typeH = *p_typeH;
        int typeA = *p_typeA;
        int typeB = *p_typeB;

        if (force->angle == nullptr || force->bond == nullptr ||
            force->angle->setflag == nullptr || force->bond->setflag == nullptr)
        	error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
        if (typeA < 1 || typeA > atom->nangletypes ||
            force->angle->setflag[typeA] == 0)
        	error->all(FLERR,"Bad TIP4P angle type for RBE/TIP4P");
        if (typeB < 1 || typeB > atom->nbondtypes ||
            force->bond->setflag[typeB] == 0)
        	error->all(FLERR,"Bad TIP4P bond type for RBE/TIP4P");
        double theta = force->angle->equilibrium_angle(typeA);
        double blen = force->bond->equilibrium_distance(typeB);
        alpha = qdist / (cos(0.5*theta) * blen);
		if(comm->me == 0) cout<<"alpha " << alpha << endl;
	}

	// compute qsum & qsqsum and warn if not charge-neutral
	// setup K-space resolution

	bigint natoms = atom->natoms;

	// use xprd,yprd,zprd even if triclinic so grid size is the same
	// adjust z dimension for 2d slab Ewald
	// 3d Ewald just uses zprd since slab_volfactor = 1.0

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd;
	double zprd_slab = zprd*slab_volfactor;

	

	scale = 1.0;
	qqrd2e = force->qqrd2e;
	qsum_qsq();
	natoms_original = atom->natoms;
	// if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
	// else accuracy = accuracy_relative * two_charge_force;
	// if (!gewaldflag) adjust_gewald();

	volume = xprd * yprd * zprd;

	if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  	else accuracy = accuracy_relative * two_charge_force;	

	if (!gewaldflag) {
		if (accuracy <= 0.0)
			error->all(FLERR,"KSpace accuracy must be > 0");
		if (q2 == 0.0)
			error->all(FLERR,"Must use 'kspace_modify gewald' for uncharged system");
		g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
		if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
		else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  	}

	Step = 0;

	setup();


	if (me == 0) {
    	std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    	utils::logmesg(lmp,mesg);
  	}
}

void RBETIP4P::settings(int narg, char** arg)
{
	MPI_Comm_rank(world, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &RankID);
	// RankID = 1;

	if ((narg != 2) && (narg!=3)) error->all(FLERR, "Illegal kspace_style RBE command");
	if (narg < 1) error->all(FLERR,"Illegal kspace_style {} command", force->kspace_style);

	accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));
	if (accuracy_relative > 1.0)
		error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}",
				accuracy_relative, force->kspace_style);
	// g_ewald = fabs(utils::numeric(FLERR,arg[0],false,lmp));
	// if (accuracy_relative > 1.0)
    // 	error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}",
    //         accuracy_relative, force->kspace_style);

	P = int(utils::numeric(FLERR, arg[1], false, lmp));
	// RankID = int(utils::numeric(FLERR, arg[2], false, lmp));

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;
	pi = 3.141592653589793;
	pi_sqrt = 1.772453850905516;
	// beta = g_ewald * g_ewald;

	K_All = new int[100][3];

	K_Sample = new float* [P];
	for (int i = 0; i < P; i++)
	{
		K_Sample[i] = new float[3];
	}

}

void RBETIP4P::setup()
{
	MPI_Comm_rank(world, &me);

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	// if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;
	volume = xprd * yprd * zprd;
	
	S_X = sqrt(beta * xprd * xprd / M_PI) * (1 + 2 * exp(-(beta * xprd * xprd)));
	S_Y = sqrt(beta * yprd * yprd / M_PI) * (1 + 2 * exp(-(beta * yprd * yprd)));
	S_Z = sqrt(beta * zprd * zprd / M_PI) * (1 + 2 * exp(-(beta * zprd * zprd)));
	S = S_X * S_Y * S_Z - 1;

	beta = g_ewald * g_ewald;
	// if (me == 0)
	// {
	// 	cout << xprd << "=Lx   " << yprd << "=Ly    " << zprd << "=Lz   " << beta << "=beta   " << atom->natoms << "=N    " << P << "=P   " << endl;
	// }
}

/* ---------------------------------------------------------------------- */

void RBETIP4P::compute(int eflag, int vflag)
{	
    double time;
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();

	ev_init(eflag, vflag);
	// if atom count has changed, update qsum and qsqsum
	// if (Step == 0)srand(comm->me * 42);
	 
	if (atom->natoms != natoms_original) {
		qsum_qsq();
		natoms_original = atom->natoms;
	}

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	scale = 1;

	S_X = sqrt(beta * xprd * xprd / M_PI) * (1 + 2 * exp(-(beta * xprd * xprd)));
	S_Y = sqrt(beta * yprd * yprd / M_PI) * (1 + 2 * exp(-(beta * yprd * yprd)));
	S_Z = sqrt(beta * zprd * zprd / M_PI) * (1 + 2 * exp(-(beta * zprd * zprd)));
	S = S_X * S_Y * S_Z - 1;

	double V = xprd * yprd * zprd;
	float K[P][3];

	int This_Index = Step * P;

	int this_rank = (Step % RankID);

	if (((Step % RankID)==0)&&(me< RankID)) {
		S1_X = sqrt(beta * xprd * xprd / M_PI) * (1 + 2 * exp(-(beta * xprd * xprd)));
		S1_Y = sqrt(beta * yprd * yprd / M_PI) * (1 + 2 * exp(-(beta * yprd * yprd)));
		S1_Z = sqrt(beta * zprd * zprd / M_PI) * (1 + 2 * exp(-(beta * zprd * zprd)));
		S0 = S1_X * S1_Y * S1_Z - 1;
		// if(me == 0 && Step % 1000 == 0){
		// 	cout << "beta: " << beta << endl;
		// 	cout << "S1_X: " << S1_X <<endl;
		// 	cout << "S1_Y: " << S1_Y <<endl;
		// 	cout << "S1_Z: " << S1_Z <<endl;
		// 	cout << "S0: " << S0 <<endl;
		// }
		xprd0 = xprd;
		yprd0 = yprd;
		zprd0 = zprd;
		int mx[5*P], my[5*P], mz[5*P];
		do {
			mx[0] = round(randn_box_muller_linear_congruential1(0, sqrt(beta * xprd * xprd / (2 * pi * pi))));
			my[0] = round(randn_box_muller_linear_congruential1(0, sqrt(beta * yprd * yprd / (2 * pi * pi))));
			mz[0] = round(randn_box_muller_linear_congruential1(0, sqrt(beta * zprd * zprd / (2 * pi * pi))));
		} while (mx[0] == 0 && my[0] == 0 && mz[0] == 0);
		for (int i = 0; i < 5*P-1; i++) 
		{
			double x = randn_box_muller_linear_congruential1(0, sqrt(beta * xprd * xprd / (2 * pi * pi)));
			double mold = mx[i];
			double mnew = round(x);
			double pup = Prob(mnew, beta, xprd, S1_X, pi);
			double qup = MH_D(mold, beta, xprd, pi);
			double pdown = Prob(mold, beta, xprd, S1_X, pi);
			double qdown = MH_D(mnew, beta, xprd, pi);
			double acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
			double y = (rand() % 10000 + 0.00) / 10000.0;
			if (y < acce) {
				mx[i + 1] = mnew;
			}
			else {
				mx[i + 1] = mold;
			}
			double xx = randn_box_muller_linear_congruential1(0, sqrt(beta * yprd * yprd / (2 * pi * pi)));
			mold = my[i];
			mnew = round(xx);
			pup = Prob(mnew, beta, yprd, S1_Y, pi);
			qup = MH_D(mold, beta, yprd, pi);
			pdown = Prob(mold, beta, yprd, S1_Y, pi);
			qdown = MH_D(mnew, beta, yprd, pi);
			acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
			double yy = (rand() % 10000 + 0.00) / 10000.0;
			if (yy < acce) {
				my[i + 1] = mnew;
			}
			else {
				my[i + 1] = mold;
			}
			double xxx = randn_box_muller_linear_congruential1(0, sqrt(beta * zprd * zprd / (2 * pi * pi)));
			mold = mz[i];
			mnew = round(xxx);
			pup = Prob(mnew, beta, zprd, S1_Z, pi);
			qup = MH_D(mold, beta, zprd, pi);
			pdown = Prob(mold, beta, zprd, S1_Z, pi);
			qdown = MH_D(mnew, beta, zprd, pi);
			acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
			double yyy = (rand() % 10000 + 0.00) / 10000.0;
			if (yyy < acce) {
				mz[i + 1] = mnew;
			}
			else {
				mz[i + 1] = mold;
			}
			if (mx[i + 1] == 0 && my[i + 1] == 0 && mz[i + 1] == 0)
				i = i - 1;
		}
		for (int i = 0; i < P; i++)
		{
			K_Sample[i][0] = mx[5*i+4] + 0.00;
			K_Sample[i][1] = my[5*i+4] + 0.00;
			K_Sample[i][2] = mz[5*i+4] + 0.00;
		}
	}

	if (me == this_rank)
	{
		for (int i = 0; i < P; i++)
		{
			K[i][0] = K_Sample[i][0];
			K[i][1] = K_Sample[i][1];
			K[i][2] = K_Sample[i][2];
		}
	}
	// if(me == 0) cout << "K " << K[0][0] << ' '<< K[0][1] << ' '<< K[0][2] <<endl;
	MPI_Bcast((float*)K, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);

	

	Step++;

	/*  Set Pointer */
	double** x = atom->x;
	double** f = atom->f;
	double* q = atom->q;
	int* type = atom->type;
	int nlocal = atom->nlocal;
	double qqrd2e = force->qqrd2e;
	double dielectric = force->dielectric;


	double *xi,xM[3];
	int iH1 = 0, iH2 = 0;

	int tmp = int ( ceil ( ( nlocal + 0.0 ) / 16.0 ) ) * 16;

	float X[int ( ceil ( ( nlocal + 0.0 ) / 16.0 ) ) * 16][3]; 
	float F[int ( ceil ( ( nlocal + 0.0 ) / 16.0 ) ) * 16][3];  
	float Q[int ( ceil ( ( nlocal + 0.0 ) / 16.0 ) ) * 16];

	// if (me == 0) {
    // 	std::string mesg = fmt::format(" RBETIP4P vectoring arguments \n");
    // 	utils::logmesg(lmp,mesg);
  	// }

	for (int i = 0; i < tmp; i++)
	{
		if (i < nlocal) {
			// find M site if TIP4P
			if (type[i] == typeO) {
				find_M(i,iH1,iH2,xM);
				xi = xM;
			} else xi = x[i];
			X[i][0] = xi[0]; X[i][1] = xi[1]; X[i][2] = xi[2];
			F[i][0] = 0.00; F[i][1] = 0.00; F[i][2] = 0.00;
			Q[i] = q[i];
			// if(me == 0 && i <= 4 && type[i] == typeO){
			// 	cout << "O position " << x[i][0] << " " << x[i][1] << " " << x[i][2]<<endl;
			// 	cout << "M site " << X[i][0] << " " << X[i][1] << " " << X[i][2]<<endl;
			// }
		}
		else
		{
			X[i][0] = 0.0; X[i][1] = 0.00; X[i][2] = 0.00;
			F[i][0] = 0.00; F[i][1] = 0.00; F[i][2] = 0.00;
			Q[i] = 0.00;
		}
	}

	float pxyz[3] = {static_cast<float>(2 * pi / xprd),static_cast<float>(2 * pi / yprd),static_cast<float>(2 * pi / zprd) };
	float Rho[P][2], Rho_All[P][2];
	float midterm[P][3];
	
	float K2_inv[P];

	float Kx[P][3];
	float fac[P];

	for (int i = 0; i < P; i++) 
	{
		
		Kx[i][0] = K[i][0] * pxyz[0];
		Kx[i][1] = K[i][1] * pxyz[1];
		Kx[i][2] = K[i][2] * pxyz[2];

		K2_inv[i] = 1 / (Kx[i][0] * Kx[i][0] + Kx[i][1] * Kx[i][1] + Kx[i][2] * Kx[i][2]);
		fac[i] = S0/S * (Prob(K[i][0], beta, xprd, S_X, pi)/Prob(K[i][0], beta, xprd0, S1_X, pi)) * 
						(Prob(K[i][1], beta, yprd, S_Y, pi)/Prob(K[i][1], beta, yprd0, S1_Y, pi)) * 
						(Prob(K[i][2], beta, zprd, S_Z, pi)/Prob(K[i][2], beta, zprd0, S1_Z, pi));

		midterm[i][0] = Kx[i][0] * K2_inv[i];
		midterm[i][1] = Kx[i][1] * K2_inv[i];
		midterm[i][2] = Kx[i][2] * K2_inv[i];
	}

	
	float KxKx0[int(ceil((P + 0.0) / 16.0)) * 16], KxKx1[int(ceil((P + 0.0) / 16.0)) * 16], KxKx2[int(ceil((P + 0.0) / 16.0)) * 16];
	float Rho_Cos[int(ceil((P + 0.0) / 16.0)) * 16], Rho_Sin[int(ceil((P + 0.0) / 16.0)) * 16];
	for (int i = 0; i < int(ceil((P + 0.0) / 16.0)) * 16; i++)
	{
		if (i < P) {
			KxKx0[i] = Kx[i][0];
			KxKx1[i] = Kx[i][1];
			KxKx2[i] = Kx[i][2];
		}
		else if (i >= P)
		{
			KxKx0[i] = 0.00;
			KxKx1[i] = 0.00;
			KxKx2[i] = 0.00;
		}
	}

	float KKx[16][3];
	float moment[16];

	// if (me == 0) {
    // 	std::string mesg = fmt::format(" RBETIP4P computing rho \n");
    // 	utils::logmesg(lmp,mesg);
  	// }

	// ── GPU offload: Rho computation ──
	bool rho_on_gpu = false;
#ifdef LMP_GPU
	if (use_gpu_accel) {
		std::vector<float> Xf(nlocal), Yf(nlocal), Zf(nlocal), Qf(nlocal);
		for (int ii = 0; ii < nlocal; ++ii) {
			Xf[ii] = static_cast<float>(X[ii][0]);
			Yf[ii] = static_cast<float>(X[ii][1]);
			Zf[ii] = static_cast<float>(X[ii][2]);
			Qf[ii] = static_cast<float>(Q[ii]);
		}
		if (rbe_gpu_compute_rho(nlocal, P, Xf.data(), Yf.data(), Zf.data(),
		                        Qf.data(), KxKx0, KxKx1, KxKx2,
		                        Rho_Cos, Rho_Sin) == 0)
			rho_on_gpu = true;
	}
#endif
	if (!rho_on_gpu) {
	for (int i = 0; i < P; i += 16)
	{
		__m512 Real, Imag, X0, X1, X2, qq, Cos, Sin,
				Moment,KXKX0,KXKX1,KXKX2;

		Real = Imag = _mm512_setzero_ps();
		KXKX0 = _mm512_load_ps(&KxKx0[i]);
		KXKX1 = _mm512_load_ps(&KxKx1[i]);
		KXKX2 = _mm512_load_ps(&KxKx2[i]);

		for (int j = 0; j < nlocal; j++)
		{
			X0 = _mm512_set1_ps(X[j][0]);
			X1 = _mm512_set1_ps(X[j][1]);
			X2 = _mm512_set1_ps(X[j][2]);
			qq = _mm512_set1_ps(Q[j]);
			Moment = KXKX0 * X0 + KXKX1 * X1 + KXKX2 * X2;
			Sin = _mm512_sincos_ps(&Cos, Moment);
			Real = Real + qq * Cos;
			Imag = Imag + qq * Sin;
		}
		_mm512_store_ps(&Rho_Cos[i], Real);
		_mm512_store_ps(&Rho_Sin[i], Imag);
	}
	}  // end GPU rho offload

	for (int i = 0; i < P; i++)
	{
		Rho[i][0] = Rho_Cos[i];
		Rho[i][1] = Rho_Sin[i];
	}

	MPI_Allreduce((float*)Rho, (float*)Rho_All, 2 * P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	// if(me == 0 && Step % 1000 == 0) cout << "Rho: " << Rho_All[0][0] << " " << Rho_All[0][1] << endl;

	double MIDTERM = -4 * pi * (S0 / (P + 0.00)) * qqrd2e / (V);

	// GPU offload: Force computation
	bool force_on_gpu = false;
#ifdef LMP_GPU
	if (use_gpu_accel) {
		std::vector<float> Xf(nlocal), Yf(nlocal), Zf(nlocal), Qf(nlocal);
		for (int ii = 0; ii < nlocal; ++ii) {
			Xf[ii] = static_cast<float>(X[ii][0]);
			Yf[ii] = static_cast<float>(X[ii][1]);
			Zf[ii] = static_cast<float>(X[ii][2]);
			Qf[ii] = static_cast<float>(Q[ii]);
		}
		std::vector<float> mterm(3 * P);
		std::vector<float> rho_c(P), rho_s(P);
		for (int ii = 0; ii < P; ++ii) {
			mterm[3*ii+0]=midterm[ii][0]; mterm[3*ii+1]=midterm[ii][1]; mterm[3*ii+2]=midterm[ii][2];
			rho_c[ii]=Rho_All[ii][0]; rho_s[ii]=Rho_All[ii][1];
		}
		std::vector<float> fgx(nlocal), fgy(nlocal), fgz(nlocal);
		if (rbe_gpu_compute_force(nlocal, P, Xf.data(), Yf.data(), Zf.data(),
			Qf.data(), KxKx0, KxKx1, KxKx2, fac, mterm.data(),
			rho_c.data(), rho_s.data(), static_cast<float>(MIDTERM),
			fgx.data(), fgy.data(), fgz.data()) == 0) {
			for (int ii=0; ii<nlocal; ++ii) { F[ii][0]+=fgx[ii]; F[ii][1]+=fgy[ii]; F[ii][2]+=fgz[ii]; }
			force_on_gpu = true;
		}
	}
#endif
	if (!force_on_gpu) {

	// double factor = Prob(mold, beta, zprd, S1_Z, pi);
	
	// if (me == 0) {
    // 	std::string mesg = fmt::format(" RBETIP4P computing forces \n");
    // 	utils::logmesg(lmp,mesg);
  	// }

	for (int i = 0; i < nlocal; i+=16)
	{
		float X1[16], X2[16], X3[16];
		for (int j = 0; j < 16; j++)
		{
			X1[j] = X[i + j][0];
			X2[j] = X[i + j][1];
			X3[j] = X[i + j][2]; 
		}

		__m512 X_0, X_1, X_2, Fx, Fy, Fz, qq, Kx, Ky, Kz, moment, Cos, Sin, midterm_512, Rho_All_0, Rho_All_1, Imag,
			midterm_0,midterm_1,midterm_2, factor;
		X_0 = _mm512_load_ps(&X1[0]);
		X_1 = _mm512_load_ps(&X2[0]);
		X_2 = _mm512_load_ps(&X3[0]);
		Fx=Fy=Fz=_mm512_setzero_ps();
		qq = _mm512_load_ps(&Q[i]);

		for (int j = 0; j < P; j++)
		{
			Kx = _mm512_set1_ps(KxKx0[j]);
			Ky = _mm512_set1_ps(KxKx1[j]);
			Kz = _mm512_set1_ps(KxKx2[j]);
			moment = -(Kx * X_0 + Ky * X_1 + Kz * X_2);

			Sin = _mm512_sincos_ps(&Cos, moment);

			midterm_512 = _mm512_set1_ps(MIDTERM);
			Rho_All_0 = _mm512_set1_ps(Rho_All[j][0]);
			Rho_All_1 = _mm512_set1_ps(Rho_All[j][1]);
			Imag = (Cos * Rho_All_1 + Sin * Rho_All_0) * midterm_512;

			// double fac = S0/S * (Prob(K[j][0], beta, xprd, S_X, pi)/Prob(K[j][0], beta, xprd0, S1_X, pi)) * 
			// 			(Prob(K[j][1], beta, yprd, S_Y, pi)/Prob(K[j][1], beta, yprd0, S1_Y, pi)) * 
			// 			(Prob(K[j][2], beta, zprd, S_Z, pi)/Prob(K[j][2], beta, zprd0, S1_Z, pi));
			// // if(me == 0 && i == 0 && j == 0) cout << "factor " << fac <<endl;
			factor = _mm512_set1_ps(fac[j]);
			
			midterm_0 = _mm512_set1_ps(midterm[j][0]);
			midterm_1 = _mm512_set1_ps(midterm[j][1]);
			midterm_2 = _mm512_set1_ps(midterm[j][2]);
			Fx = Fx + qq * Imag * midterm_0 * factor;
			Fy = Fy + qq * Imag * midterm_1 * factor;
			Fz = Fz + qq * Imag * midterm_2 * factor;

		}

		_mm512_store_ps(&X1[0], Fx);
		_mm512_store_ps(&X2[0], Fy);
		_mm512_store_ps(&X3[0], Fz);

		for (int j = 0; j < 16; j++) {
			F[i+j][0] = F[i+j][0] + X1[j];
			F[i+j][1] = F[i+j][1] + X2[j];
			F[i+j][2] = F[i+j][2] + X3[j];
		}
	}
	}  // end GPU force offload


	for (int i = 0; i < nlocal; i++)
	{
		if (type[i] != typeO) {
      		f[i][0] += F[i][0];
			f[i][1] += F[i][1]; 
			f[i][2] += F[i][2];

    	} else {
			find_M(i,iH1,iH2,xM);

			f[i][0] += F[i][0]*(1 - alpha);
			f[i][1] += F[i][1]*(1 - alpha);
			if (slabflag != 2) f[i][2] += F[i][2]*(1 - alpha);

			f[iH1][0] += 0.5*alpha*F[i][0];
			f[iH1][1] += 0.5*alpha*F[i][1];
			if (slabflag != 2) f[iH1][2] += 0.5*alpha*F[i][2];

			f[iH2][0] += 0.5*alpha*F[i][0];
			f[iH2][1] += 0.5*alpha*F[i][1];
			if (slabflag != 2) f[iH2][2] += 0.5*alpha*F[i][2];
		}
	}
	
		time = MPI_Wtime() - time;

    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
    }

    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_RBE_Vec.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
    }
double sum = accumulate(TimeSet, TimeSet+1000, 0.0);  
double mean =  sum / 1000; //��ֵ  
    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Mean_Vec.txt");
            outfile << mean << endl;
        outfile.close();
    }

	// sum global energy across Kspace vevs and add in volume-dependent term
	const double qscale = qqrd2e * scale;
	double energy1 = 0;
	if (eflag_global) {
		float KXX[3];
		for (int i = 0; i < P; i++)
		{
			KXX[0] = K[i][0] * pxyz[0];
			KXX[1] = K[i][1] * pxyz[1];
			KXX[2] = K[i][2] * pxyz[2];

			energy1 += fac[i] * (2 * pi * S0 / ((P + 0.00) * V)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) * K2_inv[i];
		}

		// if(me == 1 && Step % 1000 == 0){cout << "energy1: " << energy1 << endl;}
		
		energy = energy1;
		double energy_self = g_ewald * qsqsum / MY_PIS + MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);
		energy -= energy_self;
		energy *= qscale;

	}

	//vflag_global = 1;
// global virial
	if (vflag_global) {
		float KKXX[3], Moment_Term= 2 * pi * (S0 / (P + 0.00)) * qqrd2e / (V+0.00);

		// float virial_rbe[6];
		// for (int i = 0; i < 6; i++)
		// 	virial_rbe[i] = virial[i];
#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
		for (int i = 0; i < P; i++)
		{
			KKXX[0] = K[i][0] * pxyz[0];
			KKXX[1] = K[i][1] * pxyz[1]; 
			KKXX[2] = K[i][2] * pxyz[2];

			float coef1 = fac[i] * (Rho_All[i][0]* Rho_All[i][0]+ Rho_All[i][1]* Rho_All[i][1]) * K2_inv[i];
			float coef2 = (1.0 / (4 * g_ewald * g_ewald)) + K2_inv[i];
			virial[0] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[0] * KKXX[0]);
			virial[1] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[1] * KKXX[1]);
			virial[2] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[2] * KKXX[2]);

			virial[3] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[1]);
			virial[4] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[2]);
			virial[5] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[1] * KKXX[2]);
		}
		// if(me == 0 && Step % 1000 == 0){
		// 	cout << "after calculation" << endl;
		// 	cout<< "virial:" <<virial[0] << endl;
		// 	cout<< "virial:" <<virial[1] << endl;
		// 	cout<< "virial:" <<virial[2] << endl;
		// 	cout<< "virial:" <<virial[3] << endl;
		// 	cout<< "virial:" <<virial[4] << endl;
		// 	cout<< "virial:" <<virial[5] << endl;
		// }

		// for (int i = 0; i < 6; i++)
		// 	virial_rbe[i] = virial[i]- virial_rbe[i];
	} 
			
		 

// per-atom energy/virial
// energy includes self-energy correction
		if (evflag_atom) {
			if (eflag_atom) {
#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
				for (int i = 0; i < nlocal; i++) {
					if (type[i] != typeO) {
						eatom[i] -= (g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /(g_ewald*g_ewald*volume))*qscale;
						// eatom[i] *= qscale;
					} else {
						find_M(i,iH1,iH2,xM);
						eatom[i] -= (g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /(g_ewald*g_ewald*volume))*qscale*(1-alpha);
						// eatom[i] *= qscale;
						eatom[iH1] -= (g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /(g_ewald*g_ewald*volume))*qscale*0.5*alpha;
						eatom[iH2] -= (g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /(g_ewald*g_ewald*volume))*qscale*0.5*alpha;
					}
					
				}
			}
		}
// // 			if (vflag_atom){
// // #if defined(LMP_SIMD_COMPILER)
// // #pragma vector aligned
// // #pragma simd
// // #endif
// // 				for (int i = 0; i < nlocal; i++){
					
// // 					for (int j = 0; j < 6; j++){
// // 						if (type[i] != typeO && type[i] != typeH) { vatom[i][j] *= q[i] * qscale;
// // 						}
// // 						if (type[i] == typeO) {
// // 							find_M(i,iH1,iH2,xM);
// // 							double v0 = vatom[i][j];
// // 							vatom[i][j] *= q[i] * qscale * (1-alpha);
// // 							vatom[iH1][j] = vatom[iH1][j]* q[iH1] * qscale + v0 * q[i] * qscale * 0.5 * alpha;
// // 							vatom[iH1][j] = vatom[iH2][j]* q[iH2] * qscale + v0 * q[i] * qscale * 0.5 * alpha;
// // 						}
// // 					}
					
				
// // 				}
					
// // 			}
// 		}
	if (slabflag == 1) slabcorr();
}

/* ---------------------------------------------------------------------- */

RBETIP4P::~RBETIP4P()
{
	for (int i = 0; i < P; i++)
		delete[] K_Sample[i];
	delete[] K_Sample;
}

/* ---------------------------------------------------------------------- */

double RBETIP4P::memory_usage()
{
	return 1.0;
}

/* ---------------------------------------------------------------------- */

void RBETIP4P::slabcorr()
{
	// compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd_slab = domain->zprd*slab_volfactor;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double *xi, xM[3]; int iH1, iH2;  //for TIP4P virtual site

  // sum local contributions to get global dipole moment
  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (type[i] == typeO) {
      find_M(i,iH1,iH2,xM);
      xi = xM;
    } else xi = x[i];
    dipole += q[i]*xi[2];
  }

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++)
      dipole_r2 += q[i]*x[i][2]*x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_r2 = tmp;
  }

  // compute corrections

  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all -
    qsum*dipole_r2 - qsum*qsum*zprd_slab*zprd_slab/12.0)/volume;
  const double qscale = force->qqrd2e * scale;

  if (eflag_global) energy_1 += qscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * q[i]*(x[i][2]*dipole_all - 0.5*(dipole_r2 +
        qsum*x[i][2]*x[i][2]) - qsum*zprd_slab*zprd_slab/12.0);
  }

  // add on force corrections

  double ffact = qscale * (-4.0*MY_PI/volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) {
    double fzi_corr = ffact * q[i]*(dipole_all - qsum*x[i][2]);
    if (type[i] == typeO) {
      find_M(i,iH1,iH2,xM);
      f[i][2] += fzi_corr*(1 - alpha);
      f[iH1][2] += 0.5*alpha*fzi_corr;
      f[iH2][2] += 0.5*alpha*fzi_corr;
    }
    else f[i][2] += fzi_corr;
  }

}

void RBETIP4P::find_M(int i, int &iH1, int &iH2, double *xM)
{
	// if (me == 0) {
    // 	std::string mesg = fmt::format(" RBETIP4P finding M site \n");
    // 	utils::logmesg(lmp,mesg);
  	// }
  double **x = atom->x;

  iH1 = atom->map(atom->tag[i] + 1);
  iH2 = atom->map(atom->tag[i] + 2);

  if (iH1 == -1 || iH2 == -1) error->one(FLERR,"TIP4P hydrogen is missing");
  if (atom->type[iH1] != typeH || atom->type[iH2] != typeH)
    error->one(FLERR,"TIP4P hydrogen has incorrect atom type");

  if (triclinic) {

    // need to use custom code to find the closest image for triclinic,
    // since local atoms are in lambda coordinates, but ghosts are not.

    int *sametag = atom->sametag;
    double xo[3],xh1[3],xh2[3],xm[3];
    const int nlocal = atom->nlocal;

    for (int ii = 0; ii < 3; ++ii) {
      xo[ii] = x[i][ii];
      xh1[ii] = x[iH1][ii];
      xh2[ii] = x[iH2][ii];
    }

    if (i < nlocal) domain->lamda2x(x[i],xo);
    if (iH1 < nlocal) domain->lamda2x(x[iH1],xh1);
    if (iH2 < nlocal) domain->lamda2x(x[iH2],xh2);

    double delx = xo[0] - xh1[0];
    double dely = xo[1] - xh1[1];
    double delz = xo[2] - xh1[2];
    double rsqmin = delx*delx + dely*dely + delz*delz;
    double rsq;
    int closest = iH1;

    // no need to run lamda2x() here -> ghost atoms

    while (sametag[iH1] >= 0) {
      iH1 = sametag[iH1];
      delx = xo[0] - x[iH1][0];
      dely = xo[1] - x[iH1][1];
      delz = xo[2] - x[iH1][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < rsqmin) {
        rsqmin = rsq;
        closest = iH1;
        xh1[0] = x[iH1][0];
        xh1[1] = x[iH1][1];
        xh1[2] = x[iH1][2];
      }
    }
    iH1 = closest;

    closest = iH2;
    delx = xo[0] - xh2[0];
    dely = xo[1] - xh2[1];
    delz = xo[2] - xh2[2];
    rsqmin = delx*delx + dely*dely + delz*delz;

    while (sametag[iH2] >= 0) {
      iH2 = sametag[iH2];
      delx = xo[0] - x[iH2][0];
      dely = xo[1] - x[iH2][1];
      delz = xo[2] - x[iH2][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < rsqmin) {
        rsqmin = rsq;
        closest = iH2;
        xh2[0] = x[iH2][0];
        xh2[1] = x[iH2][1];
        xh2[2] = x[iH2][2];
      }
    }
    iH2 = closest;

    // finally compute M in real coordinates ...

    double delx1 = xh1[0] - xo[0];
    double dely1 = xh1[1] - xo[1];
    double delz1 = xh1[2] - xo[2];

    double delx2 = xh2[0] - xo[0];
    double dely2 = xh2[1] - xo[1];
    double delz2 = xh2[2] - xo[2];

    xm[0] = xo[0] + alpha * 0.5 * (delx1 + delx2);
    xm[1] = xo[1] + alpha * 0.5 * (dely1 + dely2);
    xm[2] = xo[2] + alpha * 0.5 * (delz1 + delz2);

    // ... and convert M to lamda space for PPPM

    domain->x2lamda(xm,xM);

  } else {

    // set iH1,iH2 to index of closest image to O

    iH1 = domain->closest_image(i,iH1);
    iH2 = domain->closest_image(i,iH2);

    double delx1 = x[iH1][0] - x[i][0];
    double dely1 = x[iH1][1] - x[i][1];
    double delz1 = x[iH1][2] - x[i][2];

    double delx2 = x[iH2][0] - x[i][0];
    double dely2 = x[iH2][1] - x[i][1];
    double delz2 = x[iH2][2] - x[i][2];

    xM[0] = x[i][0] + alpha * 0.5 * (delx1 + delx2);
    xM[1] = x[i][1] + alpha * 0.5 * (dely1 + dely2);
    xM[2] = x[i][2] + alpha * 0.5 * (delz1 + delz2);
  }
//   if (me == 0) {
//     	std::string mesg = fmt::format(" RBETIP4P finded M site \n");
//     	utils::logmesg(lmp,mesg);
//   	}
}

