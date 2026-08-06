#include "rbe.h"
#include <mpi.h>
#include <cmath>
#include "atom.h"
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
 
#include <immintrin.h> 
#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef> 
#include <vector>
#include <random>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;
#define SMALL 0.00001
#define LARGE 1000000

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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);
std::uniform_real_distribution<> dis_uniform(-1, 1);


/*
random number generator
*/
double randn_box_muller_linear_congruential(const double Mean, const double SquareMargin)
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

inline double Prob(double m, double alpha, double L, double S1, double pi)
{
	double k = 2 * pi * m / L;
	return exp(-(k) * (k) / (4 * alpha)) / S1;
}

static float gaussian(float k2, float alpha)
{
    return exp(-k2 / (4 * alpha));
}

double MH_D(double m, double alpha, double L, double pi)
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

/* ---------------------------------------------------------------------- */
Rbe::Rbe(LAMMPS* lmp) : KSpace(lmp)
{
	TimeSet = new double[1000];
	ewaldflag = 1;
	use_gpu_accel = false;
}
 
void Rbe::init()
{
	if (comm->me == 0) utils::logmesg(lmp,"Ewald initialization ...\n");

	// error check

	triclinic_check();
	if (domain->dimension == 2)
		error->all(FLERR,"Cannot use Ewald with 2d simulation");

	if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

	if (slabflag == 0 && domain->nonperiodic > 0)
		error->all(FLERR,"Cannot use non-periodic boundaries with Ewald");
	if (slabflag) {
		if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
			domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
		error->all(FLERR,"Incorrect boundaries with slab Ewald");
		if (domain->triclinic)
		error->all(FLERR,"Cannot (yet) use Ewald with triclinic box "
					"and slab correction");
	}

	// compute two charge force

	two_charge();

	// extract short-range Coulombic cutoff from pair style

	triclinic = domain->triclinic;
	pair_check();

	int itmp;
	auto p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
	if (p_cutoff == nullptr)
		error->all(FLERR,"KSpace style is incompatible with Pair style");
	double cutoff = *p_cutoff;

	// compute qsum & qsqsum and warn if not charge-neutral

	scale = 1.0;
	qqrd2e = force->qqrd2e;
	qsum_qsq();
	natoms_original = atom->natoms;

	// set accuracy (force units) from accuracy_relative or accuracy_absolute

	if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
	else accuracy = accuracy_relative * two_charge_force;

	// setup K-space resolution

	bigint natoms = atom->natoms;

	// use xprd,yprd,zprd even if triclinic so grid size is the same
	// adjust z dimension for 2d slab Ewald
	// 3d Ewald just uses zprd since slab_volfactor = 1.0

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd;
	double zprd_slab = zprd*slab_volfactor;

	// make initial g_ewald estimate
	// based on desired accuracy and real space cutoff
	// fluid-occupied volume used to estimate real-space error
	// zprd used rather than zprd_slab

	if (!gewaldflag) {
		if (accuracy <= 0.0)
		error->all(FLERR,"KSpace accuracy must be > 0");
		if (q2 == 0.0)
		error->all(FLERR,"Must use 'kspace_modify gewald' for uncharged system");
		g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
		if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
		else g_ewald = sqrt(-log(g_ewald)) / cutoff;
	}

	// setup Ewald coefficients so can print stats

	setup();

	if (me == 0) {
    	std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    	utils::logmesg(lmp,mesg);
  	}
}

/* ---------------------------------------------------------------------- */

void Rbe::settings(int narg, char** arg)
{
	MPI_Comm_rank(world, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &RankID);
	// RankID = 1;

	if ((narg != 2) && (narg!=3)) error->all(FLERR, "Illegal kspace_style RBE command");
	accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));
	if (accuracy_relative > 1.0)
		error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}",
				accuracy_relative, force->kspace_style);	
	P = int(utils::numeric(FLERR, arg[1], false, lmp));
	if (narg == 2){
		MPI_Comm_size(MPI_COMM_WORLD, &RankID);
	}
	else RankID = int(utils::numeric(FLERR, arg[2], false, lmp));

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;

	pi = 3.141592653589793;
	pi_sqrt = 1.772453850905516;
	// g_ewald = alpha_sqrt;

	K_All = new int[100][3];

	K_Sample = new int* [P];
	for (int i = 0; i < P; i++)
	{
		K_Sample[i] = new int[3];
	}

}

/* ---------------------------------------------------------------------- */

void Rbe::setup()
{

	int me;
	MPI_Comm_rank(world, &me);
	alpha = g_ewald * g_ewald;
	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	volume = xprd * yprd * zprd;

	S_X = sqrt(alpha * xprd * xprd / M_PI) * (1 + 2 * exp(-(alpha * xprd * xprd)));
	S_Y = sqrt(alpha * yprd * yprd / M_PI) * (1 + 2 * exp(-(alpha * yprd * yprd)));
	S_Z = sqrt(alpha * zprd * zprd / M_PI) * (1 + 2 * exp(-(alpha * zprd * zprd)));
	S = S_X * S_Y * S_Z - 1;
    Step = 0;
}

/* ---------------------------------------------------------------------- */

void Rbe::compute(int eflag, int vflag)
{

	double time;
	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime();

	ev_init(eflag, vflag);
	alpha = g_ewald * g_ewald;
	// srand(time);
	// if(me==0){
	// 	cout << "g_ewald: " << g_ewald << ' ' << "alpha: " << alpha << endl;
	// 	cout << "S: " << S << endl;
	// }

	// if atom count has changed, update qsum and qsqsum
	 
	if (atom->natoms != natoms_original) {
		qsum_qsq();
		natoms_original = atom->natoms;
	}

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	scale = 1;

	double V = xprd * yprd * zprd;
	float K[P][3];
	Step = update->ntimestep;

	int This_Index = Step * P;

	int this_rank = (Step % RankID);

	if (((Step % RankID)==0)&&(me< RankID)) {
		S1_X = sqrt(alpha * xprd * xprd / M_PI) * (1 + 2 * exp(-(alpha * xprd * xprd)));
		S1_Y = sqrt(alpha * yprd * yprd / M_PI) * (1 + 2 * exp(-(alpha * yprd * yprd)));
		S1_Z = sqrt(alpha * zprd * zprd / M_PI) * (1 + 2 * exp(-(alpha * zprd * zprd)));
		S0 = S1_X * S1_Y * S1_Z - 1;
		// if(me == 0 && Step % 1000 == 0){
		// 	cout << "alpha: " << alpha << endl;
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
			mx[0] = round(randn_box_muller_linear_congruential(0, sqrt(alpha * xprd * xprd / (2 * pi * pi))));
			my[0] = round(randn_box_muller_linear_congruential(0, sqrt(alpha * yprd * yprd / (2 * pi * pi))));
			mz[0] = round(randn_box_muller_linear_congruential(0, sqrt(alpha * zprd * zprd / (2 * pi * pi))));
		} while (mx[0] == 0 && my[0] == 0 && mz[0] == 0);
		
		
		
		for (int i = 0; i < 5*P-1; i++) 
		{
			double x = randn_box_muller_linear_congruential(0, sqrt(alpha * xprd * xprd / (2 * pi * pi)));
			double mold = mx[i];
			double mnew = round(x);
			double pup = Prob(mnew, alpha, xprd, S1_X, pi);
			double qup = MH_D(mold, alpha, xprd, pi);
			double pdown = Prob(mold, alpha, xprd, S1_X, pi);
			double qdown = MH_D(mnew, alpha, xprd, pi);
			double acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
			double y = (rand() % 10000 + 0.00) / 10000.0;
			if (y < acce) {
				mx[i + 1] = mnew;
			}
			else {
				mx[i + 1] = mold;
			}
			double xx = randn_box_muller_linear_congruential(0, sqrt(alpha * yprd * yprd / (2 * pi * pi)));
			mold = my[i];
			mnew = round(xx);
			pup = Prob(mnew, alpha, yprd, S1_Y, pi);
			qup = MH_D(mold, alpha, yprd, pi);
			pdown = Prob(mold, alpha, yprd, S1_Y, pi);
			qdown = MH_D(mnew, alpha, yprd, pi);
			acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
			double yy = (rand() % 10000 + 0.00) / 10000.0;
			if (yy < acce) {
				my[i + 1] = mnew;
			}
			else {
				my[i + 1] = mold;
			}
			double xxx = randn_box_muller_linear_congruential(0, sqrt(alpha * zprd * zprd / (2 * pi * pi)));
			mold = mz[i];
			mnew = round(xxx);
			pup = Prob(mnew, alpha, zprd, S1_Z, pi);
			qup = MH_D(mold, alpha, zprd, pi);
			pdown = Prob(mold, alpha, zprd, S1_Z, pi);
			qdown = MH_D(mnew, alpha, zprd, pi);
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
			K_Sample[i][0] = mx[5*i + 4] + 0.00;
			K_Sample[i][1] = my[5*i + 4] + 0.00;
			K_Sample[i][2] = mz[5*i + 4] + 0.00;
		}
	}

	if (comm->me == this_rank)
	{
		for (int i = 0; i < P; i++)
		{
			K[i][0] = K_Sample[i][0];
			K[i][1] = K_Sample[i][1];
			K[i][2] = K_Sample[i][2];
		}
	}

    /*if (comm->me == 0)
	{
	for (int i = 0; i < P; i++)
	{
		cout << "sample " << i << " =   " << K[i][0] << "    " << K[i][1] << "    " << K[i][2] << endl;
	}
	}*/

	MPI_Bcast((float*)K, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);

	// Step++;

	/*  Set Pointer */
	double** x = atom->x;
	double** f = atom->f;
	double* q = atom->q;
	int* type = atom->type;
	int nlocal = atom->nlocal;
	double qqrd2e = force->qqrd2e;
	double dielectric = force->dielectric;

	int tmp = int ( ceil ( ( nlocal + 0.0 ) / 16.0 ) ) * 16;

	float X[tmp][3]; float F[tmp][3];  float Q[tmp];

	for (int i = 0; i < tmp; i++)
	{
		if (i < nlocal) {
			X[i][0] = x[i][0]; X[i][1] = x[i][1]; X[i][2] = x[i][2];
			F[i][0] = 0.00; F[i][1] = 0.00; F[i][2] = 0.00;
			Q[i] = q[i];
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
	
	float K2;

	float Kx[P][3];
	float fac[P];

	for (int i = 0; i < P; i++) 
	{
		
		Kx[i][0] = K[i][0] * pxyz[0];
		Kx[i][1] = K[i][1] * pxyz[1];
		Kx[i][2] = K[i][2] * pxyz[2];
        float Kx0 = Kx[i][0] * xprd / xprd0;
        float Ky0 = Kx[i][1] * yprd / yprd0;
        float Kz0 = Kx[i][2] * zprd / zprd0;
		K2 = (Kx[i][0] * Kx[i][0] + Kx[i][1] * Kx[i][1] + Kx[i][2] * Kx[i][2]);
        float K2_0 = (Kx0 * Kx0 + Ky0 * Ky0 + Kz0 * Kz0);
		fac[i] = S0/S * gaussian(K2, alpha) / gaussian(K2_0, alpha);

		midterm[i][0] = Kx[i][0] / K2;
		midterm[i][1] = Kx[i][1] / K2;
		midterm[i][2] = Kx[i][2] / K2;
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
		std::vector<float> rho_all_cos(P), rho_all_sin(P);
		for (int ii = 0; ii < P; ++ii) {
			mterm[3*ii + 0] = midterm[ii][0];
			mterm[3*ii + 1] = midterm[ii][1];
			mterm[3*ii + 2] = midterm[ii][2];
			rho_all_cos[ii] = Rho_All[ii][0];
			rho_all_sin[ii] = Rho_All[ii][1];
		}
		std::vector<float> fx_gpu(nlocal), fy_gpu(nlocal), fz_gpu(nlocal);
		if (rbe_gpu_compute_force(nlocal, P, Xf.data(), Yf.data(), Zf.data(),
		                          Qf.data(), KxKx0, KxKx1, KxKx2,
		                          fac, mterm.data(),
		                          rho_all_cos.data(), rho_all_sin.data(),
		                          static_cast<float>(MIDTERM),
		                          fx_gpu.data(), fy_gpu.data(), fz_gpu.data()) == 0) {
			for (int ii = 0; ii < nlocal; ++ii) {
				F[ii][0] += fx_gpu[ii];
				F[ii][1] += fy_gpu[ii];
				F[ii][2] += fz_gpu[ii];
			}
			force_on_gpu = true;
		}
	}
#endif
	if (!force_on_gpu) {
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
			factor = _mm512_set1_ps(fac[j]);
			midterm_512 = midterm_512 * factor;
			Rho_All_0 = _mm512_set1_ps(Rho_All[j][0]);
			Rho_All_1 = _mm512_set1_ps(Rho_All[j][1]);
			Imag = (Cos * Rho_All_1 + Sin * Rho_All_0) * midterm_512;

			// if(me == 0 && i == 0 && j == 0) cout << "factor " << fac <<endl;
			
			midterm_0 = _mm512_set1_ps(midterm[j][0]);
			midterm_1 = _mm512_set1_ps(midterm[j][1]);
			midterm_2 = _mm512_set1_ps(midterm[j][2]);
			Fx = Fx + qq * Imag * midterm_0;
			Fy = Fy + qq * Imag * midterm_1;
			Fz = Fz + qq * Imag * midterm_2;

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
		f[i][0] += F[i][0];
		f[i][1] += F[i][1]; 
		f[i][2] += F[i][2];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;
	if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
		TimeSet[update->ntimestep - 0] = time * 1000;
	}

	if (comm->me == 0 && update->ntimestep == 1000) {
		ofstream outfile;
		outfile.open("./Time_Kspace_RBE.txt");
		//outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
		for (int i = 0; i < 1000; i++)
			outfile << TimeSet[i] << endl;
		outfile.close();
	}


	// if(me == 0){
	// 	cout << "f[0]: " << f[0][0] << " " << f[0][1] << ' ' << f[0][2] << endl;
	// }
	
		
	// sum global energy across Kspace vevs and add in volume-dependent term
		const double qscale = qqrd2e * scale;

        if (eflag_global) {
			float KXX[3];
			for (int i = 0; i < P; i++)
			{
				KXX[0] = K[i][0] * pxyz[0];
				KXX[1] = K[i][1] * pxyz[1];
				KXX[2] = K[i][2] * pxyz[2];

				energy += fac[i] * (2 * pi * S0 / ((P + 0.00) * V)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (KXX[0] * KXX[0] + KXX[1] * KXX[1] + KXX[2] * KXX[2]);
			}
			energy -= g_ewald * qsqsum / MY_PIS +
				MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);// self energy
			energy *= qscale;
			// if(me == 0){cout << "energy: " << energy << endl;}
		}


// global virial
		if (vflag_global) {
			float KKXX[3],Moment_Term= 2 * pi * (S0 / (P + 0.00)) * qqrd2e / (V+0.00);

#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
			for (int i = 0; i < P; i++)
			{
				KKXX[0] = K[i][0] * pxyz[0];
				KKXX[1] = K[i][1] * pxyz[1]; 
				KKXX[2] = K[i][2] * pxyz[2];

				float coef1 = fac[i] * (Rho_All[i][0]* Rho_All[i][0]+ Rho_All[i][1]* Rho_All[i][1])/ (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				float coef2 = (1.0 / (4 * g_ewald * g_ewald)) + 1.0 / (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				virial[0] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[0] * KKXX[0]);
				virial[1] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[1] * KKXX[1]);
				virial[2] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[2] * KKXX[2]);

				virial[3] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[1]);
				virial[4] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[2]);
				virial[5] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[1] * KKXX[2]);
			}
			/*if(comm->me == 0){
			cout << "virial: " << virial[0] << " " << virial[1] << " " << virial[2] << " " << virial[3] << " " 
			<< virial[4] << " " << virial[5] << endl;
			}*/
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
					eatom[i] -= g_ewald * q[i] * q[i] / sqrt(pi) + pi*pi * q[i] * qsum /
						(g_ewald * g_ewald * volume);
					eatom[i] *= qscale;
				}
			}

		if (vflag_atom)
#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
				for (int i = 0; i < nlocal; i++)
					for (int j = 0; j < 6; j++) vatom[i][j] *= q[i] * qscale;
		}
    
	if (slabflag == 1) slabcorr();
}

/* ---------------------------------------------------------------------- */

Rbe::~Rbe()
{
	for (int i = 0; i < P; i++)
		delete[] K_Sample[i];
	delete[] K_Sample;
}

/* ---------------------------------------------------------------------- */

double Rbe::memory_usage()
{
	return 1.0;
}

/* ---------------------------------------------------------------------- */

void Rbe::slabcorr()
{
	// compute local contribution to global dipole moment
	qqrd2e = force->qqrd2e;
	scale = 1;

	double* q = atom->q;
	double** x = atom->x;
	double zprd = domain->zprd;
	int nlocal = atom->nlocal;

	double dipole = 0.0;
	for (int i = 0; i < nlocal; i++) dipole += q[i] * x[i][2];

	// sum local contributions to get global dipole moment

	double dipole_all;
	MPI_Allreduce(&dipole, &dipole_all, 1, MPI_DOUBLE, MPI_SUM, world);

	// need to make non-neutral systems and/or
	//  per-atom energy translationally invariant

	double dipole_r2 = 0.0;
	if (eflag_atom || fabs(qsum) > SMALL) {
		for (int i = 0; i < nlocal; i++)
			dipole_r2 += q[i] * x[i][2] * x[i][2];

		// sum local contributions

		double tmp;
		MPI_Allreduce(&dipole_r2, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
		dipole_r2 = tmp;
	}

	// compute corrections

	const double e_slabcorr = MY_2PI * (dipole_all * dipole_all -
		qsum * dipole_r2 - qsum * qsum * zprd * zprd / 12.0) / volume;
	const double qscale = qqrd2e * scale;

	if (eflag_global) energy += qscale * e_slabcorr;


	// per-atom energy

	// add on force corrections

	double ffact = qscale * (-4.0 * MY_PI / volume);
	double** f = atom->f;

	for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum * x[i][2]);

}