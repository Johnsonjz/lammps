#include "./rbe_disp.h"
#include <mpi.h>
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
#include <cmath>
 
#include <immintrin.h> 
#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef>   
#include <vector>

//#include"sse_mathfun.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

struct t_complex
{
	float im, re;
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

double RbeDisp::randn_box_muller_linear_congruential1(const double Mean, const double SquareMargin)
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


inline double RbeDisp::Prob(double m, double alpha, double L, double S1, double pi)
{
	return exp(-(2 * pi * m / L) * (2 * pi * m / L) / (4 * alpha)) / S1;
}


double RbeDisp::MH_D(double m, double alpha, double L, double pi)
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

RbeDisp::RbeDisp(LAMMPS* lmp) : KSpace(lmp) 
{
	ewaldflag = dipoleflag = pppmflag = 1;
    for(int i = 0; i < 2; i++){
        function[i]=1;
    }
    nfunctions = 2;
    q2 = 0;
    mu2 = 0;
}
 
void RbeDisp::init()
{
	//int me;
	//MPI_Comm_rank(world,&me);
    if (!comm->me) utils::logmesg(lmp, "RbeDisp initialization ...\n");
	triclinic_check();
	if (domain->dimension == 2)
		error->all(FLERR, "Cannot use Ewald with 2d simulation");

	if (!atom->q_flag) error->all(FLERR, "Kspace style requires atom attribute q");

	if (slabflag == 0 && domain->nonperiodic > 0)
		error->all(FLERR, "Cannot use non-periodic boundaries with Ewald");
	if (slabflag) {
		if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
			domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
			error->all(FLERR, "Incorrect boundaries with slab Ewald");
		if (domain->triclinic)
			error->all(FLERR, "Cannot (yet) use Ewald with triclinic box "
				"and slab correction");
	}

	// extract short-range Coulombic cutoff from pair style
	dipoleflag = atom->mu?1:0;
	// cout<<"dipoleflag = " << dipoleflag << endl;
	int triclinic = domain->triclinic;
	pair_check();
    scale = 1.0;
    mumurd2e = force->qqrd2e;

	int itmp;
	double* p_cutoff = (double*)force->pair->extract("cut_coul", itmp);
	if (p_cutoff == NULL)
		error->all(FLERR, "KSpace style is incompatible with Pair style");
	double cutoff = *p_cutoff;

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	if (slabflag == 1 && me == 0)utils::logmesg(lmp, "This is a 3D simulation with slab correction ...\n");
	volume = xprd * yprd * zprd;
	//cout << volume << endl;
	// if (me == 0)
	// {
	// 	cout << "cutoff==" << cutoff << endl;
	// 	cout << "The dimensions of domain are " << xprd << "   " << yprd << "   " << domain->zprd << "    " << zprd << endl;
	// 	cout << "pi==" << MY_PI << endl;
	// 	cout << "If compute total energy?" << eflag_global << endl;
	// }

	S1_X = 0.00;
	S1_Y = 0.00;
	S1_Z = 0.00;
	for (int i = -10000; i <= 10000; i++)
	{
		S1_X = S1_X + exp(-pi * pi * i * i / (alpha * xprd * xprd));
	}
	for (int i = -10000; i <= 10000; i++)
	{
		S1_Y = S1_Y + exp(-pi * pi * i * i / (alpha * yprd * yprd));
	}
	for (int i = -10000; i <= 10000; i++)
	{
		S1_Z = S1_Z + exp(-pi * pi * i * i / (alpha * zprd * zprd));
	}
	S = S1_X * S1_Y * S1_Z - 1;

	Step = 0;


	qsum_qsq();
    q2 = qsqsum * force->qqrd2e;
    if (function[0] && qsqsum == 0.0) {
        function[0] = 0;
        nfunctions -= 1;
    }
    if (fabs(qsum) > SMALL && comm->me == 0){
        error->warning(FLERR, "System is not charge neutral, net charge = {:.8g}", qsum);
    }

    
    musum_musq();
    if(musqsum == 0){
        function[1] = 0;
        nfunctions -= 1;
    }
	if(nfunctions == 0){
		error->all(FLERR,"Cannot use rbe/disp solver on system without charged, dipole particles");
	}
	if(!comm->me){cout << "function0: " << function[0] << " ; " << "function1: " << function[1]<<endl;}
	setup();

	Time = new float[2000];
}

/* ---------------------------------------------------------------------- */

void RbeDisp::settings(int narg, char** arg)
{
	MPI_Comm_rank(world, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &RankID);
	RankID = 1;

	if ((narg != 2) && (narg!=3)) error->all(FLERR, "Illegal kspace_style RBE command");
	// accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));
	// if (accuracy_relative > 1.0)
    // error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}",
    //            accuracy_relative, force->kspace_style);
	g_ewald = utils::numeric(FLERR, arg[0], false, lmp);
	P = int(utils::numeric(FLERR, arg[1], false, lmp));
	RankID = int(utils::numeric(FLERR, arg[2], false, lmp));

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	// if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;

	alpha = g_ewald * g_ewald;
	pi = 3.141592653589793;
	pi_sqrt = 1.772453850905516;

	K_All = new int[100][3];

	K_Sample = new float* [P];
	for (int i = 0; i < P; i++)
	{
		K_Sample[i] = new float[3];
	}

	// if (me == 0)
	// {
	// 	cout << xprd << "=Lx   " << yprd << "=Ly    " << zprd << "=Lz   " << alpha << "=alpha   " << atom->natoms << "=N    " << P << "=P   " << endl;
	// }
}

/* ---------------------------------------------------------------------- */

void RbeDisp::setup()
{
	int me;
	MPI_Comm_rank(world, &me);

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	// if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;
	volume = xprd * yprd * zprd;

	S1_X = 0.00;
	S1_Y = 0.00;
	S1_Z = 0.00;
	
	S1_X = sqrt(alpha * xprd * xprd / M_PI) * (1 + 2 * exp(-(alpha * xprd * xprd)));
	S1_Y = sqrt(alpha * yprd * yprd / M_PI) * (1 + 2 * exp(-(alpha * yprd * yprd)));
	S1_Z = sqrt(alpha * zprd * zprd / M_PI) * (1 + 2 * exp(-(alpha * zprd * zprd)));
	S = S1_X * S1_Y * S1_Z - 1;

	if (me == 0)
	{
		//cout << S << "    " << S1_X * S1_Y * S1_Z - 1 << "   " << endl;
	}
}

/* ---------------------------------------------------------------------- */

void RbeDisp::compute(int eflag, int vflag)
{
	// cout << "function0: " << function[0] << ";" << "function1: " << function[1]<<endl;
	double time; 
	//time = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	//time = MPI_Wtime()-time;
	time = MPI_Wtime();

	ev_init(eflag, vflag);

	// if (Step == 0)
	// 	srand(comm->me * 42);//����Ҫ ��Ȼ����CPU���������Ƶ�����

	//if (me == 0 && Step % 1000 == 0)cout << "The total time of wait is " << time * 1000 << "   ms" << endl;
	//time = MPI_Wtime();

	// if atom count has changed, update qsum and qsqsum
	 
	if (atom->natoms != natoms_original) {
		qsum_qsq();
		musum_musq();
		natoms_original = atom->natoms;
	}

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	//if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;
	scale = 1;

	// if (me == 0&&Step==0) {
	// 	cout << "slab_factor = " << slab_volfactor << "    slabflag = " << slabflag << "   eflag_atom = " << eflag_atom << "    SMALL = " << SMALL << "     qsum" << qsum << endl;
	// 	cout << xprd << "   " << yprd << "   " << zprd << endl; 
	// }

	double V = xprd * yprd * zprd;
	float K[P][3];

	int This_Index = Step * P;

	int this_rank = (Step % RankID);

	if (((Step % RankID)==0)&&(me< RankID)) {
		int mx[5*P], my[5*P], mz[5*P];
		do {
			mx[0] = round(randn_box_muller_linear_congruential1(0, sqrt(alpha * xprd * xprd / (2 * pi * pi))));
			my[0] = round(randn_box_muller_linear_congruential1(0, sqrt(alpha * yprd * yprd / (2 * pi * pi))));
			mz[0] = round(randn_box_muller_linear_congruential1(0, sqrt(alpha * zprd * zprd / (2 * pi * pi))));
		} while (mx[0] == 0 && my[0] == 0 && mz[0] == 0);
		for (int i = 0; i < 5*P-1; i++) 
		{
			double x = randn_box_muller_linear_congruential1(0, sqrt(alpha * xprd * xprd / (2 * pi * pi)));
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
			double xx = randn_box_muller_linear_congruential1(0, sqrt(alpha * yprd * yprd / (2 * pi * pi)));
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
			double xxx = randn_box_muller_linear_congruential1(0, sqrt(alpha * zprd * zprd / (2 * pi * pi)));
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

	MPI_Bcast((float*)K, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);
	// cout<<"me= "<< me << "; " << "K: " << K[10][0] << " " << K[10][1] << " " << K[10][2] << " "<<endl;

	Step++;

	/*  Set Pointer */
	double** x = atom->x;//atom->xָ��������ӵ�λ��
	double** f = atom->f;//atom->fָ����Ǵ洢��������
	double* q = atom->q;//atom->qָ��洢������������
	double **mu=atom->mu ? atom->mu : nullptr;//atom->mu dipole of per atom
	double **t=atom->torque ? atom->torque : nullptr;//torque of dipole
	// cout << "mu1: " << mu[0][0] << ", " << mu[0][1] << ", " << mu[0][2] << endl;
	// cout << "t1: " << t[0][0] << ", " << t[0][1] << ", " << t[0][2] << endl;
    // if(function[1]){mu = atom->mu;}
	// if (atom->torque) t = atom->torque;
	int* type = atom->type;//ԭ�ӵ�����
	int nlocal = atom->nlocal;//��ǰmpi�ϴ洢��ԭ������
	double qqrd2e = force->qqrd2e;//����һ��ϵ��
	double dielectric = force->dielectric;//��糣��

	double pxyz[3] = { 2 * pi / xprd,2 * pi / yprd,2 * pi / zprd };
	float Rho[P][2], Rho_All[P][2];
	float midterm[P][3];
        
	float K2;

	float Kx[P][3];

	for (int i = 0; i < P; i++) 
	{
		Kx[i][0] = K[i][0] * pxyz[0];
		Kx[i][1] = K[i][1] * pxyz[1];
		Kx[i][2] = K[i][2] * pxyz[2];

		K2 = 1 / (Kx[i][0] * Kx[i][0] + Kx[i][1] * Kx[i][1] + Kx[i][2] * Kx[i][2]);//����һ�γ˷����Ƕ�γ���
		//��λ��
		midterm[i][0] = Kx[i][0] * K2;
		midterm[i][1] = Kx[i][1] * K2;
		midterm[i][2] = Kx[i][2] * K2;
	}

	float kx, ky, kz, moment, Sin, Cos, Sin1, Cos1;
	// float muk;

	
	time = MPI_Wtime();
	// cout << "function0: " << function[0] << ";" << "function1: " << function[1]<<endl;
	// cout << " S/P: " << S/P << endl;
	// cout << "mumurd2e == qqrd2e: " << bool(mumurd2e == qqrd2e) << endl;

	for (int i = 0; i < P; i++)
	{
		kx = Kx[i][0];
		ky = Kx[i][1];
		kz = Kx[i][2];
		float Real=0.00, Image = 0.00;
		for (int j = 0; j < nlocal; j++)
		{
			moment = kx * x[j][0] + ky * x[j][1] + kz * x[j][2];
			// muk = kx * mu[j][0] + ky * mu[j][1] + kz * mu[j][2];
			sincosf(moment,&Sin,&Cos);
			if(function[0]){
				Real = Real + q[j] * Cos;
				Image = Image + q[j] * Sin;
			}
			if(function[1]){
				double muk = kx * mu[j][0] + ky * mu[j][1] + kz * mu[j][2];
				Real += -muk * Sin;
				Image += muk * Cos;
			}
			
		}
		Rho[i][0] = Real;
		Rho[i][1] = Image;
	}


	time = MPI_Wtime() - time;  // ��ֹ��ʱ
	//cout << "The time for the first part I is " << time << endl;
	time = MPI_Wtime();

	
	time = MPI_Wtime() - time;  // ��ֹ��ʱ
	//cout << "The time for the first part is " << time << endl;

	MPI_Allreduce((float*)Rho, (float*)Rho_All, 2 * P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	float MIDTERM = -4 * pi * (S / (P + 0.00)) * qqrd2e / (V);
	float Im, Re;
	float Im_mu;

	for (int i = 0; i < nlocal; i++)
	{
		float x1 = x[i][0], y1 = x[i][1], z1 = x[i][2], qq=q[i];
		float mu0, mu1, mu2;
		float Fx=0.00, Fy=0.00, Fz=0.00;
		// if(function[1]){
		// 	mu0 = mu[i][0], mu1 = mu[i][1], mu2 = mu[i][2];
		// }
		// cout << "force_before: " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
    	// cout << "torque_before: " << t[i][0] << " " << t[i][1] << " " << t[i][2] << endl;
		for (int j = 0; j < P; j++)
		{
			moment = -(Kx[j][0] * x1 + Kx[j][1] * y1 + Kx[j][2] * z1);
			sincosf(moment, &Sin1, &Cos1);
			Im = (Cos1*Rho_All[j][1]+Sin1*Rho_All[j][0]) * MIDTERM;
			Re = (Cos1*Rho_All[j][0]-Sin1*Rho_All[j][1]) * MIDTERM;
			if(function[0]){
				Fx = Fx + qq * Im * midterm[j][0];
				Fy = Fy + qq * Im * midterm[j][1];
				Fz = Fz + qq * Im * midterm[j][2];
			}
			if(function[1]){
				mu0 = mu[i][0], mu1 = mu[i][1], mu2 = mu[i][2];
				double muk_minus = -(Kx[j][0] * mu0 + Kx[j][1] * mu1 + Kx[j][2] * mu2);
				Im_mu = muk_minus * Re;
				Fx = Fx + Im_mu * midterm[j][0];
				Fy = Fy + Im_mu * midterm[j][1];
				Fz = Fz + Im_mu * midterm[j][2];
				t[i][0] += (mu1 * midterm[j][2] - mu2 * midterm[j][1])*Im;
				t[i][1] += (mu2 * midterm[j][0] - mu0 * midterm[j][2])*Im;
				t[i][2] += (mu0 * midterm[j][1] - mu1 * midterm[j][0])*Im;
			}
		}

		f[i][0] += Fx;
		f[i][1] += Fy;
		f[i][2] += Fz;
		
		// cout << "elec_force: " << Fx << " " << Fy << " " << Fz << endl;
		// cout << "force: " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
    	// cout << "torque: " << t[i][0] << " " << t[i][1] << " " << t[i][2] << endl;
	}
	if(me == 0){
		cout << "force "  << f[0][0] << " " << f[0][1] << " " << f[0][2] << endl;
	}

// sum global energy across Kspace vevs and add in volume-dependent term
		const double qscale = qqrd2e * scale;
		double energy1 = 0, energy_self = 0;
		
		// cout << "qsqsum = " << qsqsum << endl;
		// cout << "musqsum = " << musqsum << endl;
        
		if (eflag_global) {
			float KXX[3];
			for (int i = 0; i < P; i++)
			{
				KXX[0] = K[i][0] * pxyz[0];
				KXX[1] = K[i][1] * pxyz[1];
				KXX[2] = K[i][2] * pxyz[2];

				energy1 += (2 * pi * S / ((P + 0.00) * volume)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (KXX[0] * KXX[0] + KXX[1] * KXX[1] + KXX[2] * KXX[2]);
			}
			if(me == 0) cout << "energy1: " << energy1 << endl;
			energy1 *= qscale;
			if(function[0]){
				energy_self += g_ewald * qsqsum * qscale / MY_PIS + 0.5 * MY_PI * qscale * qsum * qsum / (g_ewald * g_ewald * volume); 
			}
			if(function[1]){
				energy_self += 2.0/3.0 /MY_PIS * mumurd2e * musqsum * (g_ewald * g_ewald * g_ewald);
			}
			// cout<< "energy_self: "<< energy_self <<endl;
			energy = energy1 - energy_self;
			// cout<< "energy "<< energy <<endl;
			// cout << "energy2: " << -qscale * g_ewald * qsqsum / MY_PIS + MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume) << endl;
		}

		//vflag_global = 1;
// global virial
		if (vflag_global) {
			float KKXX[3], Moment_Term= 2 * pi * (S / (P + 0.00)) * qqrd2e / (V+0.00);

			float virial_rbe[6];
			for (int i = 0; i < 6; i++)
				virial_rbe[i] = virial[i];
#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
			for (int i = 0; i < P; i++)
			{
				KKXX[0] = K[i][0] * pxyz[0];
				KKXX[1] = K[i][1] * pxyz[1]; 
				KKXX[2] = K[i][2] * pxyz[2];

				float coef1 = (Rho_All[i][0]* Rho_All[i][0]+ Rho_All[i][1]* Rho_All[i][1])/ (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				float coef2 = (1.0 / (4 * g_ewald * g_ewald)) + 1.0 / (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				virial[0] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[0] * KKXX[0]);
				virial[1] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[1] * KKXX[1]);
				virial[2] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[2] * KKXX[2]);

				virial[3] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[1]);
				virial[4] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[2]);
				virial[5] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[1] * KKXX[2]);
			}
			if(me == 0){
				cout << "after calculation" << endl;
				cout<< "virial:" <<virial[0] << endl;
				cout<< "virial:" <<virial[1] << endl;
				cout<< "virial:" <<virial[2] << endl;
				cout<< "virial:" <<virial[3] << endl;
				cout<< "virial:" <<virial[4] << endl;
				cout<< "virial:" <<virial[5] << endl;
			}

			for (int i = 0; i < 6; i++)
				virial_rbe[i] = virial[i]- virial_rbe[i];
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
	time = MPI_Wtime() - time;  // ��ֹ��ʱ

	if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
		Time[Step - 1] = time * 1000;
	}

	// if (me == 0 && update->ntimestep == 1000) {
	// 	ofstream outfile;
	// 	outfile.open("./Time.txt");
	// 	for (int i = 0; i < 1000; i++)
	// 		outfile << Time[i] << endl;
	// 	outfile.close();
	// }
}

/* ---------------------------------------------------------------------- */

RbeDisp::~RbeDisp()
{
	

}

/* ---------------------------------------------------------------------- */

double RbeDisp::memory_usage()
{
	return 1.0;
}

/* ---------------------------------------------------------------------- */

void RbeDisp::slabcorr()
{
	// compute local contribution to global dipole moment
	qqrd2e = force->qqrd2e;
	scale = 1;
	//cout << "qsum == " << qsum <<"   "<< -qsum / (Lx * Ly)<< endl;

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

	// add on force corrections

	double ffact = qscale * (-4.0 * MY_PI / volume);
	double** f = atom->f;

	for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum * x[i][2]);
}

void RbeDisp::musum_musq()
{
  const int nlocal = atom->nlocal;

  musum = musqsum = mu2 = 0.0;
  if (atom->mu_flag) {
    double** mu = atom->mu;
    double musum_local(0.0), musqsum_local(0.0);

    for (int i = 0; i < nlocal; i++) {
      musum_local += mu[i][0] + mu[i][1] + mu[i][2];
      musqsum_local += mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2];
    }

    MPI_Allreduce(&musum_local,&musum,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&musqsum_local,&musqsum,1,MPI_DOUBLE,MPI_SUM,world);

    mu2 = musqsum * force->qqrd2e;
	
  }

  if (mu2 == 0 && comm->me == 0)
    utils::logmesg(lmp,"Using kspace solver RbeDipole on system with no dipoles ...\n");
}