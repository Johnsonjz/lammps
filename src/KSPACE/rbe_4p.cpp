#include "./rbe_4p.h"
#include <mpi.h>
#include "atom.h"
#include "comm.h"
#include "angle.h"
#include "bond.h"
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
// #include <mkl.h> 
 
// #include <immintrin.h> 
#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef>   
#include <vector>
#include<numeric>

//#include"sse_mathfun.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;


double RBE4P::randn_box_muller_linear_congruential1(const double Mean, const double SquareMargin)
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


inline double RBE4P::Prob(double m, double alpha, double L, double S1, double pi)
{
	return exp(-(2 * pi * m / L) * (2 * pi * m / L) / (4 * alpha)) / S1;
}


double RBE4P::MH_D(double m, double alpha, double L, double pi)
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

RBE4P::RBE4P(LAMMPS* lmp) : KSpace(lmp)
{
    // triclinic_support = 1;
    tip4pflag = 1;
	rbeflag = 1;
	// triclinic = domain->triclinic;
}

void RBE4P::init()
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

    TimeSet = new double[1000];

	if (me == 0) {
    	std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    	utils::logmesg(lmp,mesg);
  	}
}

void RBE4P::settings(int narg, char** arg)
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

void RBE4P::setup()
{
	// int me;
	MPI_Comm_rank(world, &me);

	double xprd = domain->xprd;
	double yprd = domain->yprd;
	double zprd = domain->zprd * slab_volfactor;
	// if (slabflag == 1 && me == 0)cout << "This is a 3D simulation with slab correction" << endl;
	volume = xprd * yprd * zprd;
	beta = g_ewald * g_ewald;
	S1_X = 0.00;
	S1_Y = 0.00;
	S1_Z = 0.00;
	
	S1_X = sqrt(beta * xprd * xprd / M_PI) * (1 + 2 * exp(-(beta * xprd * xprd)));
	S1_Y = sqrt(beta * yprd * yprd / M_PI) * (1 + 2 * exp(-(beta * yprd * yprd)));
	S1_Z = sqrt(beta * zprd * zprd / M_PI) * (1 + 2 * exp(-(beta * zprd * zprd)));
	S = S1_X * S1_Y * S1_Z - 1;

	// if (me == 0)
	// {
	// 	cout << xprd << "=Lx   " << yprd << "=Ly    " << zprd << "=Lz   " << beta << "=beta   " << atom->natoms << "=N    " << P << "=P   " << endl;
	// }
}

void RBE4P::compute(int eflag, int vflag)
{
	 double time; 
	 MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime();

	ev_init(eflag, vflag);

	 
	if (atom->natoms != natoms_original) {
		qsum_qsq();
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
	// cout<<"me= "<< me << "; " << "K: " << K[10][0] << " " << K[10][1] << " " << K[10][2] << " "<<endl;

	Step++;

	/*  Set Pointer */
	double** x = atom->x;//atom->xָ        ӵ λ  
	double** f = atom->f;//atom->fָ    Ǵ洢        
	double* q = atom->q;//atom->qָ  洢            
	double **mu=atom->mu ? atom->mu : nullptr;//atom->mu dipole of per atom
	double **t=atom->torque ? atom->torque : nullptr;//torque of dipole
	int* type = atom->type;//ԭ ӵ     
	int nlocal = atom->nlocal;//  ǰmpi ϴ洢  ԭ      
	double qqrd2e = force->qqrd2e;//    һ  ϵ  
	double dielectric = force->dielectric;//  糣  

	double pxyz[3] = { 2 * pi / xprd,2 * pi / yprd,2 * pi / zprd };
	// if(me == 0) cout << "pxyz " << pxyz[0] << ' '<<pxyz[1] << ' '<< pxyz[2] <<endl;
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

		K2 = 1 / (Kx[i][0] * Kx[i][0] + Kx[i][1] * Kx[i][1] + Kx[i][2] * Kx[i][2]);
		fac[i] = S0/S * (Prob(K[i][0], beta, xprd, 1, pi)/Prob(K[i][0], beta, xprd0, 1, pi)) * 
						(Prob(K[i][1], beta, yprd, 1, pi)/Prob(K[i][1], beta, yprd0, 1, pi)) * 
						(Prob(K[i][2], beta, zprd, 1, pi)/Prob(K[i][2], beta, zprd0, 1, pi));

		midterm[i][0] = Kx[i][0] * K2;
		midterm[i][1] = Kx[i][1] * K2;
		midterm[i][2] = Kx[i][2] * K2;
	}

	float kx, ky, kz, moment, Sin, Cos, Sin1, Cos1;
	double *xi,xM[3];
	int iH1 = 0, iH2 = 0;
	// float muk;

	
	// time = MPI_Wtime();
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
			if (type[j] == typeO) {
				find_M(j,iH1,iH2,xM);
				xi = xM;
				// if(me == 0 && i <= 4){
				// 	cout << "O position " << x[i][0] << " " << x[i][1] << " " << x[i][2]<<endl;
				// 	cout << "M site " << xM[0] << " " << xM[1] << " " << xM[2]<<endl;
				// }
			} else xi = x[j];
			moment = kx * xi[0] + ky * xi[1] + kz * xi[2];
			// muk = kx * mu[j][0] + ky * mu[j][1] + kz * mu[j][2];
			sincosf(moment,&Sin,&Cos);
			Real = Real + q[j] * Cos;
			Image = Image + q[j] * Sin;
		}
		Rho[i][0] = Real;
		Rho[i][1] = Image;
	}
	// if(me == 0){
	// 	cout << "rho "  << Rho[0][0] << " " << Rho[0][1] << endl;
	// }


	// time = MPI_Wtime() - time;  //   ֹ  ʱ
	//cout << "The time for the first part I is " << time << endl;
	// time = MPI_Wtime();

	
	// time = MPI_Wtime() - time;  //   ֹ  ʱ
	//cout << "The time for the first part is " << time << endl;

	MPI_Allreduce((float*)Rho, (float*)Rho_All, 2 * P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	float MIDTERM = -4 * pi * (S0 / (P + 0.00)) * qqrd2e / (V); //Summation constant
	float Imag;

	for (int i = 0; i < nlocal; i++)
	{
		if (type[i] == typeO) {
				find_M(i,iH1,iH2,xM);
				xi = xM;
				// if(me == 0 && i <= 4){
				// 	cout << "O position " << x[i][0] << " " << x[i][1] << " " << x[i][2]<<endl;
				// 	cout << "M site " << xM[0] << " " << xM[1] << " " << xM[2]<<endl;
				// }
		} else xi = x[i];
		float x1 = xi[0], y1 = xi[1], z1 = xi[2], qq=q[i];
		float Fx=0.00, Fy=0.00, Fz=0.0;

		for (int j = 0; j < P; j++)
		{
			moment = -(Kx[j][0] * x1 + Kx[j][1] * y1 + Kx[j][2] * z1);
			sincosf(moment, &Sin, &Cos);
			Imag = (Cos*Rho_All[j][1]+Sin*Rho_All[j][0]) * MIDTERM;
			Fx = Fx + qq * Imag * midterm[j][0] * fac[j];
			Fy = Fy + qq * Imag * midterm[j][1] * fac[j];
			Fz = Fz + qq * Imag * midterm[j][2] * fac[j];
		}

		// f[i][0] += Fx;
		// f[i][1] += Fy;
		// f[i][2] += Fz;
		if (type[i] != typeO) {
      		f[i][0] += Fx;
			f[i][1] += Fy; 
			f[i][2] += Fz;

    	} else {
			find_M(i,iH1,iH2,xM);

			f[i][0] += Fx*(1 - alpha);
			f[i][1] += Fy*(1 - alpha);
			if (slabflag != 2) f[i][2] += Fz*(1 - alpha);

			f[iH1][0] += 0.5*alpha*Fx;
			f[iH1][1] += 0.5*alpha*Fy;
			if (slabflag != 2) f[iH1][2] += 0.5*alpha*Fz;

			f[iH2][0] += 0.5*alpha*Fx;
			f[iH2][1] += 0.5*alpha*Fy;
			if (slabflag != 2) f[iH2][2] += 0.5*alpha*Fz;
		}
	}
		
		// cout << "elec_force: " << Fx << " " << Fy << " " << Fz << endl;
		// cout << "force: " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
    	// cout << "torque: " << t[i][0] << " " << t[i][1] << " " << t[i][2] << endl;
	
	// if(me == 0){
	// 	cout << "force "  << f[0][0] << " " << f[0][1] << " " << f[0][2] << endl;
	// }
time = MPI_Wtime() - time;

    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
    }

    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_RBE_Sca.txt");
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
    }
double sum = accumulate(TimeSet, TimeSet+1000, 0.0);  
double mean =  sum / 1000; //均值  
    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Mean_Kspace.txt");
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

				energy1 += fac[i] * (2 * pi * S0 / ((P + 0.00) * V)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (KXX[0] * KXX[0] + KXX[1] * KXX[1] + KXX[2] * KXX[2]);
			}
			// if(me==0)cout << "energy1: " << energy1 << endl;
			energy = energy1;
			double energy_self = g_ewald * qsqsum / MY_PIS + MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);
			energy -= g_ewald * qsqsum / MY_PIS +
				MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);
			energy *= qscale;

		}

		//vflag_global = 1;
// global virial
		if (vflag_global) {
			float KKXX[3], Moment_Term= 2 * pi * (S0 / (P + 0.00)) * qqrd2e / (V+0.00);

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

				float coef1 = fac[i] * (Rho_All[i][0]* Rho_All[i][0]+ Rho_All[i][1]* Rho_All[i][1])/ (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				float coef2 = (1.0 / (4 * g_ewald * g_ewald)) + 1.0 / (KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2]);
				virial[0] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[0] * KKXX[0]);
				virial[1] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[1] * KKXX[1]);
				virial[2] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[2] * KKXX[2]);

				virial[3] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[1]);
				virial[4] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[2]);
				virial[5] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[1] * KKXX[2]);
			}
			// if(me == 0){
			// 	cout << "after calculation" << endl;
			// 	cout<< "virial:" <<virial[0] << endl;
			// 	cout<< "virial:" <<virial[1] << endl;
			// 	cout<< "virial:" <<virial[2] << endl;
			// 	cout<< "virial:" <<virial[3] << endl;
			// 	cout<< "virial:" <<virial[4] << endl;
			// 	cout<< "virial:" <<virial[5] << endl;
			// }

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
		
// 			if (vflag_atom){
// #if defined(LMP_SIMD_COMPILER)
// #pragma vector aligned
// #pragma simd
// #endif
// 				for (int i = 0; i < nlocal; i++){
					
// 					for (int j = 0; j < 6; j++){
// 						if (type[i] != typeO && type[i] != typeH) { vatom[i][j] *= q[i] * qscale;
// 						}
// 						if (type[i] == typeO) {
// 							find_M(i,iH1,iH2,xM);
// 							double v0 = vatom[i][j];
// 							vatom[i][j] *= q[i] * qscale * (1-alpha);
// 							vatom[iH1][j] = vatom[iH1][j]* q[iH1] * qscale + v0 * q[i] * qscale * 0.5 * alpha;
// 							vatom[iH1][j] = vatom[iH2][j]* q[iH2] * qscale + v0 * q[i] * qscale * 0.5 * alpha;
// 						}
// 					}
					
				
// 				}
					
// 			}
		}


	if (slabflag == 1) slabcorr();

	// if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
	// 	Time[Step - 1] = time * 1000;
	// }

	// if (me == 0 && update->ntimestep == 1000) {
	// 	ofstream outfile;
	// 	outfile.open("./Time.txt");
	// 	for (int i = 0; i < 1000; i++)
	// 		outfile << Time[i] << endl;
	// 	outfile.close();
	// }
}

/* ---------------------------------------------------------------------- */

RBE4P::~RBE4P()
{
	

}

/* ---------------------------------------------------------------------- */

double RBE4P::memory_usage()
{
	return 1.0;
}

/* ---------------------------------------------------------------------- */

void RBE4P::slabcorr()
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

void RBE4P::find_M(int i, int &iH1, int &iH2, double *xM)
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
