/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Jiuyang Liang (SJTU)
------------------------------------------------------------------------- */

#include "DRBSOG.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "math_const.h"
#include "modify.h"
#include "memory.h"
#include "update.h"
#include "pair.h" 
#include <immintrin.h> 
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst; 

#define SMALL 0.00001
#define MY_IPIS 1.0/MY_PIS
#define MY_PI3S MY_PI*MY_PIS

extern double randn_box_muller_linear_congruential(const double, const double);

inline double f_m(double r, double t, double lambda)
{
	double a = exp(t);
	double Gaussian = MY_IPIS * exp(-r * r * a - 1.0 / (4 * lambda * lambda * a) + t / 2.0);
	return Gaussian;
}

double GaussFourier(double h, double t_0, double w_M2, int M1, int M2, double lambda, double k2)//注意这里是k2
{
	double factor = MY_PI3S * h;
	double Gaussian = 0.00;
	for (int m = -M1; m < M2; m++)//M2部分要乘一个前因子，另外单独加
	{
		double t_m = t_0 + h * m;
		Gaussian += factor * f_m(0.0, t_m, lambda) * exp(-3 * t_m / 2 - exp(-t_m) * k2 / 4.0);
	}
	double t_M2 = t_0 + h * M2;
	Gaussian = Gaussian + w_M2 * factor * f_m(0.0, t_M2, lambda) * exp(-3 * t_M2 / 2 - exp(-t_M2) * k2 / 4.0);
	return Gaussian;
}

double MH_D_DRBSOG(int xx, double factor)
{
	double qes;
	if (xx == 0)
		qes = erf(0.5 * factor);
	else
		qes = 0.5 * (erf((abs(xx) + 0.5) * factor) - erf((abs(xx) - 0.5) * factor));

	return qes;
}

/* ----------------------------------------------------------------------
   required functions
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

DRBSOG::DRBSOG(LAMMPS* lmp) : KSpace(lmp)
{
	TimeSet = new double[1000];
}

void DRBSOG::settings(int narg, char** arg)
{
	if (narg != 8) error->all(FLERR, "Illegal kspace_style DRBSOG command");
	h = fabs(utils::numeric(FLERR, arg[0], false, lmp));
	t_0 = utils::numeric(FLERR, arg[1], false, lmp);
	M1 = fabs(utils::numeric(FLERR, arg[2], false, lmp));
	M2 = fabs(utils::numeric(FLERR, arg[3], false, lmp));
	w_M2 = utils::numeric(FLERR, arg[4], false, lmp);
    lambda = utils::numeric(FLERR, arg[5], false, lmp);
	P = fabs(utils::numeric(FLERR, arg[6], false, lmp));
    Kcut = utils::numeric(FLERR, arg[7], false, lmp);//以后再改

	Step = 0;

	K_Sample = new double* [P];
	for (int i = 0; i < P; i++)
	{
		K_Sample[i] = new double [3];
	}

    if(comm->me==0)
    cout << "The coefficients are " << endl << "h == " << h << "   t_0 == " << t_0 << "    M1 == " << M1 << "   M2 == " << M2 << "    w_M2 == " << w_M2 << "   lambda == " << lambda << "    P == " << P <<"    Kcut == "<<Kcut<< endl;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

DRBSOG::~DRBSOG()
{

}

/* ---------------------------------------------------------------------- */

void DRBSOG::init()
{
    if (comm->me == 0) utils::logmesg(lmp, "DRBSOG initialization ...\n");

    // error check

    triclinic_check();
    if (domain->dimension == 2)
        error->all(FLERR, "Cannot use DRBSOG with 2d simulation");

    if (!atom->q_flag) error->all(FLERR, "Kspace style requires atom attribute q");

    if (slabflag == 0 && domain->nonperiodic > 0)
        error->all(FLERR, "Cannot use non-periodic boundaries with DRBSOG");
    if (slabflag) {
        if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
            domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
            error->all(FLERR, "Incorrect boundaries with slab DRBSOG");
        if (domain->triclinic)
            error->all(FLERR, "Cannot (yet) use DRBSOG with triclinic box "
                "and slab correction");
        }

    // compute two charge force

    two_charge();

    // extract short-range Coulombic cutoff from pair style

    int triclinic = domain->triclinic;
    pair_check();

    // compute qsum & qsqsum and warn if not charge-neutral

    scale = 1.0;
    qqrd2e = force->qqrd2e;
    qsum_qsq();
    natoms_original = atom->natoms;

    // setup K-space resolution

    bigint natoms = atom->natoms;

    // use xprd,yprd,zprd even if triclinic so grid size is the same
    // adjust z dimension for 2d slab Ewald
    // 3d Ewald just uses zprd since slab_volfactor = 1.0

    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    double zprd_slab = zprd * slab_volfactor;

    // make initial g_ewald estimate
    // based on desired accuracy and real space cutoff
    // fluid-occupied volume used to estimate real-space error
    // zprd used rather than zprd_slab

    // setup Ewald coefficients so can print stats

    int itmp;
    double* p_cutoff = (double*)force->pair->extract("cut_coul", itmp);
    if (p_cutoff == nullptr)
        error->all(FLERR, "KSpace style is incompatible with Pair style");
    rcut = *p_cutoff;

    //cout << "rcut == " << rcut << endl;

    //r0 = rcut / sigma;
    //w0 = Compute_W0(r0, b);

    char id_nh[] = "fxnvt";
    int ifix = modify->find_fix(id_nh);
    double t_start;
    if (ifix >= 0)
    {
        Fix* nh = modify->fix[ifix];
        double* aaa = (double*)nh->extract("t_start", ifix);
        t_start = *aaa;
        //error->all(FLERR, "Fix id for nvt or npt fix does not exist");
    }
    else
    {
        t_start = 298;
    }
    KbT = t_start * force->boltz;

    setup();

    if (comm->me == 0 && update->ntimestep == 0)
        cout << "The temperature is " << t_start << " Boltz is " << force->boltz << "  KbT == " << KbT << endl;
}

/* ----------------------------------------------------------------------
   adjust DRBSOG coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void DRBSOG::setup()
{
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd * slab_volfactor;
    V = xprd * yprd * zprd;
    double sum = 0.00;
    Kmax = 60;//注意这里在算S的时候很重要

    qsum_qsq();
    QsumQ = q2 / V;

    // 
    //如果是NPT 这里可能要想个好的估算方法
    for (int i = -Kmax; i < Kmax + 1; i++)
    {
        double Kx = (2 * MY_PI / xprd) * (i + 0.00);
        
        for (int j = -Kmax; j < Kmax + 1; j++)
        {
            double Ky = (2 * MY_PI / yprd) * (j + 0.00);

            for (int k = -Kmax; k < Kmax + 1; k++)
            {
                double Kz = (2 * MY_PI / zprd) * (k + 0.00);

                if ( (i * i + j * j + k * k) > Kcut)
                {
                    double K2 = Kx * Kx + Ky * Ky + Kz * Kz;

                    //sum = sum + GaussFourier(h, t_0, w_M2, M1, M2, lambda, K2) * K2;
                    sum = sum + GaussFourier(h, t_0, w_M2, M1, M2, lambda, K2) * (KbT / (KbT / QsumQ + 1.0 / (K2 + 1.0 / (lambda * lambda))));//这里有一步修正
					 }
            }
        }
    }
    S = sum;

    //if (comm->me == 0 && update->ntimestep == 0)
    //  cout << "The total normalized number S = " << S << endl;


    //if (comm->me == 0 && update->ntimestep == 0)
    //    cout << "Q == " << QsumQ << "   q2 == " << q2 << "   volume == " << V << endl;

    MPI_Comm_size(MPI_COMM_WORLD, &RankID);
}

/* ----------------------------------------------------------------------
   compute the UserRBE long-range force, energy, virial
------------------------------------------------------------------------- */

void DRBSOG::compute(int eflag, int vflag)
{
    double time;
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();

    if (Step == 0)
        srand(comm->me * 5 + 1993);//很重要 不然各个CPU会生成类似的样本

    ev_init(eflag, vflag);
    if (atom->natoms != natoms_original) {
        qsum_qsq();
        natoms_original = atom->natoms;
        }
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd * slab_volfactor;
    scale = 1;
    V = xprd * yprd * zprd;

    double K[P][3];
    double pxyz[3] = { 2 * MY_PI / xprd, 2 * MY_PI / yprd, 2 * MY_PI / zprd };
    double midterm[P][3];

    double exp_tm_2 = exp(-(t_0 + M2 * h) / 2.0);

    /****************************************采样*************************************************************************/
    int This_Index = Step * P;
    int this_rank = (Step % RankID);
    if (((Step % RankID) == 0) && (comm->me < RankID)) {
        int mx[5 * P], my[5 * P], mz[5 * P];
        int index = 0;

        double factor_xyz[3] = { xprd / (sqrt(2.0) * MY_PI * exp_tm_2) , yprd / (sqrt(2.0) * MY_PI * exp_tm_2) , zprd / (sqrt(2.0) * MY_PI * exp_tm_2) };
        //double factor_xyz[3] = { xprd / (2 * MY_PI * sigma) , yprd / (2 * MY_PI * sigma) , zprd / (2 * MY_PI * sigma) };

        double MHD_factor[3] = {  MY_PI * exp_tm_2 / xprd, MY_PI * exp_tm_2 / yprd, MY_PI * exp_tm_2 / zprd };
        //double MHD_factor[3] = { sqrt(2 * sigma * sigma * MY_PI * MY_PI) / xprd, sqrt(2 * sigma * sigma * MY_PI * MY_PI) / yprd, sqrt(2 * sigma * sigma * MY_PI * MY_PI) / zprd };

        double x, mold_x, mnew_x, xx, mold_y, mnew_y, xxx, mold_z, mnew_z, Kx_new, Ky_new, Kz_new, K2_new, Kx_old, Ky_old, Kz_old, K2_old, pup, qup, pdown, qdown, acce, yyy, K_debye_new, K_debye_old;

        do {
            mx[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[0]));
            my[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[1]));
            mz[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[2]));
            } while (mx[0] == 0 && my[0] == 0 && mz[0] == 0);

            for (int i = 0; i < 5 * P - 1; i++)
            {
                x = randn_box_muller_linear_congruential(0, factor_xyz[0]);
                mold_x = mx[i];
                mnew_x = round(x);

                xx = randn_box_muller_linear_congruential(0, factor_xyz[1]);
                mold_y = my[i];
                mnew_y = round(xx);

                xxx = randn_box_muller_linear_congruential(0, factor_xyz[2]);
                mold_z = mz[i];
                mnew_z = round(xxx);

                Kx_new = pxyz[0] * (mnew_x + 0.00);
                Ky_new = pxyz[1] * (mnew_y + 0.00);
                Kz_new = pxyz[2] * (mnew_z + 0.00);
                K2_new = Kx_new * Kx_new + Ky_new * Ky_new + Kz_new * Kz_new;

                //K_debye_new = K2_new;
                K_debye_new = (KbT / (KbT / QsumQ + 1.0 / (K2_new + 1.0 / (lambda * lambda))));

                Kx_old = pxyz[0] * (mold_x + 0.00);
                Ky_old = pxyz[1] * (mold_y + 0.00);
                Kz_old = pxyz[2] * (mold_z + 0.00);
                K2_old = Kx_old * Kx_old + Ky_old * Ky_old + Kz_old * Kz_old;

                //K_debye_old = K2_old;
                K_debye_old = (KbT / (KbT / QsumQ + 1.0 / (K2_old + 1.0 / (lambda * lambda))));

                //pup = Gaussian(mnew_x, mnew_y, mnew_z, xprd, yprd, zprd, sigma, b, w0, Mmax) * K_debye_new;
                pup = GaussFourier(h, t_0, w_M2, M1, M2, lambda, K2_new) * K_debye_new;

                qup = MH_D_DRBSOG(mold_x, MHD_factor[0]) * MH_D_DRBSOG(mold_y, MHD_factor[1]) * MH_D_DRBSOG(mold_z, MHD_factor[2]);

                pdown = GaussFourier(h, t_0, w_M2, M1, M2, lambda, K2_old) * K_debye_old;

                qdown = MH_D_DRBSOG(mnew_x, MHD_factor[0]) * MH_D_DRBSOG(mnew_y, MHD_factor[1]) * MH_D_DRBSOG(mnew_z, MHD_factor[2]);

                acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
                yyy = (rand() % 10000 + 0.00) / 10000.0;

                //cout << yyy << "   " << acce << "    "<< pup <<"    "<< qup<<endl;

                if (yyy < acce) {
                    mx[i + 1] = mnew_x;
                    my[i + 1] = mnew_y;
                    mz[i + 1] = mnew_z;
                    index++;
                    }
                else {
                    mx[i + 1] = mold_x;
                    my[i + 1] = mold_y;
                    mz[i + 1] = mold_z;
                    }
                if (mx[i + 1] * mx[i + 1] + my[i + 1] * my[i + 1] + mz[i + 1] * mz[i + 1] <= Kcut)//调整前几个不采样
                    i = i - 1;
            }
            for (int i = 0; i < P; i++)
                {
                K_Sample[i][0] = mx[5 * i + 4] + 0.00;
                K_Sample[i][1] = my[5 * i + 4] + 0.00;
                K_Sample[i][2] = mz[5 * i + 4] + 0.00;
                }
    }

    if (comm->me == this_rank)
    {
        for (int i = 0; i < P; i++)
         {
            K[i][0] = K_Sample[i][0] * pxyz[0];
            K[i][1] = K_Sample[i][1] * pxyz[1];
            K[i][2] = K_Sample[i][2] * pxyz[2];
            //cout << K[i][0] << "    " << K[i][1] << "     " << K[i][2] << endl;
        }
    }

    MPI_Bcast((double*)K, 3 * P, MPI_DOUBLE, this_rank, MPI_COMM_WORLD);

    Step++;

    time = MPI_Wtime();

    /*  Set Pointer */
    double** x = atom->x;//atom->x指向的是粒子的位置
    double** f = atom->f;//atom->f指向的是存储力的数组
    double* q = atom->q;//atom->q指向存储带电量的数组
    int* type = atom->type;//原子的类型
    int nlocal = atom->nlocal;//当前mpi上存储的原子数量
    double qqrd2e = force->qqrd2e;//力的一个系数
    double dielectric = force->dielectric;//介电常数

    double Rho[P][2], Rho_All[P][2];

    double X[int(ceil((nlocal + 0.0) / 8.0)) * 8], Y[int(ceil((nlocal + 0.0) / 8.0)) * 8], Z[int(ceil((nlocal + 0.0) / 8.0)) * 8];
    double F[int(ceil((nlocal + 0.0) / 8.0)) * 8][3];
    double Q[int(ceil((nlocal + 0.0) / 8.0)) * 8];
    for (int i = 0; i < int(ceil((nlocal + 0.0) / 8.0)) * 8; i++)
        {
        if (i < nlocal) {
            X[i] = x[i][0];
            Y[i] = x[i][1];
            Z[i] = x[i][2];
            F[i][0] = 0.00;
            F[i][1] = 0.00;
            F[i][2] = 0.00;
            Q[i] = q[i];
            }
        else
            {
            X[i] = 0.0;
            Y[i] = 0.00;
            Z[i] = 0.00;
            F[i][0] = 0.00;
            F[i][1] = 0.00;
            F[i][2] = 0.00;
            Q[i] = 0.00;
            }
        }

    double KxKx0[int(ceil((P + 0.0) / 8.0)) * 8], KxKx1[int(ceil((P + 0.0) / 8.0)) * 8], KxKx2[int(ceil((P + 0.0) / 8.0)) * 8];
    double Rho_Cos[int(ceil((P + 0.0) / 8.0)) * 8], Rho_Sin[int(ceil((P + 0.0) / 8.0)) * 8];

    for (int i = 0; i < int(ceil((P + 0.0) / 8.0)) * 8; i++)
    {
        if (i < P) {
            KxKx0[i] = K[i][0];
            KxKx1[i] = K[i][1];
            KxKx2[i] = K[i][2];
            }
        else if (i >= P)
            {
            KxKx0[i] = 0.00;
            KxKx1[i] = 0.00;
            KxKx2[i] = 0.00;
            }
    }

    double KKx[8][3];
    //float moment[16];

    __m512d Real, Imag, X0, X1, X2, qq, Cos, Sin,
        Moment, Kx, Ky, Kz;

    for (int i = 0; i < P; i += 8)
    {
        Real = Imag = _mm512_setzero_pd();
        Kx = _mm512_load_pd(&KxKx0[i]);
        Ky = _mm512_load_pd(&KxKx1[i]);
        Kz = _mm512_load_pd(&KxKx2[i]);

        for (int j = 0; j < nlocal; j++)
        {
            X0 = _mm512_set1_pd(X[j]);
            X1 = _mm512_set1_pd(Y[j]);
            X2 = _mm512_set1_pd(Z[j]);
            qq = _mm512_set1_pd(Q[j]);
            Moment = Kx * X0 + Ky * X1 + Kz * X2;
            Sin = _mm512_sincos_pd(&Cos, Moment);
            Real = Real + qq * Cos;
            Imag = Imag + qq * Sin;
			//cout << j << " REAL==  " << Real << "  IMAG == " << Imag << endl;

        }

        _mm512_store_pd(&Rho_Cos[i], Real);
        _mm512_store_pd(&Rho_Sin[i], Imag);
    }

    for (int i = 0; i < P; i++)
    {
        Rho[i][0] = Rho_Cos[i];
        Rho[i][1] = Rho_Sin[i];

    }

    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce((double*)Rho, (double *)Rho_All, 2 * P, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
    double MIDTERM = - (S / (P + 0.00)) * qqrd2e / V;


    __m512d Fx, Fy, Fz, K2, moment_512, midterm_512, Rho_All_0, Rho_All_1, K2_Debye;
    double F_x[8], F_y[8], F_z[8];

    // sum global energy across Kspace vevs and add in volume-dependent term
    const double qscale = qqrd2e * scale;

    for (int i = 0; i < nlocal; i += 8)
    {
        X0 = _mm512_load_pd(&X[i]);
        X1 = _mm512_load_pd(&Y[i]);
        X2 = _mm512_load_pd(&Z[i]);
        qq = _mm512_load_pd(&Q[i]);

        Fx = Fy = Fz = _mm512_setzero_pd();

        for (int j = 0; j < P; j++)
        {
            Kx = _mm512_set1_pd(KxKx0[j]);
            Ky = _mm512_set1_pd(KxKx1[j]);
            Kz = _mm512_set1_pd(KxKx2[j]);

            double k2 = KxKx0[j] * KxKx0[j] + KxKx1[j] * KxKx1[j] + KxKx2[j] * KxKx2[j];
            //K2 = Kx * Kx + Ky * Ky + Kz * Kz;

            //K2_Debye = _mm512_set1_pd(k2);
            K2_Debye = _mm512_set1_pd(KbT / (KbT / QsumQ + 1.0 / (k2 + 1.0 / (lambda * lambda))));


            moment_512 = -(Kx * X0 + Ky * X1 + Kz * X2);

            Sin = _mm512_sincos_pd(&Cos, moment_512);

            midterm_512 = _mm512_set1_pd(MIDTERM);

            Rho_All_0 = _mm512_set1_pd(Rho_All[j][0]);
            Rho_All_1 = _mm512_set1_pd(Rho_All[j][1]);
            Imag = (Cos * Rho_All_1 + Sin * Rho_All_0) * midterm_512 / K2_Debye;
				//cout <<  " Imag==  " << Imag << endl;

            Fx = Fx + qq * Imag * Kx;
            Fy = Fy + qq * Imag * Ky;
            Fz = Fz + qq * Imag * Kz;
        }
        _mm512_store_pd(&F_x[0], Fx);
        _mm512_store_pd(&F_y[0], Fy);
        _mm512_store_pd(&F_z[0], Fz);

        for (int j = 0; j < 8; j++) {
            F[i + j][0] = F[i + j][0] + F_x[j];
            F[i + j][1] = F[i + j][1] + F_y[j];
            F[i + j][2] = F[i + j][2] + F_z[j];
        }
    }

    if (eflag_global) {
        
        double fact_ener;
        for (int j = 0; j < P; j++)
        {
            double k2 = KxKx0[j] * KxKx0[j] + KxKx1[j] * KxKx1[j] + KxKx2[j] * KxKx2[j];
            //fact_ener = k2; 
            fact_ener = KbT / (KbT / QsumQ + 1.0 / (k2 + 1.0 / (lambda * lambda)));

            energy += (Rho_All[j][0] * Rho_All[j][0] + Rho_All[j][1] * Rho_All[j][1]) * S * qscale / (2 * P * V * fact_ener);
        }
        //先不管这部分能量
    }

    time = MPI_Wtime() - time;  // 终止计时

    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
        }

    /*                    前面直接计算的部分                            */
    //有些系数还要改
    if ((Kcut >= 0))
    {
        if (RankID == 1)
        {
            for (int i = -ceil(sqrt(Kcut)); i < ceil(sqrt(Kcut)) + 1; i++)
            {
                double Kx = pxyz[0] * (i + 0.00);
                for (int j = -ceil(sqrt(Kcut)); j < ceil(sqrt(Kcut)) + 1; j++)
                {
                    double Ky = pxyz[1] * (j + 0.00);
                    for (int k = -ceil(sqrt(Kcut)); k < ceil(sqrt(Kcut)) + 1; k++)
                    {
                        double x0, x1, x2, qq, moment, Sin, Cos;

                        double Kz = pxyz[2] * (k + 0.00);
                        double Real = 0.00, Imag = 0.00;

                        if (i * i + j * j + k * k <= Kcut)
                        {
                            for (int m = 0; m < nlocal; m++)
                            {
                                x0 = x[m][0];
                                x1 = x[m][1];
                                x2 = x[m][2];
                                qq = q[m];
                                moment = Kx * x0 + Ky * x1 + Kz * x2;
                                sincos(moment, &Sin, &Cos);
                                Real = Real + qq * Cos;
                                Imag = Imag + qq * Sin;
                            }
                            //到此 得到rho(k)的实部和虚部
                            double midterm;
                            double k2 = Kx * Kx + Ky * Ky + Kz * Kz;
                            //double F_b_sigma = Gaussian_Fourier_Plus(Kx, Ky, Kz, xprd, yprd, zprd, sigma, b, w0, Mmax);
                            double F_b_sigma = GaussFourier(h, t_0, w_M2, M1, M2, lambda, k2);

                            for (int m = 0; m < nlocal; m++)
                            {
                                x0 = x[m][0];
                                x1 = x[m][1];
                                x2 = x[m][2];
                                qq = q[m];
                                moment = Kx * x0 + Ky * x1 + Kz * x2;
                                sincos(-moment, &Sin, &Cos);
                                midterm = Sin * Real + Imag * Cos;

                                F[m][0] = F[m][0] - qq * Kx * qqrd2e * midterm * F_b_sigma / V;
                                F[m][1] = F[m][1] - qq * Ky * qqrd2e * midterm * F_b_sigma / V;
                                F[m][2] = F[m][2] - qq * Kz * qqrd2e * midterm * F_b_sigma / V;
                            }

                            if (eflag_global) {
                                energy = energy + qscale * F_b_sigma * (Real * Real + Imag * Imag) / (2.0 * V);
                            }
                        }
                    }
                }
            }
        }
        else if (RankID > 1)
        {
            int Size = (2 * ceil(sqrt(Kcut)) + 1) * (2 * ceil(sqrt(Kcut)) + 1) * (2 * ceil(sqrt(Kcut)) + 1);
            double Kx_Dir[Size][3];

            int index = 0;
            for (int i = -ceil(sqrt(Kcut)); i < ceil(sqrt(Kcut)) + 1; i++)
            {
                double Kx = pxyz[0] * (i + 0.00);
                for (int j = -ceil(sqrt(Kcut)); j < ceil(sqrt(Kcut)) + 1; j++)
                {
                    double Ky = pxyz[1] * (j + 0.00);
                    for (int k = -ceil(sqrt(Kcut)); k < ceil(sqrt(Kcut)) + 1; k++)
                    {
                        double Kz = pxyz[2] * (k + 0.00);

                        //if ((!(i == 0 && j == 0 && k == 0)) && (i * i + j * j + k * k <= Kcut))
                        if ((i * i + j * j + k * k <= Kcut))//问题要不要带0mode？
                        {
                            Kx_Dir[index][0] = Kx;
                            Kx_Dir[index][1] = Ky;
                            Kx_Dir[index][2] = Kz;
                            index++;
                        }
                    }
                }
            }
            double Rho[index][2], Rho_All[index][2];
            double F_b_sigma[index];

            for (int i = 0; i < index; i++)
            {
                double Kx = Kx_Dir[i][0];
                double Ky = Kx_Dir[i][1];
                double Kz = Kx_Dir[i][2];
                double Real = 0.00, Imag = 0.00;

                double x0, x1, x2, qq, moment, Sin, Cos;

                double k2 = Kx * Kx + Ky * Ky + Kz * Kz;
                F_b_sigma[i] = GaussFourier(h, t_0, w_M2, M1, M2, lambda, k2);

                for (int m = 0; m < nlocal; m++)
                {
                    x0 = x[m][0];
                    x1 = x[m][1];
                    x2 = x[m][2];
                    qq = q[m];
                    moment = Kx * x0 + Ky * x1 + Kz * x2;
                    sincos(moment, &Sin, &Cos);
                    Real = Real + qq * Cos;
                    Imag = Imag + qq * Sin;
                }

                Rho[i][0] = Real;
                Rho[i][1] = Imag;

            }

            MPI_Allreduce((double*)Rho, (double*)Rho_All, 2 * index, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double x0, x1, x2, qq, moment, Sin, Cos;
            for (int m = 0; m < nlocal; m++)
            {
                x0 = x[m][0];
                x1 = x[m][1];
                x2 = x[m][2];
                qq = q[m];

                double midterm;

                for (int i = 0; i < index; i++)
                {
                    double Kx = Kx_Dir[i][0];
                    double Ky = Kx_Dir[i][1];
                    double Kz = Kx_Dir[i][2];

                    moment = Kx * x0 + Ky * x1 + Kz * x2;
                    sincos(-moment, &Sin, &Cos);
                    midterm = Sin * Rho_All[i][0] + Rho_All[i][1] * Cos;

                    F[m][0] = F[m][0] - qq * Kx * qqrd2e * midterm * F_b_sigma[i] / V;
                    F[m][1] = F[m][1] - qq * Ky * qqrd2e * midterm * F_b_sigma[i] / V;
                    F[m][2] = F[m][2] - qq * Kz * qqrd2e * midterm * F_b_sigma[i] / V;

                }
            }

            if (eflag_global) {
                for (int i = 0; i < index; i++)
                {
                    energy = energy + qscale * F_b_sigma[i] * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (2.0 * V);

}
            }
        }
        else
            {
            error->all(FLERR, "Need RankID >=1");
            }
    }
        
    if (eflag_global) {
        double sume = 0.00;
        sume = w_M2 * f_m(0, t_0 + M2 * h, lambda);
        for (int j = -M1; j <= M2 - 1; j++)
            {
            sume = sume + f_m(0, t_0 + j * h, lambda);
            }
        sume = sume * h * qsqsum * qscale / 2.0;
        energy -= sume;//能量修正

    }

    /*                    力的归约                            */
    for (int i = 0; i < nlocal; i++)
    {
        f[i][0] += F[i][0];
        f[i][1] += F[i][1];
        f[i][2] += F[i][2];
    }

    if (comm->me == 0)
    {
        for (int i = 0; i < 100; i++)
        {
            //cout << i << " fff  " << f[i][0] << "    " << f[i][1] << "     " << f[i][2] << endl;
        }
     }

    if (comm->me == 0)
        //cout << F[0][0] << "    " << F[0][1] << "    " << F[0][2] << endl;
    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_Kspace_RBSOG.txt");
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
        }

}

double DRBSOG::memory_usage()
{

}