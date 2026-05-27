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
   Contributing authors: Jiuyang Liang (SJTU), Qi Zhou (SJTU), Zhen Jiang (SJTU)
------------------------------------------------------------------------- */

#include "rbsog_tip4p.h"
#include <mpi.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "angle.h"
#include "bond.h"
#include "math_const.h"
#include "memory.h"
#include "update.h"
#include "pair.h"

#include <time.h>
#include <stdlib.h>


#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef> 
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(0, 1);
static std::uniform_real_distribution<> dis_uniform(-1, 1);

#define SMALL 0.00001
#ifdef LMP_GPU
int rbsog_gpu_compute_rho(const int nlocal, const int pcount, const float *x, const float *y,
                          const float *z, const float *q, const float *kx, const float *ky,
                          const float *kz, float *rho_cos, float *rho_sin);
int rbsog_gpu_compute_force_sampled(const int nlocal, const int pcount, const float *x,
                                    const float *y, const float *z, const float *q,
                                    const float *kx, const float *ky, const float *kz,
                                    const float *fac, const float *rho_all_cos,
                                    const float *rho_all_sin, const float midterm, float *fx,
                                    float *fy, float *fz);
int rbsog_gpu_compute_force_direct_single_rank(const int nlocal, const int pcount, const float *x,
                                               const float *y, const float *z, const float *q,
                                               const float *kx, const float *ky, const float *kz,
                                               const float *coeff, const float midterm, float *fx,
                                               float *fy, float *fz, float *rho_cos, float *rho_sin);
#endif



/* ----------------------------------------------------------------------
   required functions
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */
RBSOGTIP4P::RBSOGTIP4P(LAMMPS* lmp) : RBSOG(lmp)
{
    // TimeSet = new double* [5];
    // for (int i = 0; i < 5; i++)
    // {
    //     TimeSet[i] = new double[10];
    // }
    // TimeSet_sampling = new double[1000];
    tip4pflag = 1;
    rbsogflag = 1;
}


void RBSOGTIP4P::init()
{
    RBSOG::init();
    qdist = 0.0;
    int itmp;
    if (tip4pflag) {
        if (comm->me == 0) utils::logmesg(lmp,"  extracting TIP4P info from pair style\n");

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
}

/* ----------------------------------------------------------------------
   compute the UserRBE long-range force, energy, virial
------------------------------------------------------------------------- */

void RBSOGTIP4P::compute(int eflag, int vflag)
{
    // double time_test;
    // MPI_Barrier(MPI_COMM_WORLD);
    
    Step = update->ntimestep;
    // if (Step == 0)
    //     srand(comm->me * 5);
    // srand(comm->me * dis(gen));

    ev_init(eflag, vflag);

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
    float K_npt[P][3];
    float fac[P];
    float fac_npt[P];
    for(int i = 0; i < P; i++)
    {
        fac[i] = 1;
        fac_npt[i] = 1;
    }
    // double xprd0, yprd0, zprd0;
    // float S0, S_npt0;

    int idx_npt_all[P];

    float pxyz[3] = {  static_cast<float>(2 * MY_PI / xprd),  static_cast<float>(2 * MY_PI / yprd),  static_cast<float>(2 * MY_PI / zprd) };
    float Rho[P][2], Rho_All[P][2];
    float midterm[P][3];

    double time;
    // MPI_Barrier(MPI_COMM_WORLD);
    // time = MPI_Wtime();

    // time = MPI_Wtime();
    int This_Index = Step * P;
    int this_rank = (Step % RankID);

    //srand(time(0));

    //sampling

    if (((Step % RankID) == 0) && (comm->me < RankID)) {
        xprd0 = xprd;
        yprd0 = yprd;
        zprd0 = zprd;
        S0 = S;
        S_npt0 = S_npt;
        int mx[5 * P], my[5 * P], mz[5 * P];
        double factor_xyz[3] = { xprd / (2 * MY_PI * sigma) , yprd / (2 * MY_PI * sigma) , zprd / (2 * MY_PI * sigma) };
        double MHD_factor[3] = { sqrt(2 * sigma * sigma * MY_PI * MY_PI) / xprd, sqrt(2 * sigma * sigma * MY_PI * MY_PI) / yprd, sqrt(2 * sigma * sigma * MY_PI * MY_PI) / zprd };

        double x, mold_x, mnew_x, xx, mold_y, mnew_y, xxx, mold_z, mnew_z, Kx_new, Ky_new, Kz_new, K2_new, Kx_old, Ky_old, Kz_old, K2_old, pup, qup, pdown, qdown, acce, yyy;
        do {
            mx[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[0]));
            my[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[1]));
            mz[0] = round(randn_box_muller_linear_congruential(0, factor_xyz[2]));
        } while (mx[0] * mx[0] + my[0] * my[0] + mz[0] * mz[0] <= Kcut);
        for (int i = 0; i < 5*P - 1; i++)
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
            Kx_old = pxyz[0] * (mold_x + 0.00);
            Ky_old = pxyz[1] * (mold_y + 0.00);
            Kz_old = pxyz[2] * (mold_z + 0.00);
            K2_old = Kx_old * Kx_old + Ky_old * Ky_old + Kz_old * Kz_old;

            pup = Gaussian(mnew_x, mnew_y, mnew_z, xprd, yprd, zprd, sigma, b, w0, Mmax, coef)* K2_new;
            qup = MH_D_Modify(mold_x, MHD_factor[0]) * MH_D_Modify(mold_y, MHD_factor[1]) * MH_D_Modify(mold_z, MHD_factor[2]);
            pdown = Gaussian(mold_x, mold_y, mold_z, xprd, yprd, zprd, sigma, b, w0, Mmax, coef)* K2_old;
            qdown = MH_D_Modify(mnew_x, MHD_factor[0]) * MH_D_Modify(mnew_y, MHD_factor[1]) * MH_D_Modify(mnew_z, MHD_factor[2]);

            acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
            yyy = rand() * (1.0 / RAND_MAX);


            if (yyy < acce) {
                mx[i + 1] = mnew_x;
                my[i + 1] = mnew_y;
                mz[i + 1] = mnew_z;
            }
            else {
                mx[i + 1] = mold_x;
                my[i + 1] = mold_y;
                mz[i + 1] = mold_z;
            }
            if ((mx[i + 1] == 0 && my[i + 1] == 0 && mz[i + 1] == 0) || (mx[i + 1] * mx[i + 1] + my[i + 1] * my[i + 1] + mz[i + 1] * mz[i + 1] <= Kcut))
                i = i - 1;
        }
        for (int i = 0; i < P; i++)
        {
            K_Sample[i][0] = mx[5 * i + 4] + 0.00;
            K_Sample[i][1] = my[5 * i + 4] + 0.00;
            K_Sample[i][2] = mz[5 * i + 4] + 0.00;
        }
        for (int i = 1; i < P; i++)
        {
            double mmx_old, mmy_old, mmz_old, mmx_new, mmy_new, mmz_new, Kx_new, Ky_new, Kz_new, K2_new, K4_new, Kx_old, Ky_old, Kz_old, K2_old, K4_old, pup, qup, pdown, qdown, acce, yyy;
            int idx;
            idx = idx_npt[i - 1];

            mmx_old = K_Sample[idx][0];
            mmy_old = K_Sample[idx][1];
            mmz_old = K_Sample[idx][2];
            mmx_new = K_Sample[i][0];
            mmy_new = K_Sample[i][1];
            mmz_new = K_Sample[i][2];

            Kx_new = pxyz[0] * (mmx_new + 0.00);
            Ky_new = pxyz[1] * (mmy_new + 0.00);
            Kz_new = pxyz[2] * (mmz_new + 0.00);
            K2_new = Kx_new * Kx_new + Ky_new * Ky_new + Kz_new * Kz_new;
            K4_new = K2_new * K2_new;
            Kx_old = pxyz[0] * (mmx_old + 0.00);
            Ky_old = pxyz[1] * (mmy_old + 0.00);
            Kz_old = pxyz[2] * (mmz_old + 0.00);
            K2_old = Kx_old * Kx_old + Ky_old * Ky_old + Kz_old * Kz_old;
            K4_old = K2_old * K2_old;

            pup = Gaussian_modify(mmx_new, mmy_new, mmz_new, xprd, yprd, zprd, sigma, b, w0, Mmax, coef_npt)* K4_new;
            qup = Gaussian(mmx_old, mmy_old, mmz_old, xprd, yprd, zprd, sigma, b, w0, Mmax, coef)* K2_old;
            pdown = Gaussian_modify(mmx_old, mmy_old, mmz_old, xprd, yprd, zprd, sigma, b, w0, Mmax, coef_npt)* K4_old;
            qdown = Gaussian(mmx_new, mmy_new, mmz_new, xprd, yprd, zprd, sigma, b, w0, Mmax, coef)* K2_new;
            acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
            yyy = rand() * (1.0 / RAND_MAX);
            if (yyy < acce) {
                idx_npt[i] = i;
            }
            else {
                idx_npt[i] = idx;
            }
                    
        }
    }

    if (comm->me == this_rank)
    {
        for (int i = 0; i < P; i++)
        {
            K[i][0] = K_Sample[i][0] * pxyz[0];
            K[i][1] = K_Sample[i][1] * pxyz[1];
            K[i][2] = K_Sample[i][2] * pxyz[2];
            int id = idx_npt[i];
            K_npt[i][0] = K_Sample[id][0] * pxyz[0];
            K_npt[i][1] = K_Sample[id][1] * pxyz[1];
            K_npt[i][2] = K_Sample[id][2] * pxyz[2];
            idx_npt_all[i] = id;
        }
    }
    // time = MPI_Wtime() - time; 
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[0][index] = time * 1000;
    // }

    MPI_Bcast((float*)K, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);
    MPI_Bcast((float*)K_npt, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);
    MPI_Bcast(idx_npt_all, P, MPI_INT, this_rank, MPI_COMM_WORLD);

    //stop

    //time = MPI_Wtime() - time;
    //if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
    //    TimeSet_sampling[update->ntimestep - 0] = time * 1000;
    //}

    //if (comm->me == 0 && update->ntimestep == 1000) {
    //    ofstream outfile;
    //    outfile.open("./Time_Kspace_RBSOGTIP4P_sampling.txt");
    //    //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
    //    for (int i = 0; i < 1000; i++)
    //        outfile << TimeSet_sampling[i] << endl;
    //    outfile.close();
    //}


    // Step++;


    // if (comm->me == 0 && update->ntimestep%1000 == 0){
    //     cout << "fac = " << fac[P/2] << endl;
    //     cout << "fac_npt = " << fac_npt[P/2] << endl;
    // }

    // time = MPI_Wtime();
    /*  Set Pointer */

    //time = MPI_Wtime();
    double** x = atom->x;
    double** f = atom->f;
    double* q = atom->q;
    int* type = atom->type;
    int nlocal = atom->nlocal;
    double qqrd2e = force->qqrd2e;
    double dielectric = force->dielectric;

    double *xi,xM[3];
	int iH1 = 0, iH2 = 0;

    float X[int(ceil((nlocal + 0.0) / 16.0)) * 16], Y[int(ceil((nlocal + 0.0) / 16.0)) * 16], Z[int(ceil((nlocal + 0.0) / 16.0)) * 16];
    float F[int(ceil((nlocal + 0.0) / 16.0)) * 16][3];
    float Q[int(ceil((nlocal + 0.0) / 16.0)) * 16];
    for (int i = 0; i < int(ceil((nlocal + 0.0) / 16.0)) * 16; i++)
    {
		if (i < nlocal) {
			// find M site if TIP4P
			if (type[i] == typeO) {
				find_M(i,iH1,iH2,xM);
				xi = xM;
			} else xi = x[i];
			X[i] = xi[0]; Y[i] = xi[1]; Z[i] = xi[2];
			F[i][0] = 0.00; F[i][1] = 0.00; F[i][2] = 0.00;
			Q[i] = q[i];
			// if(me == 0 && i <= 4 && type[i] == typeO){
			// 	cout << "O position " << x[i][0] << " " << x[i][1] << " " << x[i][2]<<endl;
			// 	cout << "M site " << X[i][0] << " " << X[i][1] << " " << X[i][2]<<endl;
			// }
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

    float KxKx0[P], KxKx1[P], KxKx2[P];
    float KxKx0_npt[P], KxKx1_npt[P], KxKx2_npt[P];
    float Rho_Cos[P], Rho_Sin[P];

    for (int i = 0; i < P; i++)
    {
        KxKx0[i] = K[i][0];
        KxKx1[i] = K[i][1];
        KxKx2[i] = K[i][2];
        KxKx0_npt[i] = K_npt[i][0];
        KxKx1_npt[i] = K_npt[i][1];
        KxKx2_npt[i] = K_npt[i][2];
    }
    // time = MPI_Wtime();

    for (int i = 0; i < P; i+=16)
    {   
        RBSOGVec Kx, Ky, Kz, Kx0, Ky0, Kz0;
        RBSOGVec Kx_npt, Ky_npt, Kz_npt, Kx0_npt, Ky0_npt, Kz0_npt;
        RBSOGVec K2, K2_0, K2_npt, K2_npt_0, K4, K4_0;
        RBSOGVec L_ratio, S_ratio, S_npt_ratio;
        L_ratio = rbsog_set1_ps(xprd / xprd0);
        S_ratio = rbsog_set1_ps(S0 / S);
        S_npt_ratio = rbsog_set1_ps(S_npt0 / S_npt);
        Kx = rbsog_load_ps(&KxKx0[i]);
        Ky = rbsog_load_ps(&KxKx1[i]);
        Kz = rbsog_load_ps(&KxKx2[i]);
        Kx_npt = rbsog_load_ps(&KxKx0_npt[i]);
        Ky_npt = rbsog_load_ps(&KxKx1_npt[i]);
        Kz_npt = rbsog_load_ps(&KxKx2_npt[i]);
        Kx0 = Kx * L_ratio;
        Ky0 = Ky * L_ratio;
        Kz0 = Kz * L_ratio;
        Kx0_npt = Kx_npt * L_ratio;
        Ky0_npt = Ky_npt * L_ratio;
        Kz0_npt = Kz_npt * L_ratio;
        K2 = Kx * Kx + Ky * Ky + Kz * Kz;
        K2_0 = Kx0 * Kx0 + Ky0 * Ky0 + Kz0 * Kz0;
        K2_npt = Kx_npt * Kx_npt + Ky_npt * Ky_npt + Kz_npt * Kz_npt;
        K2_npt_0 = Kx0_npt * Kx0_npt + Ky0_npt * Ky0_npt + Kz0_npt * Kz0_npt;
        K4 = K2_npt * K2_npt;
        K4_0 = K2_npt_0 * K2_npt_0;

        RBSOGVec mid, mid0, mid_npt, mid0_npt;
        mid = Gaussian_Fourier_Plus_AVX(Kx, Ky, Kz, sigma, b, w0, Mmax, coef) * K2;
        mid0 = Gaussian_Fourier_Plus_AVX(Kx0, Ky0, Kz0, sigma, b, w0, Mmax, coef) * K2_0;
        mid_npt = Gaussian_Fourier_Plus_AVX(Kx_npt, Ky_npt, Kz_npt, sigma, b, w0, Mmax, coef_npt) * K4;
        mid0_npt = Gaussian_Fourier_Plus_AVX(Kx0_npt, Ky0_npt, Kz0_npt, sigma, b, w0, Mmax, coef_npt) * K4_0;
        RBSOGVec fac_tmp, fac_npt_tmp;
        fac_tmp = S_ratio * mid / mid0;
        fac_npt_tmp = S_npt_ratio * mid_npt / mid0_npt;

        rbsog_store_ps(&fac[i], fac_tmp);
        rbsog_store_ps(&fac_npt[i], fac_npt_tmp);
    }
    // if (comm->me == 0 && update->ntimestep % 100 == 0)
    // {
    //     ofstream outfile;
    //     outfile.open("./fac.txt");
    //     for (int i = 0; i < P; i++)
    //         outfile << "fac: " << fac[i] << ", fac_npt: " <<fac_npt[i] << endl;
    //     outfile.close();
    // }
    

    // time = MPI_Wtime() - time;
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[1][index] = time * 1000;    
    // }
    // time = MPI_Wtime();
    RBSOGVec Real, Imag, X0, X1, X2, qq, Cos, Sin,
        Moment, Kx, Ky, Kz;

    bool rho_on_gpu = false;
#ifdef LMP_GPU
    if (use_gpu_accel) {
        if (rbsog_gpu_compute_rho(nlocal, P, X, Y, Z, Q, KxKx0, KxKx1, KxKx2, Rho_Cos, Rho_Sin) == 0)
            rho_on_gpu = true;
    }
#endif

    if (!rho_on_gpu)
    for (int i = 0; i < P; i += 16)
    {

        Real = Imag = rbsog_setzero_ps();
        Kx = rbsog_load_ps(&KxKx0[i]);
        Ky = rbsog_load_ps(&KxKx1[i]);
        Kz = rbsog_load_ps(&KxKx2[i]);

        for (int j = 0; j < nlocal; j++)
        {
            X0 = rbsog_set1_ps(X[j]);
            X1 = rbsog_set1_ps(Y[j]);
            X2 = rbsog_set1_ps(Z[j]);
            qq = rbsog_set1_ps(Q[j]);


           //__mm512 mid0, mid1;
           // mid0 = vmulq_f32(Kx, X0);
           // mid1 = vfmaq_f32(mid0, Ky, X1);
           // Moment = vfmaq_f32(mid1, Kz, X2);
            Moment = Kx * X0 + Ky * X1 + Kz * X2;

            Sin = rbsog_sincos_ps(&Cos, Moment);
            //svml128_sincos_f32(Moment, &Sin, &Cos);


            Real = Real + qq * Cos;
            Imag = Imag + qq * Sin;
        }

        rbsog_store_ps(&Rho_Cos[i], Real);
        rbsog_store_ps(&Rho_Sin[i], Imag);
    }

    for (int i = 0; i < P; i++)
    {
        Rho[i][0] = Rho_Cos[i];
        Rho[i][1] = Rho_Sin[i];
    }
    // time = MPI_Wtime() - time;
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[2][index] = time * 1000;    
    // }
    // if (comm->me == 0 && update->ntimestep  % 100 == 0)
    // {
    //     ofstream outfile;
    //     outfile.open("./Rho.txt");
    //     for (int i = 0; i < P; i++)
    //         outfile << "Rho: " << Rho[i][0] << ", Rho: " << Rho[i][1] << endl;
    //     outfile.close();
    // }

    if (comm->nprocs == 1) {
        std::memcpy((float *)Rho_All, (float *)Rho, sizeof(float) * 2 * P);
    } else {
        MPI_Allreduce((float*)Rho, (float*)Rho_All, 2 * P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    // time = MPI_Wtime();
    double MIDTERM = - (S0 / (P + 0.00)) * qqrd2e / V;

    RBSOGVec Fx, Fy, Fz, K2, moment_512, midterm_512, Rho_All_0, Rho_All_1, factor;
    float F_x[16],F_y[16],F_z[16];

    bool sampled_force_on_gpu = false;
#ifdef LMP_GPU
    if (use_gpu_accel) {
        if (static_cast<int>(rho_all_cos_cache_.size()) < P) {
            rho_all_cos_cache_.resize(P);
            rho_all_sin_cache_.resize(P);
        }
        for (int i = 0; i < P; i++) {
            rho_all_cos_cache_[i] = Rho_All[i][0];
            rho_all_sin_cache_[i] = Rho_All[i][1];
        }
        if (static_cast<int>(fx_gpu_cache_.size()) < nlocal) {
            fx_gpu_cache_.resize(nlocal);
            fy_gpu_cache_.resize(nlocal);
            fz_gpu_cache_.resize(nlocal);
        }
        if (rbsog_gpu_compute_force_sampled(nlocal, P, X, Y, Z, Q, KxKx0, KxKx1, KxKx2, fac,
                                            rho_all_cos_cache_.data(), rho_all_sin_cache_.data(),
                                            static_cast<float>(MIDTERM), fx_gpu_cache_.data(),
                                            fy_gpu_cache_.data(), fz_gpu_cache_.data()) == 0) {
            for (int i = 0; i < nlocal; i++) {
                F[i][0] += fx_gpu_cache_[i];
                F[i][1] += fy_gpu_cache_[i];
                F[i][2] += fz_gpu_cache_[i];
            }
            sampled_force_on_gpu = true;
        }
    }
#endif

    if (!sampled_force_on_gpu)
    for (int i = 0; i < nlocal; i += 16)
    {

        X0 = rbsog_load_ps(&X[i]);
        X1 = rbsog_load_ps(&Y[i]);
        X2 = rbsog_load_ps(&Z[i]);
        qq = rbsog_load_ps(&Q[i]);

        Fx = Fy = Fz = rbsog_setzero_ps();

        for (int j = 0; j < P; j++)
        {

            Kx = rbsog_set1_ps(KxKx0[j]);
            Ky = rbsog_set1_ps(KxKx1[j]);
            Kz = rbsog_set1_ps(KxKx2[j]);

            K2 = Kx * Kx + Ky * Ky + Kz * Kz;

            moment_512 = -(Kx * X0 + Ky * X1 + Kz * X2);

            Sin = rbsog_sincos_ps(&Cos, moment_512);

            factor = rbsog_set1_ps(fac[j]);

            midterm_512 = rbsog_set1_ps(MIDTERM);
            midterm_512 = factor * midterm_512;
            Rho_All_0 = rbsog_set1_ps(Rho_All[j][0]);
            Rho_All_1 = rbsog_set1_ps(Rho_All[j][1]);

            Imag = (Cos * Rho_All_1 + Sin * Rho_All_0) * midterm_512 / K2;

            Fx = Fx + qq * Imag * Kx;
            Fy = Fy + qq * Imag * Ky;
            Fz = Fz + qq * Imag * Kz;
        }


        rbsog_store_ps(&F_x[0], Fx);
        rbsog_store_ps(&F_y[0], Fy);
        rbsog_store_ps(&F_z[0], Fz);

        for (int j = 0; j < 16; j++) {
            F[i + j][0] = F[i + j][0] + F_x[j];
            F[i + j][1] = F[i + j][1] + F_y[j];
            F[i + j][2] = F[i + j][2] + F_z[j];
        }
    }
    // if (comm->me == 0)
    // {
    //     cout << "Force part1 : " << F[0][0] << " " << F[0][1] << " " << F[0][2] << endl;
    // }

    // time = MPI_Wtime() - time;
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[3][index] = time * 1000;    
    // }

    // time = MPI_Wtime();
    //Direct calucating the region in Kcut
    int index = 0;
    int Size = (2 * ceil(sqrt(Kcut)) + 1) * (2 * ceil(sqrt(Kcut)) + 1) * (2 * ceil(sqrt(Kcut)) + 1);
    float Kx_Dir[Size][3];
    int bound = ceil(sqrt(Kcut));
    for ( int i = -bound; i <= bound; i++)
    {
        float Kx = pxyz[0] * (i + 0.00);
        for (int j = -bound; j <= bound; j++)
        {
            float Ky = pxyz[1] * (j + 0.00);
            for (int k = -bound; k <= bound; k++)
            {
                float Kz = pxyz[2] * (k + 0.00);

                if ((!(i == 0 && j == 0 && k == 0)) && (i * i + j * j + k * k <= Kcut))
                {
                    Kx_Dir[index][0] = Kx;
                    Kx_Dir[index][1] = Ky;
                    Kx_Dir[index][2] = Kz;
                    index++;
                }
            }
        }
    }
    // if(comm->me == 0 && update->ntimestep == 0)
    //     cout << "The total number of Kx_Dir = " << index << endl;

    float rho[index][2], rho_All[index][2];
    std::vector<float> F_b_sigma(index);
    std::vector<float> F_b_sigma_npt(index);
    for (int i = 0; i < index; i++)
    {
        F_b_sigma[i] = Gaussian_Fourier_Plus(Kx_Dir[i][0], Kx_Dir[i][1], Kx_Dir[i][2], sigma, b, w0, Mmax, coef);
        F_b_sigma_npt[i] = Gaussian_Fourier_Plus_modify(Kx_Dir[i][0], Kx_Dir[i][1], Kx_Dir[i][2], sigma, b, w0, Mmax, coef_npt);
    }
    // if(comm->me == 0 && update->ntimestep == 0)
    //     cout << "F_b = " << F_b_sigma[0] << ' ' << F_b_sigma_npt[0] << endl;

    bool direct_force_on_gpu = false;
#ifdef LMP_GPU
    if (use_gpu_accel && comm->nprocs == 1 && index > 0) {
        if (static_cast<int>(kx_direct_cache_.size()) < index) {
            kx_direct_cache_.resize(index);
            ky_direct_cache_.resize(index);
            kz_direct_cache_.resize(index);
            coeff_direct_cache_.resize(index);
            rho_direct_cos_cache_.resize(index);
            rho_direct_sin_cache_.resize(index);
        }
        for (int i = 0; i < index; i++) {
            kx_direct_cache_[i] = Kx_Dir[i][0];
            ky_direct_cache_[i] = Kx_Dir[i][1];
            kz_direct_cache_[i] = Kx_Dir[i][2];
            coeff_direct_cache_[i] = F_b_sigma[i];
        }
        if (static_cast<int>(fx_gpu_cache_.size()) < nlocal) {
            fx_gpu_cache_.resize(nlocal);
            fy_gpu_cache_.resize(nlocal);
            fz_gpu_cache_.resize(nlocal);
        }
        const float MID_direct_gpu = -qqrd2e / V;
        if (rbsog_gpu_compute_force_direct_single_rank(nlocal, index, X, Y, Z, Q,
                                                      kx_direct_cache_.data(),
                                                      ky_direct_cache_.data(),
                                                      kz_direct_cache_.data(),
                                                      coeff_direct_cache_.data(),
                                                      MID_direct_gpu, fx_gpu_cache_.data(),
                                                      fy_gpu_cache_.data(), fz_gpu_cache_.data(),
                                                      rho_direct_cos_cache_.data(),
                                                      rho_direct_sin_cache_.data()) == 0) {
            for (int i = 0; i < index; i++) {
                rho_All[i][0] = rho_direct_cos_cache_[i];
                rho_All[i][1] = rho_direct_sin_cache_[i];
            }
            for (int i = 0; i < nlocal; i++) {
                F[i][0] += fx_gpu_cache_[i];
                F[i][1] += fy_gpu_cache_[i];
                F[i][2] += fz_gpu_cache_[i];
            }
            direct_force_on_gpu = true;
        }
    }
#endif

    // float *real = new float[4];
    // float *imag = new float[4];
    float sum_real = 0.0;
    float sum_imag = 0.0;
    // float32x4_t Real, Imag, X0, X1, X2, qq, Cos, Sin, Moment, Kx, Ky, Kz;
    if (!direct_force_on_gpu)
    for (int i = 0; i < index; i++)
    {   
        // for (int i = 0; i < 4; i++){
        //     real[i] = 0;
        //     imag[i] = 0;
        // }
        sum_real = 0.0;
        sum_imag = 0.0;
        Kx = rbsog_set1_ps(Kx_Dir[i][0]);
        Ky = rbsog_set1_ps(Kx_Dir[i][1]);
        Kz = rbsog_set1_ps(Kx_Dir[i][2]);
        Real = Imag = rbsog_setzero_ps();

        for (int j = 0; j < nlocal; j += 16)
        {
            // Real = vld1q_f32(&real[0]);
            // Imag = vld1q_f32(&imag[0]);
            X0 = rbsog_load_ps(&X[j]);
            X1 = rbsog_load_ps(&Y[j]);
            X2 = rbsog_load_ps(&Z[j]);
            qq = rbsog_load_ps(&Q[j]);

            //float32x4_t mid0, mid1;
            //mid0 = vmulq_f32(Kx, X0);
            //mid1 = vfmaq_f32(mid0, Ky, X1);
            //Moment = vfmaq_f32(mid1, Kz, X2);
            Moment = Kx * X0 + Ky * X1 + Kz * X2;

            Sin = rbsog_sincos_ps(&Cos, Moment);
            //svml128_sincos_f32(Moment, &Sin, &Cos);

            Real = Real + qq * Cos;
            Imag = Imag + qq * Sin;
        }
        sum_real = rbsog_reduce_add_ps(Real);
        sum_imag = rbsog_reduce_add_ps(Imag);
        rho[i][0] = sum_real;
        rho[i][1] = sum_imag;
    }

    if (!direct_force_on_gpu) {
        if (comm->nprocs == 1) {
            std::memcpy((float *)rho_All, (float *)rho, sizeof(float) * 2 * index);
        } else {
            MPI_Allreduce((float*)rho, (float*)rho_All, 2 * index, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    // if (comm->me == 0 && update->ntimestep  % 100 == 0)
    // {
    //     ofstream outfile;
    //     outfile.open("./rho_All.txt");
    //     for (int i = 0; i < index; i++)
    //         outfile << "rho_All: " << rho_All[i][0] << ", rho_All: " << rho_All[i][1] << endl;
    //     outfile.close();
    // }
    // if (comm->me == 0)
    // {
    //     cout << "The rho_All = " << rho_All[0][0] << " " << rho_All[0][1] << endl;
    // }

    RBSOGVec rho_All_0, rho_All_1;
    // float F_xx[4],F_yy[4],F_zz[4];
    float MID = - qqrd2e / V;
    if (!direct_force_on_gpu)
    for (int m = 0; m < nlocal; m += 16)
    {
        X0 = rbsog_load_ps(&X[m]);
        X1 = rbsog_load_ps(&Y[m]);
        X2 = rbsog_load_ps(&Z[m]);
        qq = rbsog_load_ps(&Q[m]);
        Fx = Fy = Fz = rbsog_setzero_ps();
        for (int i = 0; i < index; i++)
        {
            Kx = rbsog_set1_ps(Kx_Dir[i][0]);
            Ky = rbsog_set1_ps(Kx_Dir[i][1]);
            Kz = rbsog_set1_ps(Kx_Dir[i][2]);
            Real = Imag = rbsog_setzero_ps();
            RBSOGVec f_b_sigma = rbsog_set1_ps(F_b_sigma[i]);
            RBSOGVec coeff = rbsog_set1_ps(MID);
            //RBSOGVec mmid0, mmid1;
            // Kx2 = vmulq_f32(Kx, Kx);
            // Ky2 = vfmaq_f32(Kx2, Ky, Ky);
            // K2 = vfmaq_f32(Ky2, Kz, Kz);

            //mmid0 = vmulq_f32(Kx, X0);
            //mmid1 = vfmaq_f32(mmid0, Ky, X1);
            //moment_128 = vfmaq_f32(mmid1, Kz, X2);
            //moment_128 = vnegq_f32(moment_128);
            moment_512 = -(Kx * X0 + Ky * X1 + Kz * X2);

            Sin = rbsog_sincos_ps(&Cos, moment_512);
            //svml128_sincos_f32(moment_128, &Sin, &Cos);

            //midterm_128 = vmulq_f32(f_b_sigma, coeff);
            midterm_512 = f_b_sigma * coeff;

            rho_All_0 = rbsog_set1_ps(rho_All[i][0]);
            rho_All_1 = rbsog_set1_ps(rho_All[i][1]);

            //float32x4_t imag_cos, imag_temp;
            //imag_cos = vmulq_f32(Cos, rho_All_1);
            //imag_temp = vfmaq_f32(imag_cos, Sin, rho_All_0);
            //Imag = vmulq_f32(imag_temp, midterm_128);

            Imag = (Cos * rho_All_1 + Sin * rho_All_0) * midterm_512;
            // Imag = vdivq_f32(Imag, K2);


            //float32x4_t qqimag;
            //qqimag = vmulq_f32(qq, Imag);
            Fx = Fx + qq * Imag * Kx;
            Fy = Fy + qq * Imag * Ky;
            Fz = Fz + qq * Imag * Kz;
        }
        rbsog_store_ps(&F_x[0], Fx);
        rbsog_store_ps(&F_y[0], Fy);
        rbsog_store_ps(&F_z[0], Fz);

        for (int j = 0; j < 16; j++) {
            F[m + j][0] = F[m + j][0] + F_x[j];
            F[m + j][1] = F[m + j][1] + F_y[j];
            F[m + j][2] = F[m + j][2] + F_z[j];
        }

        
    }
        // }
        // else
        // {
        //     error->all(FLERR, "Need RankID >=1"); 
        // }
    
    // time = MPI_Wtime() - time;
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[4][index] = time * 1000;    
    // }

    // if (comm->me == 0)
    // {
    //     cout << "Force part2 : " << F[0][0] << " " << F[0][1] << " " << F[0][2] << endl;
    // }

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
    

    // if (comm->me == 0)
    // {
    //     for (int i = 0; i < 10; i++)
    //     {
    //         cout << "Force " << i << " =   " << f[i][0] << "    " << f[i][1] << "     " << f[i][2] << endl;
    //     }
    // }


    /*time = MPI_Wtime() - time;
    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
    }*/

    // if (comm->me == 0 && update->ntimestep == 1000) {
    //     ofstream outfile;
    //     outfile.open("./Time_Kspace_RBSOGTIP4P.txt");
    //     //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
    //     for (int i = 0; i < 1000; i++)
    //         outfile << TimeSet[i] << endl;
    //     outfile.close();
    // }

    // if (comm->me == this_rank && update->ntimestep == 2000) {
    //     ofstream outfile;
    //     outfile.open("./Time_Kspace_RBSOGTIP4P_" + std::to_string(P) + ".txt");
    //     for (int i = 0; i < 10; i++)
    //         outfile << "Sampling time: " << TimeSet[0][i] << endl
    //             << "Fac calulating time: " << TimeSet[1][i] << endl
    //             << "Rho calulating time: " << TimeSet[2][i] << endl
    //             << "Force calulating time: " << TimeSet[3][i] << endl
    //             << "Force in Kcut calulating time: " << TimeSet[4][i] << endl;
    //     outfile.close();
    // }
    // sum global energy across Kspace vevs and add in volume-dependent term
		const double qscale = qqrd2e * scale;

        if (eflag_global) {
			float KXX[3];
            // double coeff = 0.5 * w0 + logf(b) * (1 - powf(b, -Mmax)) / (sqrtf(2 * MY_PI) * sigma * (b - 1));
            double coeff = (logf(b)/(sqrtf(2 * MY_PI) * sigma)) * (w0 + (1 - powf(b, -Mmax)) / (b - 1));
			for (int i = 0; i < P; i++)
			{
				KXX[0] = K[i][0];
				KXX[1] = K[i][1];
				KXX[2] = K[i][2];

				// energy += fac[i] * (S / (2 * (P + 0.00) * V)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (KXX[0] * KXX[0] + KXX[1] * KXX[1] + KXX[2] * KXX[2]);
				energy += fac[i] * (S0 / (2 * (P + 0.00) * V)) * (Rho_All[i][0] * Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1]) / (KXX[0] * KXX[0] + KXX[1] * KXX[1] + KXX[2] * KXX[2]);
            }
            // if(comm->me == 0 && update->ntimestep % 100 == 0)
            //     cout << "The energy = " << energy << endl;
            if(Kcut > 0){
                for (int i = 0; i < index; i++)
                {
                    energy += (1.0 / (2 * V)) * F_b_sigma[i] * (rho_All[i][0] * rho_All[i][0] + rho_All[i][1] * rho_All[i][1]);
                }
            }
            // if(comm->me == 0 && update->ntimestep % 100 == 0)
            //     cout << "The energy = " << energy * qscale << endl;
			energy -= coeff * qsqsum;// self energy
			energy *= qscale;
            // if(comm->me == 0 && update->ntimestep % 100 == 0)
            //     cout << "The energy = " << energy << endl;
		}


// global virial
		if (vflag_global) {
			float KKXX[3], KKXX_npt[3];
            float K2, K2_npt, K4_npt;
            float Moment_Term = (S0 / (P + 0.00)) * qscale / ( 2 * (V+0.00));
            float Moment_Term_npt = - (S_npt0 / (P + 0.00)) * qscale / ( 4 * (V+0.00));
            float c1 = qscale / ( 2 * (V+0.00));
            float c2 = - qscale / ( 4 * (V+0.00));
            int indx;
#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd
#endif
			for (int i = 0; i < P; i++)
			{
				KKXX[0] = K[i][0];
				KKXX[1] = K[i][1];
				KKXX[2] = K[i][2];
                K2 = KKXX[0] * KKXX[0] + KKXX[1] * KKXX[1] + KKXX[2] * KKXX[2];

                indx = idx_npt_all[i];
                // if (indx >= P && comm->me == 0){
                //     cout << "Invalid indx = " << indx << ", with i = " << i << endl;
                //     cout << "idx_npt = " << idx_npt[i] << endl;
                //     error->all(FLERR, "Invalid indx!");
                // }
                    
                // int idx = i;
                KKXX_npt[0] = K_npt[i][0];
                KKXX_npt[1] = K_npt[i][1];
                KKXX_npt[2] = K_npt[i][2];
                K2_npt = KKXX_npt[0] * KKXX_npt[0] + KKXX_npt[1] * KKXX_npt[1] + KKXX_npt[2] * KKXX_npt[2];
                K4_npt = K2_npt * K2_npt;
				float coef1 = fac[i] * (Rho_All[i][0]* Rho_All[i][0] + Rho_All[i][1] * Rho_All[i][1])/ K2;
                float coef2 = fac_npt[i] * (Rho_All[indx][0]* Rho_All[indx][0] + Rho_All[indx][1]* Rho_All[indx][1])/ K4_npt;
				virial[0] += coef1 * Moment_Term + coef2 * Moment_Term_npt * KKXX_npt[0] * KKXX_npt[0];
                virial[1] += coef1 * Moment_Term + coef2 * Moment_Term_npt * KKXX_npt[1] * KKXX_npt[1];
                virial[2] += coef1 * Moment_Term + coef2 * Moment_Term_npt * KKXX_npt[2] * KKXX_npt[2];
				// virial[1] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[1] * KKXX[1]);
				// virial[2] += coef1 * Moment_Term * (1 - coef2 * 2 * KKXX[2] * KKXX[2]);
                virial[3] += coef2 * Moment_Term_npt * KKXX_npt[0] * KKXX_npt[1];
                virial[4] += coef2 * Moment_Term_npt * KKXX_npt[0] * KKXX_npt[2];
                virial[5] += coef2 * Moment_Term_npt * KKXX_npt[1] * KKXX_npt[2];
				// virial[3] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[1]);
				// virial[4] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[0] * KKXX[2]);
				// virial[5] += coef1 * Moment_Term * (- coef2 * 2 * KKXX[1] * KKXX[2]);

			}
            for (int i = 0; i < index; i++)
            {   
                float rho2 = rho_All[i][0] * rho_All[i][0] + rho_All[i][1] * rho_All[i][1];
                virial[0] += c1 * F_b_sigma[i] * rho2 + c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][0] * Kx_Dir[i][0];
                virial[1] += c1 * F_b_sigma[i] * rho2 + c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][1] * Kx_Dir[i][1];
                virial[2] += c1 * F_b_sigma[i] * rho2 + c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][2] * Kx_Dir[i][2];
                virial[3] += c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][0] * Kx_Dir[i][1];
                virial[4] += c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][0] * Kx_Dir[i][2];
                virial[5] += c2 * F_b_sigma_npt[i] * rho2 * Kx_Dir[i][1] * Kx_Dir[i][2];
            }

            // if(comm->me == 0 && update->ntimestep % 100 == 0)
            //     cout << "The virial = " << virial[0] << "  " << virial[1] << "  " << virial[2] << "  " << virial[3] << "  " << virial[4] << "  " << virial[5] << endl;
		} 
}

// double RBSOGTIP4P::memory_usage()
// {
//     return 1.0;

// }

void RBSOGTIP4P::find_M(int i, int &iH1, int &iH2, double *xM)
{
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
}
