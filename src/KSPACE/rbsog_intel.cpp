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

#include "rbsog_intel.h"
#include <mpi.h>
#include <mathimf.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "update.h"
#include "pair.h"

#include <time.h>
#include <stdlib.h>


#include <immintrin.h> 
#include <unistd.h>
#include <stdexcept>
#include <cassert>
#include <cstddef> 
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <mkl.h>
#include <random>

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<double> dis(0, 1);
static std::uniform_real_distribution<> dis_uniform(-1, 1);

#define SMALL 0.00001


/* ----------------------------------------------------------------------
   required functions
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

RBSOG::RBSOG(LAMMPS* lmp) : KSpace(lmp)
{
    TimeSet = new double [1000];
    // for (int i = 0; i < 5; i++)
    // {
    //     TimeSet[i] = new double[10];
    // }
    TimeSet_sampling = new double[1000];
    rbsogflag = 1;
}

void RBSOG::settings(int narg, char** arg)
{
    MPI_Comm_size(MPI_COMM_WORLD, &RankID);
    if (narg != 5) error->all(FLERR, "Illegal kspace_style UserRBSOG command");
    b = fabs(utils::numeric(FLERR, arg[0], false, lmp));
    sigma = fabs(utils::numeric(FLERR, arg[1], false, lmp));
    Mmax = fabs(utils::numeric(FLERR, arg[2], false, lmp));
    P = fabs(utils::numeric(FLERR, arg[3], false, lmp));
    Kcut = fabs(utils::numeric(FLERR, arg[4], false, lmp));
    Kcut = 1;
    Step = 0;
    P = int(ceil(P / 16)) * 16;
    K_Sample = new int* [P];
    for (int i = 0; i < P; i++)
    {
        K_Sample[i] = new int[3];
    }
    idx_npt = new int[P];
    for (int i = 0; i < P; i++)
    {
        idx_npt[i] = 0;
    }
    sl = new float[Mmax];
    coef = new float[Mmax];
    coef_npt = new float[Mmax];
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

RBSOG::~RBSOG()
{
    for (int i = 0; i < P; i++) {
    	delete[] K_Sample[i];
	}
	delete[] K_Sample;
    delete[] idx_npt;
    delete[] sl;
    delete[] coef;
    delete[] coef_npt;
}

/* ---------------------------------------------------------------------- */

void RBSOG::init()
{
    if (comm->me == 0) utils::logmesg(lmp, "UserRBSOG initialization ...\n");

    // error check

    triclinic_check();
    if (domain->dimension == 2)
        error->all(FLERR, "Cannot use SOG with 2d simulation");

    if (!atom->q_flag) error->all(FLERR, "Kspace style requires atom attribute q");

    if (slabflag == 0 && domain->nonperiodic > 0)
        error->all(FLERR, "Cannot use non-periodic boundaries with UserRBSOG");
    if (slabflag) {
        if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
            domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
            error->all(FLERR, "Incorrect boundaries with slab UserRBSOG");
        if (domain->triclinic)
            error->all(FLERR, "Cannot (yet) use UserRBE with triclinic box "
                "and slab correction");
    }

    // compute two charge force

    two_charge();

    // extract short-range Coulombic cutoff from pair style

    triclinic = domain->triclinic;
    pair_check();

    // compute qsum & qsqsum and warn if not charge-neutral

    scale = 1.0;
    qqrd2e = force->qqrd2e;
    qsum_qsq();
    //if (fabs(qsum) > SMALL) 
    //{
        //error->all(FLERR, "System is not charge neutral, net charge = %g",qsum);
    //}
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
    r0 = rcut / sigma;
    w0 = Compute_W0(r0, b);
    sl[0] = sqrt(2.0) * sigma;
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    float b2 = b * b;
    float b4 = b2 * b2;
    coef[0] = 4 * MY_PI * log(b) * w0 * sigma2;
    coef_npt[0] = 8 * MY_PI * log(b) * w0 * sigma4;

    for (int i = 1; i < Mmax; ++i) {
        sl[i] = sl[i-1] * b;
        if( i == 1){
            coef[i] = 4 * MY_PI * log(b) * sigma2* b2;
            coef_npt[i] = 8 * MY_PI * log(b) * sigma4 * b4;
        }
        coef[i] = coef[i-1] * b2;
        coef_npt[i] = coef_npt[i-1] * b4;
    }
    setup();
}

/* ----------------------------------------------------------------------
   adjust UserRBE coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void RBSOG::setup()
{
    MPI_Comm_size(MPI_COMM_WORLD, &RankID);

    // if (comm->me == 0 && Step == 0) cout << "w0 = " << w0 << endl;
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd * slab_volfactor;
    double prd = max(xprd, max(yprd, zprd));
    float sum = 0.00;
    float sum_npt = 0.00;
    float sum_1 = 0.00;
    float sum_npt_1 = 0.00;
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    float b2 = b * b;
    float b4 = b2 * b2;
    double log_b = log(b);
    double bool_bound = 1.0;
    Step = 0;
    for (int i = 0; i < Kcut + 1; i++)
    {
        float Kx = (2 * MY_PI / xprd) * (i + 0.00);
        for (int j = 0; j < Kcut + 1; j++)
        {
            float Ky = (2 * MY_PI / yprd) * (j + 0.00);
            for (int k = 0; k < Kcut + 1; k++)
            {
                float Kz = (2 * MY_PI / zprd) * (k + 0.00);
                if ((i * i + j * j + k * k) <= Kcut)
                {
                    float K2 = Kx * Kx + Ky * Ky + Kz * Kz;
                    float K4 = K2 * K2;
                    sum_1 = sum_1 + 2 * Gaussian(i, j, k, xprd, yprd, zprd, sigma, b, w0, Mmax, coef)*K2;
                    sum_npt_1 = sum_npt_1 + 2 * Gaussian_modify(i, j, k, xprd, yprd, zprd, sigma, b, w0, Mmax, coef_npt) * K4;
                }
            }
        }
    }

    for (int i = 0; i < Mmax; ++i) 
    {
        double Hx = 0, Hy = 0, Hz = 0;
        double Yx = 0, Yy = 0, Yz = 0;
        double Yxx = 0, Yyy = 0, Yzz = 0;

        int j = 1;
        bool_bound = 1.0;
        while (bool_bound > 1e-16) {
            double kx = MY_PI * j / xprd;
            double ky = MY_PI * j / yprd;
            double kz = MY_PI * j / zprd;
            double k_min = MY_PI * j / prd;
            double k_min2 = k_min * k_min;
            double k_min4 = k_min2 * k_min2;
            double kx2_s = kx * sl[i] * kx * sl[i];
            double ky2_s = ky * sl[i] * ky * sl[i];
            double kz2_s = kz * sl[i] * kz * sl[i];
            double kx2 = 4 * kx * kx;
            double ky2 = 4 * ky * ky;
            double kz2 = 4 * kz * kz;
            double kx4 = 16 * kx * kx * kx * kx;
            double ky4 = 16 * ky * ky * ky * ky;
            double kz4 = 16 * kz * kz * kz * kz;
            Hx += 2 * exp(-kx2_s);
            Yx += 2 * exp(-kx2_s) * kx2;
            Yxx += 2 * exp(-kx2_s) * kx4;
            Hy += 2 * exp(-ky2_s);
            Yy += 2 * exp(-ky2_s) * ky2;
            Yyy += 2 * exp(-ky2_s) * ky4;
            Hz += 2 * exp(-kz2_s);
            Yz += 2 * exp(-kz2_s) * kz2;
            Yzz += 2 * exp(-kz2_s) * kz4;
            ++j;
            bool_bound = exp(4 * i * log_b - k_min2 * sl[i] * sl[i]) * k_min4 * sigma4;
        }
        Hx += 1;
        Hy += 1;
        Hz += 1;

        sum += coef[i] * (Yx * Hy * Hz + Hx * Yy * Hz + Hx * Hy * Yz);
        sum_npt += coef_npt[i] * (Yxx * Hy * Hz + Hx * Yyy * Hz + Hx * Hy * Yzz + 2 * Yx * Yy * Hz + 2 * Yx * Hy * Yz + 2 * Hx * Yy * Yz);
    }
    
    S = sum - sum_1;
    S_npt = sum_npt - sum_npt_1;

    // Kmax = 20;
    // float sum_test = 0.00;
    // float sum_npt_test = 0.00;
    // for (int i = -Kmax; i < Kmax + 1; i++)
    // {
    //     float Kx = (2 * MY_PI / xprd) * (i + 0.00);
    //     for (int j = -Kmax; j < Kmax + 1; j++)
    //     {
    //         float Ky = (2 * MY_PI / yprd) * (j + 0.00);
    //         for (int k = -Kmax; k < Kmax + 1; k++)
    //         {
    //             float Kz = (2 * MY_PI / zprd) * (k + 0.00);
    //             if ((!(i == 0 && j == 0 && k == 0)) && (i * i + j * j + k * k) > Kcut)
    //             {
    //                 float K2 = Kx * Kx + Ky * Ky + Kz * Kz;
    //                 sum_test = sum_test + Gaussian(i, j, k, xprd, yprd, zprd, sigma, b, w0, Mmax)*K2;
    //                 sum_npt_test = sum_npt_test + Gaussian_modify(i, j, k, xprd, yprd, zprd, sigma, b, w0, Mmax) * K2 * K2;
    //             }
    //         }
    //     }
    // }


    if (comm->me == 0 && update->ntimestep == 0)
    {
        cout << "The total normalized number S = " << S << endl;
        cout << "The total normalized number S_npt = " << S_npt << endl;
    }

}

/* ----------------------------------------------------------------------
   compute the UserRBE long-range force, energy, virial
------------------------------------------------------------------------- */

void RBSOG::compute(int eflag, int vflag)
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

    float fac[P];
    float fac_npt[P];
    for(int i = 0; i < P; i++)
    {
        fac[i] = 1;
        fac_npt[i] = 1;
    }
    // double xprd0, yprd0, zprd0;
    // float S0, S_npt0;

    float pxyz[3] = {  static_cast<float>(2 * MY_PI / xprd),  static_cast<float>(2 * MY_PI / yprd),  static_cast<float>(2 * MY_PI / zprd) };
    float Rho[P][2], Rho_All[P][2];
    float midterm[P][3];

    double time;
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();

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
        int mx[5*P], my[5*P], mz[5*P];
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
    int K_tmp[P][3];
    int idx_npt_all[P];
    if (comm->me == this_rank)
    {
        for (int i = 0; i < P; i++)
        {
            K_tmp[i][0] = K_Sample[i][0];
            K_tmp[i][1] = K_Sample[i][1];
            K_tmp[i][2] = K_Sample[i][2];
            idx_npt_all[i] = idx_npt[i];
        }
    }
    // time = MPI_Wtime() - time; 
    // if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //     int index = (update->ntimestep - 1000) / 100;
    //     TimeSet[0][index] = time * 1000;
    // }

    MPI_Bcast(K_tmp, 3 * P, MPI_INT, this_rank, MPI_COMM_WORLD);
    // MPI_Bcast((float*)K_npt, 3 * P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);
    MPI_Bcast(idx_npt_all, P, MPI_INT, this_rank, MPI_COMM_WORLD);
    float K[P][3];
    float K_npt[P][3];
    for (int i = 0; i < P; i++)
    {
        K[i][0] = K_tmp[i][0] * pxyz[0];
        K[i][1] = K_tmp[i][1] * pxyz[1];
        K[i][2] = K_tmp[i][2] * pxyz[2];
        int id = idx_npt_all[i];
        K_npt[i][0] = K_tmp[id][0] * pxyz[0];
        K_npt[i][1] = K_tmp[id][1] * pxyz[1];
        K_npt[i][2] = K_tmp[id][2] * pxyz[2];
    }

    //stop

    //time = MPI_Wtime() - time;
    //if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
    //    TimeSet_sampling[update->ntimestep - 0] = time * 1000;
    //}

    //if (comm->me == 0 && update->ntimestep == 1000) {
    //    ofstream outfile;
    //    outfile.open("./Time_Kspace_RBSOG_sampling.txt");
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


    float X[int(ceil((nlocal + 0.0) / 16.0)) * 16], Y[int(ceil((nlocal + 0.0) / 16.0)) * 16], Z[int(ceil((nlocal + 0.0) / 16.0)) * 16];
    float F[int(ceil((nlocal + 0.0) / 16.0)) * 16][3];
    float Q[int(ceil((nlocal + 0.0) / 16.0)) * 16];
    for (int i = 0; i < int(ceil((nlocal + 0.0) / 16.0)) * 16; i++)
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
        __m512 Kx, Ky, Kz, Kx0, Ky0, Kz0;
        __m512 Kx_npt, Ky_npt, Kz_npt, Kx0_npt, Ky0_npt, Kz0_npt;
        __m512 K2, K2_0, K2_npt, K2_npt_0, K4, K4_0;
        __m512 Lx_ratio, Ly_ratio, Lz_ratio;
        __m512 S_ratio, S_npt_ratio;
        Lx_ratio = _mm512_set1_ps(xprd / xprd0);
        Ly_ratio = _mm512_set1_ps(yprd / yprd0);
        Lz_ratio = _mm512_set1_ps(zprd / zprd0);
        S_ratio = _mm512_set1_ps(S0 / S);
        S_npt_ratio = _mm512_set1_ps(S_npt0 / S_npt);
        Kx = _mm512_load_ps(&KxKx0[i]);
        Ky = _mm512_load_ps(&KxKx1[i]);
        Kz = _mm512_load_ps(&KxKx2[i]);
        Kx_npt = _mm512_load_ps(&KxKx0_npt[i]);
        Ky_npt = _mm512_load_ps(&KxKx1_npt[i]);
        Kz_npt = _mm512_load_ps(&KxKx2_npt[i]);
        Kx0 = Kx * Lx_ratio;
        Ky0 = Ky * Ly_ratio;
        Kz0 = Kz * Lz_ratio;
        Kx0_npt = Kx_npt * Lx_ratio;
        Ky0_npt = Ky_npt * Ly_ratio;
        Kz0_npt = Kz_npt * Lz_ratio;
        K2 = Kx * Kx + Ky * Ky + Kz * Kz;
        K2_0 = Kx0 * Kx0 + Ky0 * Ky0 + Kz0 * Kz0;
        K2_npt = Kx_npt * Kx_npt + Ky_npt * Ky_npt + Kz_npt * Kz_npt;
        K2_npt_0 = Kx0_npt * Kx0_npt + Ky0_npt * Ky0_npt + Kz0_npt * Kz0_npt;
        K4 = K2_npt * K2_npt;
        K4_0 = K2_npt_0 * K2_npt_0;

        __m512 mid, mid0, mid_npt, mid0_npt;
        mid = Gaussian_Fourier_Plus_AVX(Kx, Ky, Kz, sigma, b, w0, Mmax, coef) * K2;
        mid0 = Gaussian_Fourier_Plus_AVX(Kx0, Ky0, Kz0, sigma, b, w0, Mmax, coef) * K2_0;
        mid_npt = Gaussian_Fourier_Plus_AVX(Kx_npt, Ky_npt, Kz_npt, sigma, b, w0, Mmax, coef_npt) * K4;
        mid0_npt = Gaussian_Fourier_Plus_AVX(Kx0_npt, Ky0_npt, Kz0_npt, sigma, b, w0, Mmax, coef_npt) * K4_0;

        __m512 fac_tmp, fac_npt_tmp;
        fac_tmp = S_ratio * mid / mid0;
        fac_npt_tmp = S_npt_ratio * mid_npt / mid0_npt;

        _mm512_store_ps(&fac[i], fac_tmp);
        _mm512_store_ps(&fac_npt[i], fac_npt_tmp);
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
    __m512 Real, Imag, X0, X1, X2, qq, Cos, Sin,
        Moment, Kx, Ky, Kz;

    for (int i = 0; i < P; i += 16)
    {

        Real = Imag = _mm512_setzero_ps();
        Kx = _mm512_load_ps(&KxKx0[i]);
        Ky = _mm512_load_ps(&KxKx1[i]);
        Kz = _mm512_load_ps(&KxKx2[i]);

        for (int j = 0; j < nlocal; j++)
        {
            X0 = _mm512_set1_ps(X[j]);
            X1 = _mm512_set1_ps(Y[j]);
            X2 = _mm512_set1_ps(Z[j]);
            qq = _mm512_set1_ps(Q[j]);


           //__mm512 mid0, mid1;
           // mid0 = vmulq_f32(Kx, X0);
           // mid1 = vfmaq_f32(mid0, Ky, X1);
           // Moment = vfmaq_f32(mid1, Kz, X2);
            Moment = Kx * X0 + Ky * X1 + Kz * X2;

            Sin = _mm512_sincos_ps(&Cos, Moment);
            //svml128_sincos_f32(Moment, &Sin, &Cos);


            Real = Real + qq * Cos;
            Imag = Imag + qq * Sin;
        }

        _mm512_store_ps(&Rho_Cos[i], Real);
        _mm512_store_ps(&Rho_Sin[i], Imag);
    }

    for (int i = 0; i < P; i++)
    {
        Rho[i][0] = Rho_Cos[i];
        Rho[i][1] = Rho_Sin[i];
    }

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
    float rho[index][2], rho_All[index][2];
    float *F_b_sigma = new float[index];
    float *F_b_sigma_npt = new float[index];
    for (int i = 0; i < index; i++)
    {
        F_b_sigma[i] = Gaussian_Fourier_Plus(Kx_Dir[i][0], Kx_Dir[i][1], Kx_Dir[i][2], sigma, b, w0, Mmax, coef);
        F_b_sigma_npt[i] = Gaussian_Fourier_Plus_modify(Kx_Dir[i][0], Kx_Dir[i][1], Kx_Dir[i][2], sigma, b, w0, Mmax, coef_npt);
    }
    // if(comm->me == 0 && update->ntimestep == 0)
    //     cout << "F_b = " << F_b_sigma[0] << ' ' << F_b_sigma_npt[0] << endl;

    // float *real = new float[4];
    // float *imag = new float[4];
    float sum_real = 0.0;
    float sum_imag = 0.0;
    // float32x4_t Real, Imag, X0, X1, X2, qq, Cos, Sin, Moment, Kx, Ky, Kz;
    for (int i = 0; i < index; i++)
    {   
        // for (int i = 0; i < 4; i++){
        //     real[i] = 0;
        //     imag[i] = 0;
        // }
        sum_real = 0.0;
        sum_imag = 0.0;
        Kx = _mm512_set1_ps(Kx_Dir[i][0]);
        Ky = _mm512_set1_ps(Kx_Dir[i][1]);
        Kz = _mm512_set1_ps(Kx_Dir[i][2]);
        Real = Imag = _mm512_setzero_ps();

        for (int j = 0; j < nlocal; j += 16)
        {
            // Real = vld1q_f32(&real[0]);
            // Imag = vld1q_f32(&imag[0]);
            X0 = _mm512_load_ps(&X[j]);
            X1 = _mm512_load_ps(&Y[j]);
            X2 = _mm512_load_ps(&Z[j]);
            qq = _mm512_load_ps(&Q[j]);

            //float32x4_t mid0, mid1;
            //mid0 = vmulq_f32(Kx, X0);
            //mid1 = vfmaq_f32(mid0, Ky, X1);
            //Moment = vfmaq_f32(mid1, Kz, X2);
            Moment = Kx * X0 + Ky * X1 + Kz * X2;

            Sin = _mm512_sincos_ps(&Cos, Moment);
            //svml128_sincos_f32(Moment, &Sin, &Cos);

            Real = Real + qq * Cos;
            Imag = Imag + qq * Sin;
        }
        sum_real = _mm512_reduce_add_ps(Real);
        sum_imag = _mm512_reduce_add_ps(Imag);
        rho[i][0] = sum_real;
        rho[i][1] = sum_imag;
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

    MPI_Allreduce((float*)Rho, (float*)Rho_All, 2 * P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce((float*)rho, (float*)rho_All, 2 * index, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // time = MPI_Wtime();
    double MIDTERM = - (S0 / (P + 0.00)) * qqrd2e / V;

    __m512 Fx, Fy, Fz, K2, moment_512, midterm_512, Rho_All_0, Rho_All_1, factor;
    float F_x[16],F_y[16],F_z[16];

    for (int i = 0; i < nlocal; i += 16)
    {

        X0 = _mm512_load_ps(&X[i]);
        X1 = _mm512_load_ps(&Y[i]);
        X2 = _mm512_load_ps(&Z[i]);
        qq = _mm512_load_ps(&Q[i]);

        Fx = Fy = Fz = _mm512_setzero_ps();

        for (int j = 0; j < P; j++)
        {

            Kx = _mm512_set1_ps(KxKx0[j]);
            Ky = _mm512_set1_ps(KxKx1[j]);
            Kz = _mm512_set1_ps(KxKx2[j]);

            K2 = Kx * Kx + Ky * Ky + Kz * Kz;

            moment_512 = -(Kx * X0 + Ky * X1 + Kz * X2);

            Sin = _mm512_sincos_ps(&Cos, moment_512);

            factor = _mm512_set1_ps(fac[j]);

            midterm_512 = _mm512_set1_ps(MIDTERM);
            midterm_512 = factor * midterm_512;
            Rho_All_0 = _mm512_set1_ps(Rho_All[j][0]);
            Rho_All_1 = _mm512_set1_ps(Rho_All[j][1]);

            Imag = (Cos * Rho_All_1 + Sin * Rho_All_0) * midterm_512 / K2;

            Fx = Fx + qq * Imag * Kx;
            Fy = Fy + qq * Imag * Ky;
            Fz = Fz + qq * Imag * Kz;
        }


        _mm512_store_ps(&F_x[0], Fx);
        _mm512_store_ps(&F_y[0], Fy);
        _mm512_store_ps(&F_z[0], Fz);

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

    //  time = MPI_Wtime() - time;
    //  if ((update->ntimestep >= 1000) && (update->ntimestep < 2000)&& (update->ntimestep%100 == 0)) {
    //      int index = (update->ntimestep - 1000) / 100;
    //      TimeSet[index] = time * 1000;    
    // }

    // time = MPI_Wtime();
    //Direct calucating the region in Kcut
    // if(comm->me == 0 && update->ntimestep == 0)
    //     cout << "The total number of Kx_Dir = " << index << endl;

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

    __m512 rho_All_0, rho_All_1;
    // float F_xx[4],F_yy[4],F_zz[4];
    float MID = - qqrd2e / V;

    for (int m = 0; m < nlocal; m += 16)
    {
        X0 = _mm512_load_ps(&X[m]);
        X1 = _mm512_load_ps(&Y[m]);
        X2 = _mm512_load_ps(&Z[m]);
        qq = _mm512_load_ps(&Q[m]);
        Fx = Fy = Fz = _mm512_setzero_ps();
        for (int i = 0; i < index; i++)
        {
            Kx = _mm512_set1_ps(Kx_Dir[i][0]);
            Ky = _mm512_set1_ps(Kx_Dir[i][1]);
            Kz = _mm512_set1_ps(Kx_Dir[i][2]);
            Real = Imag = _mm512_setzero_ps();
            __m512 f_b_sigma = _mm512_set1_ps(F_b_sigma[i]);
            __m512 coeff = _mm512_set1_ps(MID);
            //__m512 mmid0, mmid1;
            // Kx2 = vmulq_f32(Kx, Kx);
            // Ky2 = vfmaq_f32(Kx2, Ky, Ky);
            // K2 = vfmaq_f32(Ky2, Kz, Kz);

            //mmid0 = vmulq_f32(Kx, X0);
            //mmid1 = vfmaq_f32(mmid0, Ky, X1);
            //moment_128 = vfmaq_f32(mmid1, Kz, X2);
            //moment_128 = vnegq_f32(moment_128);
            moment_512 = -(Kx * X0 + Ky * X1 + Kz * X2);

            Sin = _mm512_sincos_ps(&Cos, moment_512);
            //svml128_sincos_f32(moment_128, &Sin, &Cos);

            //midterm_128 = vmulq_f32(f_b_sigma, coeff);
            midterm_512 = f_b_sigma * coeff;

            rho_All_0 = _mm512_set1_ps(rho_All[i][0]);
            rho_All_1 = _mm512_set1_ps(rho_All[i][1]);

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
        _mm512_store_ps(&F_x[0], Fx);
        _mm512_store_ps(&F_y[0], Fy);
        _mm512_store_ps(&F_z[0], Fz);

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
        f[i][0] += F[i][0];
        f[i][1] += F[i][1];
        f[i][2] += F[i][2];
    }
    

    // if (comm->me == 0)
    // {
    //     for (int i = 0; i < 10; i++)
    //     {
    //         cout << "Force " << i << " =   " << f[i][0] << "    " << f[i][1] << "     " << f[i][2] << endl;
    //     }
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
    }

    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_Kspace_RBSOG.txt");
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
    }

    // if (comm->me == this_rank && update->ntimestep == 2000) {
    //     ofstream outfile;
    //     outfile.open("./Time_Kspace_RBSOG_" + std::to_string(P) + ".txt");
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

double RBSOG::memory_usage()
{
    return 1.0;

}

inline float RBSOG::G_sigma(float sigma, float r)
{
    return expf(-r * r / (2 * sigma * sigma)) / sqrt(2 * MY_PI * sigma * sigma);
}
float RBSOG::Compute_W0(float r0, float b)
{
    float sum = 0.00;
    for (int i = 1; i < 200; i++)
    {
        sum = sum + powf(b, float(-i+0.00)) * G_sigma(1.0, powf(b, float(-i + 0.00)) * r0);
    }
    float W0 = (1.0 / G_sigma(1.0, r0)) * ((1.0 / (2 * logf(b) * r0)) - sum);
    return W0;
}
float RBSOG::Gaussian(int kx, int ky, int kz, float Lx, float Ly, float Lz, float sigma, float b, float w0, int Mmax, float* coef)
{
    float k2 = ((kx + 0.00) * 2 * MY_PI / Lx) * ((kx + 0.00) * 2 * MY_PI / Lx) + ((ky + 0.00) * 2 * MY_PI / Ly) * ((ky + 0.00) * 2 * MY_PI / Ly) + ((kz + 0.00) * 2 * MY_PI / Lz) * ((kz + 0.00) * 2 * MY_PI / Lz);
    float sum = 0.00;
    double b2 = b * b;
    for (int i = 0; i < Mmax; i++)
    {
        sum = sum + coef[i] * expf(-(powf(b2, (i + 0.0)) * sigma * sigma ) * k2 / 2.0);
        // sum = sum + 4 * MY_PI * sigma * sigma * logf(b) * powf(b, float(2 * (i+0.0))) * expf(-(powf(b, float(2 * (i + 0.0))) * sigma * sigma ) * k2 / 2.0);
    }
    return sum;
}
float RBSOG::Gaussian_modify(int kx, int ky, int kz, float Lx, float Ly, float Lz, float sigma, float b, float w0, int Mmax, float* coef_npt)
{
    float k2 = ((kx + 0.00) * 2 * MY_PI / Lx) * ((kx + 0.00) * 2 * MY_PI / Lx) + ((ky + 0.00) * 2 * MY_PI / Ly) * ((ky + 0.00) * 2 * MY_PI / Ly) + ((kz + 0.00) * 2 * MY_PI / Lz) * ((kz + 0.00) * 2 * MY_PI / Lz);
    float sum = 0.00;
    double b2 = b * b;
    for (int i = 0; i < Mmax; i++)
    {
        sum = sum + coef_npt[i] * expf(-(powf(b2, float((i + 0.0))) * sigma * sigma ) * k2 / 2.0);
        // sum = sum + 4 * MY_PI * sigma * sigma * logf(b) * powf(b, float(2 * (i+0.0))) * expf(-(powf(b, float(2 * (i + 0.0))) * sigma * sigma ) * k2 / 2.0);
    }
    return sum;
}

double RBSOG::randn_box_muller_linear_congruential(const double Mean, const double SquareMargin)
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
	// z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * SquareMargin + Mean;
}

double RBSOG::MH_D_Modify(int xx, double factor)
{
    double qes;
    if (xx == 0)
        qes = erf(0.5* factor);
    else
        qes = 0.5 * (erf((abs(xx) + 0.5) * factor) - erf((abs(xx) - 0.5) * factor));

    return qes;
}

float RBSOG::Gaussian_Fourier_Plus(float Kx, float Ky, float Kz, float sigma, float b, float w0, int Mmax, float* coef)
{
    float k2 = Kx * Kx + Ky * Ky + Kz * Kz;
    float sum = 0.00;
    double b2 = b * b;
    for (int i = 0; i < Mmax; i++)
    {
        sum = sum + coef[i] * expf(-(powf(b2, (i + 0.0)) * sigma * sigma) * k2 / 2.0);
    }
    return sum;
}

float RBSOG::Gaussian_Fourier_Plus_modify(float Kx, float Ky, float Kz, float sigma, float b, float w0, int Mmax, float* coef_npt)
{
    float k2 = Kx * Kx + Ky * Ky + Kz * Kz;
    float sum = 0.00;
    double b2 = b * b;
    for (int i = 0; i < Mmax; i++)
    {
        sum = sum + coef_npt[i] * expf(-(powf(b2, float((i + 0.0))) * sigma * sigma) * k2 / 2.0);
    }
    return sum;
}

__m512 RBSOG::Gaussian_Fourier_Plus_AVX(__m512 Kx, __m512 Ky, __m512 Kz, float sigma, float b, float w0, int Mmax, float* coef)
{
    __m512 k2 = Kx * Kx + Ky * Ky + Kz * Kz;
    __m512 sum = _mm512_setzero_ps();
    __m512 sigma2 = _mm512_set1_ps( -sigma * sigma * 0.5);
    __m512 b2 = _mm512_set1_ps(b * b);
    for (int i = 0; i < Mmax; i++)
    {
        __m512 index = _mm512_set1_ps(i);
        __m512 coef_i = _mm512_set1_ps(coef[i]);
        __m512 b_2i = _mm512_pow_ps(b2, index);
        __m512 mid = _mm512_mul_ps(b_2i, sigma2);
        mid = _mm512_mul_ps(mid, k2);
        __m512 exp = _mm512_exp_ps(mid);
        sum = sum + coef_i * exp;
    }
    return sum;
}