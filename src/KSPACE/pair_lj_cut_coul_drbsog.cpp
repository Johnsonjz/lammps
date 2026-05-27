#ifdef LMP_ENABLE_EXPERIMENTAL_DRBSOG
/* ----------------------------------------------------------------------
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
  http://lammps.sandia.gov, Sandia National Laboratories
  Steve Plimpton, sjplimp@sandia.gov

  Copyright (2003) Sandia Corporation.  Under the terms of Contract
  DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
  certain rights in this software.  This software is distributed under
  the GNU General Public License.

  See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jiuyang Liang (SJTU), Qi Zhou (SJTU)
------------------------------------------------------------------------- */
#include "pair_lj_cut_coul_drbsog.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "respa.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "utils.h"
#include "error.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <immintrin.h> 

#include <cmath>

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

#define MY_IPIS 1.0/MY_PIS
#define MY_PI3S MY_PI*MY_PIS

inline double f_m(double r, double t, double lambda)
{
    double a = exp(t);
    double Gaussian = MY_IPIS * exp(-r * r * a - 1.0 / (4 * lambda * lambda * a) + t / 2.0);
    return Gaussian;
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulDRBSOG::PairLJCutCoulDRBSOG(LAMMPS* lmp) : Pair(lmp)
{
	ewaldflag = pppmflag = 1;
	drbsogflag = 1;//注意这里，要在Pair里写一个
	respa_enable = 1;
	writedata = 1;
	ftable = NULL;
	qdist = 0.0;
	Step = 0;
	TimeSet = new double[1000];
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulDRBSOG::~PairLJCutCoulDRBSOG()
{

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulDRBSOG::settings(int narg, char** arg)
{
	cut_lj_global = utils::numeric(FLERR, arg[0], false, lmp);
	if (narg == 1) cut_coul = cut_lj_global;
	else cut_coul = utils::numeric(FLERR, arg[1], false, lmp);

	h = fabs(utils::numeric(FLERR, arg[2], false, lmp));
	t_0 = utils::numeric(FLERR, arg[3], false, lmp);
	M1 = fabs(utils::numeric(FLERR, arg[4], false, lmp));
	M2 = fabs(utils::numeric(FLERR, arg[5], false, lmp));
	w_M2 = utils::numeric(FLERR, arg[6], false, lmp);
	lambda = utils::numeric(FLERR, arg[7], false, lmp);
	TaylorTerms = utils::numeric(FLERR, arg[8], false, lmp);//泰勒的截断项数

	// reset cutoffs that have been explicitly set

    if(comm->me==0)
    cout << "The coefficients are " << "h == " << h << "   t_0 == " << t_0 << "    M1 == " << M1 << "   M2 == " << M2 << "    w_M2 == " << w_M2 << "   lambda == " << lambda << "    TaylorTerms == " << TaylorTerms << endl;

	TaylorCoeff = new double[TaylorTerms];

	if (allocated) {
		int i, j;
		for (i = 1; i <= atom->ntypes; i++)
			for (j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
		}

	//cout << "r0==" << r0 << "   " << "w0==" << w0 << endl;

	//****************************如果用泰勒展开法****************************/
	//TaylorTerms = 4;	
	for (int j = 0; j < TaylorTerms; j++)
	{
		double sumsum = 0.00;
		for (int m = -M1; m <= M2-1; m++)
		{
			sumsum = sumsum + exp((j + 1) * (t_0 + m * h)) * f_m(0, t_0 + m * h, lambda);
		}
		sumsum = sumsum + w_M2 * exp((j + 1) * (t_0 + M2 * h)) * f_m(0, t_0 + M2 * h, lambda);
		TaylorCoeff[j] = pow(-1.0, j + 1) * 2 * h / MathSpecial::factorial(j + 0.00) * sumsum;
	}
	//*****************************************************************//
	//if(comm->me==0)
	//cout << "The Taylor Coeffs are " << TaylorCoeff[0] << "    " << TaylorCoeff[1] << "     " << TaylorCoeff[2] << endl;
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulDRBSOG::compute(int eflag, int vflag)
{
    double time;
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();

    int i, ii, j, jj, inum, jnum, itype, jtype, itable;
    double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
    double fraction, table;
    double r, rinv, r2inv, r3inv, factor, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
    double grij, expm2, prefactor, t, erfc;
    int* ilist, * jlist, * numneigh, ** firstneigh;
    double rsq;

    evdwl = ecoul = 0.0;
    ev_init(eflag, vflag);

    double** x = atom->x;
    double** f = atom->f;
    double* q = atom->q;
    int* type = atom->type;
    int nlocal = atom->nlocal;
    double* special_coul = force->special_coul;
    double* special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    double qqrd2e = force->qqrd2e;

    //cout << force->newton_pair << "   " << force->qqrd2e << "   "<< force->special_coul[0]<<endl;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    double F[nlocal][3];
    for (int i = 0; i < nlocal; i++)
    {
        F[i][0] = f[i][0];
        F[i][1] = f[i][1];
        F[i][2] = f[i][2];
    }

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        qtmp = q[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            factor_coul = special_coul[sbmask(j)];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            jtype = type[j];

            if (rsq < cutsq[itype][jtype]) {
                r2inv = 1.0 / rsq;
                if (rsq < cut_coulsq) {
                    
                    if (!ncoultablebits || rsq <= tabinnersq) {
                        r = sqrt(rsq);
                        rinv = 1.0/r;
                        r3inv = rinv * r2inv;
                        prefactor = qqrd2e * qtmp * q[j];
                        
                        //直接计算部分
                        double sumF; 
                        double exp_r = exp(-r / lambda);
                        sumF = w_M2 * exp(t_0 + M2 * h) * f_m(r, t_0 + M2 * h, lambda);
                        for (int m = -M1; m <= M2 - 1; m++)
                        {
                            sumF = sumF + exp(t_0 + m * h) * f_m(r, t_0 + m * h, lambda);
                        }
                        sumF *= (-2.0) * h;
                        
                        double r_F;
                        r_F = exp_r * (1 + r / lambda) / (r * r * r);
                        forcecoul = prefactor * (r_F + sumF);//一定得带Q啊
                        
                        

                        /*  泰勒展开
                        factor = 0.00;
                        double rsq_m = 1;
                        for (int m = 0; m < TaylorTerms; m++)
                        {
                            factor = factor + TaylorCoeff[m] * rsq_m;
                            rsq_m *= rsq;
                        }
                        forcecoul = prefactor * (exp(-r/lambda)*r3inv*(1+r/lambda) + factor);
                        */



                        //if (comm->me == 0 && ii < 10)
                        //    cout << "The approximate is " << forcecoul2 << "   The exact is " << forcecoul << endl;
                        

                        if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor * (exp(-r / lambda) * r3inv * (1 + r / lambda));

                    }
                    else
                    {
                        union_int_float_t rsq_lookup;
                        rsq_lookup.f = rsq;
                        itable = rsq_lookup.i & ncoulmask;
                        itable >>= ncoulshiftbits;
                        fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
                        table = ftable[itable] + fraction * dftable[itable];
                        forcecoul = qtmp * q[j] * table;

                        if (factor_coul < 1.0) {
                            table = ctable[itable] + fraction * dctable[itable];
                            prefactor = qtmp * q[j] * table;//table=exp(-r/lambda)/r
                            forcecoul -= (1.0 - factor_coul) * prefactor * r2inv * (1 + r / lambda);
                        }
                        
                        //r = sqrt(rsq);

                        //double exp_r = exp(-r / lambda);
                        //double r_F;
                        //r_F = exp_r * (1 + r / lambda) / (r * r * r);
                        //forcecoul = qqrd2e * (r_F + sumF);
                        //forcecoul = qqrd2e * qtmp * q[j] * (r_F);
                    }


                }
                else forcecoul = 0.0;

                //if (rsq < cut_ljsq[itype][jtype]) {
                //    r6inv = r2inv * r2inv * r2inv;
               //     forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
               //     }
               // else forcelj = 0.0;

                fpair = forcecoul;// + factor_lj * forcelj * r2inv;

                f[i][0] += delx * fpair;
                f[i][1] += dely * fpair;
                f[i][2] += delz * fpair;
                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                    }

                if (eflag) {
                    if (rsq < cut_coulsq) {
                        if (!ncoultablebits || rsq <= tabinnersq)
                        {
                            //以下为精确的能量计算 在/星星/内的是需要的 也可以改成泰勒展开
                            r = sqrt(rsq);

                            double sume;
                            sume = h * w_M2 * f_m(r, t_0 + M2 * h, lambda);
                            for (int m = -M1; m <= M2 - 1; m++)
                            {
                                sume = sume + h * f_m(r, t_0 + m * h, lambda);
                            }
                            prefactor = qqrd2e * qtmp * q[j];//此处有变化
                            ecoul = prefactor * (exp(-r / lambda) / r - sume);
                            
                            //if (comm->me == 0 && r < 0.8665)
                             //   {
                            //    cout << exp(-r / lambda) / r << "   " << sume << endl;
                            //    }
                        }
                        else {
                            table = etable[itable] + fraction * detable[itable];
                            ecoul = qtmp * q[j] * table;
                            
                            /*
                            r = sqrt(rsq);
                            double sume;
                            sume = h * w_M2 * f_m(r, t_0 + M2 * h, lambda);
                            for (int m = -M1; m <= M2 - 1; m++)
                            {
                                sume = sume + h * f_m(r, t_0 + m * h, lambda);
                            }
                            prefactor = qqrd2e * qtmp * q[j];//此处有变化
                            double ecoul_exact = prefactor * (exp(-r / lambda) / r - sume);
                            if (comm->me == 0 && ii == 0)
                            {
                                cout << "The energy is " << ecoul << "    " << ecoul_exact << endl;
                            }
                            */
                        }

                        if (factor_coul < 1.0)
                            {
                                ecoul -= (1.0 - factor_coul) * prefactor * (exp(-r / lambda) / r);//去掉一部分能量
                                cout << "The coul is factor_coul" << endl;
                            }
                    }
                    else ecoul = 0.0;


                    //if (rsq < cut_ljsq[itype][jtype]) {
                    //    evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) -
                    //        offset[itype][jtype];
                    //    evdwl *= factor_lj;
                    //    }
                    //else evdwl = 0.0;
                    }


                if (evflag) ev_tally(i, j, nlocal, newton_pair,
                    evdwl, ecoul, fpair, delx, dely, delz);
                }
            }
        }

    time = MPI_Wtime() - time;  // 终止计时
    //if (comm->me == 0 && update->ntimestep % 1000 == 0)cout << "The total time of PairUseries is " << time * 1000 << "   ms" << endl;

    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
        }

    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_Pair_DRBSOG.txt");
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
        }

    for (int i = 0; i < nlocal; i++)
    {
        F[i][0] = f[i][0] - F[i][0];
        F[i][1] = f[i][1] - F[i][1];
        F[i][2] = f[i][2] - F[i][2];
    }

    if (comm->me == 0)
    {
           for (int i = 0; i < 100; i++)
           {
                //cout << i << "   " << f[i][0] << "    " << f[i][1] << "     " << f[i][2] << endl;

                //cout << i << "   " << F[i][0] << "    " << F[i][1] << "     " << F[i][2] <<"   "<<x[i][0]<<"   "<<x[i][1]<<"    "<<x[i][2]<< endl;
           }
     }


    if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulDRBSOG::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

    memory->create(cut_lj, n + 1, n + 1, "pair:cut_lj");
    memory->create(cut_ljsq, n + 1, n + 1, "pair:cut_ljsq");
    memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
    memory->create(sigma, n + 1, n + 1, "pair:sigma");
    memory->create(lj1, n + 1, n + 1, "pair:lj1");
    memory->create(lj2, n + 1, n + 1, "pair:lj2");
    memory->create(lj3, n + 1, n + 1, "pair:lj3");
    memory->create(lj4, n + 1, n + 1, "pair:lj4");
    memory->create(offset, n + 1, n + 1, "pair:offset");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulDRBSOG::init_style()
{
    if (!atom->q_flag)
        error->all(FLERR, "Pair style lj/cut/coul/user requires atom attribute q");

    // request regular or rRESPA neighbor list

    int irequest;
    int respa = 0;

    if (update->whichflag == 1 && strstr(update->integrate_style, "respa")) {
        if (((Respa*)update->integrate)->level_inner >= 0) respa = 1;
        if (((Respa*)update->integrate)->level_middle >= 0) respa = 2;
        }

    irequest = neighbor->request(this, instance_me);

    if (respa >= 1) {
        neighbor->requests[irequest]->respaouter = 1;
        neighbor->requests[irequest]->respainner = 1;
        }
    if (respa == 2) neighbor->requests[irequest]->respamiddle = 1;

    cut_coulsq = cut_coul * cut_coul;

    // set rRESPA cutoffs

    if (strstr(update->integrate_style, "respa") &&
        ((Respa*)update->integrate)->level_inner >= 0)
        cut_respa = ((Respa*)update->integrate)->cutoff;
    else cut_respa = NULL;

    // insure use of KSpace long-range solver, set g_ewald

    if (force->kspace == NULL)
        error->all(FLERR, "Pair style requires a KSpace style");
    g_ewald = force->kspace->g_ewald;

    // setup force tables

    if (ncoultablebits) init_tables(cut_coul, cut_respa);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutCoulDRBSOG::init_one(int i, int j)
{
    if (setflag[i][j] == 0) {
        epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j],
            sigma[i][i], sigma[j][j]);
        sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
        cut_lj[i][j] = mix_distance(cut_lj[i][i], cut_lj[j][j]);
        }

    // include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P

    double cut = MAX(cut_lj[i][j], cut_coul + 2.0 * qdist);
    cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

    lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
    lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
    lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
    lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

    if (offset_flag && (cut_lj[i][j] > 0.0)) {
        double ratio = sigma[i][j] / cut_lj[i][j];
        offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
        }
    else offset[i][j] = 0.0;

    cut_ljsq[j][i] = cut_ljsq[i][j];
    lj1[j][i] = lj1[i][j];
    lj2[j][i] = lj2[i][j];
    lj3[j][i] = lj3[i][j];
    lj4[j][i] = lj4[i][j];
    offset[j][i] = offset[i][j];

    // check interior rRESPA cutoff

    if (cut_respa && MIN(cut_lj[i][j], cut_coul) < cut_respa[3])
        error->all(FLERR, "Pair cutoff < Respa interior cutoff");

    // compute I,J contribution to long-range tail correction
    // count total # of atoms of type I and J via Allreduce

    if (tail_flag) {
        int* type = atom->type;
        int nlocal = atom->nlocal;

        double count[2], all[2];
        count[0] = count[1] = 0.0;
        for (int k = 0; k < nlocal; k++) {
            if (type[k] == i) count[0] += 1.0;
            if (type[k] == j) count[1] += 1.0;
            }
        MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

        double sig2 = sigma[i][j] * sigma[i][j];
        double sig6 = sig2 * sig2 * sig2;
        double rc3 = cut_lj[i][j] * cut_lj[i][j] * cut_lj[i][j];
        double rc6 = rc3 * rc3;
        double rc9 = rc3 * rc6;
        etail_ij = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] *
            sig6 * (sig6 - 3.0 * rc6) / (9.0 * rc9);
        ptail_ij = 16.0 * MY_PI * all[0] * all[1] * epsilon[i][j] *
            sig6 * (2.0 * sig6 - 3.0 * rc6) / (9.0 * rc9);
        }

    return cut;
}

/* ---------------------------------------------------------------------- */

void* PairLJCutCoulDRBSOG::extract(const char* str, int& dim)
{
    dim = 0;
    if (strcmp(str, "cut_coul") == 0) return (void*)&cut_coul;
    dim = 2;
    if (strcmp(str, "epsilon") == 0) return (void*)epsilon;
    if (strcmp(str, "sigma") == 0) return (void*)sigma;
    return NULL;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutCoulDRBSOG::coeff(int narg, char** arg)
{
    if (narg < 4 || narg > 5)
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
    double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);

    double cut_lj_one = cut_lj_global;
    if (narg == 5) cut_lj_one = utils::numeric(FLERR, arg[4], false, lmp);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
            epsilon[i][j] = epsilon_one;
            sigma[i][j] = sigma_one;
            cut_lj[i][j] = cut_lj_one;
            setflag[i][j] = 1;
            count++;
            }
        }

    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}
#endif  // LMP_ENABLE_EXPERIMENTAL_DRBSOG
