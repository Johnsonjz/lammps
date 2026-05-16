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
   Contributing author: Jiuyang Liang (SJTU), Qi Zhou (SJTU), Zhen Jiang (SJTU)
------------------------------------------------------------------------- */

#include "pair_lj_cut_coul_user.h"
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

inline double PairLJCutCoulUser::G_Sigma1(double Sigma, double r)//     斯
{
	return exp(-r * r / (2 * Sigma * Sigma)) / sqrt(2 * MY_PI * Sigma * Sigma);
}

double PairLJCutCoulUser::Compute_W01(double r0, double b)//    W0
{
	double sum = 0.00;
	for (int i = 1; i < 200; i++)
	{
		sum = sum + pow(b, double(-i + 0.00)) * G_Sigma1(1.0, pow(b, double(-i + 0.00)) * r0);
	}
	double W0 = (1.0 / G_Sigma1(1.0, r0)) * ((1.0 / (2 * log(b) * r0)) - sum);
	return W0;
}
 


/* ---------------------------------------------------------------------- */

PairLJCutCoulUser::PairLJCutCoulUser(LAMMPS* lmp) : Pair(lmp)
{
	ewaldflag = pppmflag = 1;
    rbsogflag = 1;
	respa_enable = 1;
	writedata = 1;
	ftable = NULL;
	qdist = 0.0;
	Step = 0;
	TimeSet = new double[1000];
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulUser::~PairLJCutCoulUser()
{
    if (copymode) return;

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);

        memory->destroy(cut_lj);
        memory->destroy(cut_ljsq);
        memory->destroy(epsilon);
        memory->destroy(sigma);
        memory->destroy(lj1);
        memory->destroy(lj2);
        memory->destroy(lj3);
        memory->destroy(lj4);
        memory->destroy(offset);

        memory->destroy(BL);
        memory->destroy(BL3INV);
        memory->destroy(BL2SIGMA2INV);
        memory->destroy(TaylorCoeff);
    }
    if (ftable) free_tables();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulUser::settings(int narg, char** arg)
{

	cut_lj_global = utils::numeric(FLERR, arg[0], false, lmp);
	if (narg == 1) cut_coul = cut_lj_global;
	else cut_coul = utils::numeric(FLERR, arg[1], false, lmp);

	b = utils::numeric(FLERR, arg[2], false, lmp);
	Sigma = utils::numeric(FLERR, arg[3], false, lmp);
	Mmax= utils::numeric(FLERR, arg[4], false, lmp);

	// reset cutoffs that have been explicitly set

	if (allocated) {
		int i, j;
		for (i = 1; i <= atom->ntypes; i++)
			for (j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
	}

	r0 = cut_coul / Sigma;
	w0 = Compute_W01(r0, b);

    coef = log(b) / (Sigma * Sigma * sqrt(2 * MY_PI * Sigma * Sigma));

    TaylorTerms = 6;
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulUser::compute(int eflag, int vflag)
{
    double time;
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();

	int i, ii, j, jj, inum, jnum, itype, jtype, itable;
	double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
	double fraction, table;
	double r, rinv, r2inv, r3inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
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
                        rinv = sqrt(r2inv);
                        r3inv = rinv * r2inv;
                        prefactor = qqrd2e * qtmp * q[j];
                        forcecoul = prefactor * (r3inv + TaylorCoeff[0] + TaylorCoeff[1] * rsq + TaylorCoeff[2] * rsq * rsq + TaylorCoeff[3] * rsq * rsq * rsq +TaylorCoeff[4] * rsq * rsq * rsq*rsq+TaylorCoeff[5] * rsq * rsq * rsq*rsq*rsq);

                        // if (comm->me == 0) {
                        //     static int dbg_cnt = 0;
                        //     const int dbg_max = 5;
                        //     if (dbg_cnt < dbg_max) {
                        //         dbg_cnt++;
                        //         std::cout << "DEBUG_LJ_COUL: i=" << i << " j=" << j
                        //                   << " prefactor=" << prefactor
                        //                   << " qqrd2e=" << qqrd2e
                        //                   << " qtmp=" << qtmp << " qj=" << q[j]
                        //                   << " r3inv=" << r3inv
                        //                   << " rsq=" << rsq
                        //                   << " Taylor0=" << TaylorCoeff[0]
                        //                   << " Taylor1=" << TaylorCoeff[1]
                        //                   << " factor_coul=" << factor_coul
                        //                   << " forcecoul(before)=" << forcecoul
                        //                   << std::endl;
                        //     }
                        // }

                        if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor * r3inv;
                        
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
                            prefactor = qtmp * q[j] * table;
                            forcecoul -= (1.0 - factor_coul) * prefactor*r2inv;
                        }
                    }

                }
                else forcecoul = 0.0;

                if (rsq < cut_ljsq[itype][jtype]) {
                    r6inv = r2inv * r2inv * r2inv;
                    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
                }
                else forcelj = 0.0;

                fpair = forcecoul + factor_lj * forcelj * r2inv;

                f[i][0] += delx * fpair;
                f[i][1] += dely * fpair;
                f[i][2] += delz * fpair;
                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                }
                // if (comm->me == 0)
                // {
                //     cout << " Force = " << f[i][0] << " " << f[i][1] << " " << f[i][2] << " "  << endl;
                // }
                            
                if (eflag) {                 
                    if (rsq < cut_coulsq) {
                        if (!ncoultablebits || rsq <= tabinnersq)
                        {
                            double sume = 0.00;
                            sume = sume + (2 * log(b) / sqrt(2 * MY_PI * Sigma * Sigma)) * w0 * exp(-rsq / (2 * Sigma * Sigma));
                            for (int jjj = 1; jjj < Mmax; jjj++)
                            {
                                sume = sume + (2 * log(b) / sqrt(2 * MY_PI * Sigma * Sigma)) * (1.0 / pow(b, jjj + 0.00)) * exp(-rsq / (2 * Sigma * Sigma * pow(b, jjj + 0.00) * pow(b, jjj + 0.00)));
                            }
                            prefactor = qqrd2e * qtmp * q[j];
                            ecoul = prefactor * (rinv - sume);                           
                            prefactor = prefactor * rinv;
                        }
                        else {
                            table = etable[itable] + fraction * detable[itable];
                            ecoul = qtmp * q[j] * table;
                        }
                        if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
                    }
                    else ecoul = 0.0;
                    

                    if (rsq < cut_ljsq[itype][jtype]) {
                        evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) -
                            offset[itype][jtype];
                        evdwl *= factor_lj;
                    }
                    else evdwl = 0.0;
                }
                
                //Add short-range virial component of u-series

                if (evflag) ev_tally(i, j, nlocal, newton_pair,
                    evdwl, ecoul, fpair, delx, dely, delz);
            }
        }
    }

    time = MPI_Wtime() - time;

    if ((update->ntimestep >= 0) && update->ntimestep < 1000) {
        TimeSet[update->ntimestep - 0] = time * 1000;
    }

    if (comm->me == 0 && update->ntimestep == 1000) {
        ofstream outfile;
        outfile.open("./Time_Pair_RBSOG.txt");
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 1000; i++)
            outfile << TimeSet[i] << endl;
        outfile.close();
    }

    for (int i = 0; i < nlocal; i++)
    {
        F[i][0] = f[i][0]- F[i][0];
        F[i][1] = f[i][1]- F[i][1];
        F[i][2] = f[i][2]- F[i][2];
    }

    if (vflag_fdotr) virial_fdotr_compute();
    // if(comm->me == 0 && update->ntimestep == 0)
    // {
    //     cout << "The virial = " << virial[0] << "  " << virial[1] << "  " << virial[2] << "  " << virial[3] << "  " << virial[4] << "  " << virial[5] << endl;
    //     cout << "The force = " << endl;
    //     for (int i = 0; i < 3; i++)
    //     {
    //         cout << f[i][0] << "  " << f[i][1] << "  " << f[i][2] << endl;
    //     }
    // }

}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulUser::allocate()
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

    int m = Mmax;
    memory->create(BL, m, "pair:BL");
    memory->create(BL3INV, m, "pair:BL3INV");
    memory->create(BL2SIGMA2INV, m, "pair:BL2SIGMA2INV");
    int nterms = TaylorTerms = 6;
    memory->create(TaylorCoeff, nterms, "pair:TaylorCoeff");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulUser::init_style()
{
    if (!atom->q_flag)
        error->all(FLERR, "Pair style lj/cut/coul/user requires atom attribute q");

    // request regular or rRESPA neighbor list

    int list_style = NeighConst::REQ_DEFAULT;

   if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
    if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
  }
  neighbor->add_request(this, list_style);

    cut_coulsq = cut_coul * cut_coul;

    // set rRESPA cutoffs

    if (utils::strmatch(update->integrate_style,"^respa") &&
      (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
    cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
  else cut_respa = nullptr;

    // insure use of KSpace long-range solver, set g_ewald

    if (force->kspace == NULL)
        error->all(FLERR, "Pair style requires a KSpace style");
    g_ewald = force->kspace->g_ewald;

    // setup force tables

    if (ncoultablebits) init_tables(cut_coul, cut_respa);

    r0 = cut_coul / Sigma;
	w0 = Compute_W01(r0, b);

    for (int i = 0; i < Mmax; i++)
    {
        BL[i] = pow(b, i);
        double xxx = 1.0 / BL[i];
        BL3INV[i] = xxx * xxx * xxx;
        BL2SIGMA2INV[i] = 1.0 / (2.0 * Sigma * Sigma * BL[i] * BL[i]);
    }
    BL3INV[0] = w0;

    coef = log(b) / (Sigma * Sigma * sqrt(2 * MY_PI * Sigma * Sigma));

    for (int i = 0; i < TaylorTerms; i++)
    {
        double sumsum = 0.00;
        for (int j = 0; j < Mmax; j++)
        {
            sumsum = sumsum + BL3INV[j] * (1.0 / MathSpecial::factorial(i+0.00)) * pow(BL2SIGMA2INV[j], i+0.00);
        }
        TaylorCoeff[i] = pow(-1.0,i+1.0) * 2.0 * coef * sumsum;
    }
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutCoulUser::init_one(int i, int j)
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

void* PairLJCutCoulUser::extract(const char* str, int& dim)
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

void PairLJCutCoulUser::coeff(int narg, char** arg)
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

