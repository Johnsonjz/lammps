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
   Contributing author: Jiuyang Liang (SJTU)
------------------------------------------------------------------------- */

#include "pair_lj_cut_coul_rbl.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <string>
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
#include "memory.h"
#include "error.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h>
//#include"sse_mathfun.h"

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917 //2/sqrt(pi)
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairLJCutCoulRBL::PairLJCutCoulRBL(LAMMPS* lmp) : Pair(lmp)
{
	ewaldflag = pppmflag = 1;
	respa_enable = 1;
	writedata = 1;
	ftable = NULL;
	qdist = 0.0;
	TimeSet = new double[2000];
    Time_Sampling = new double[2000];
    Time_Compute = new double[2000]; 
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulRBL::~PairLJCutCoulRBL()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);

        memory->destroy(cut_lj);
        memory->destroy(cut_ljsq);
        memory->destroy(epsilon);
        memory->destroy(sigma);
    }
    if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulRBL::compute(int eflag, int vflag)
{
    double time2;
    MPI_Barrier(MPI_COMM_WORLD);
    time2 = MPI_Wtime();

    //if (comm->me == 0)
    //    cout << time2 << endl;

    int i, ii, j, jj, inum, jnum, itype, jtype, itable;
    double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
    double fraction, table;
    double r, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
    double grij, expm2, prefactor, t, erfc;

    int* ilist, * jlist, * numneigh, ** firstneigh, * num_core_neigh, ** first_core_neigh, * num_shell_neigh, ** first_shell_neigh;/**/

    double rsq;

    evdwl = ecoul = 0.0;
    ev_init(eflag, vflag);

    double** x = atom->x;
    double** f = atom->f;
    double* q = atom->q;
    int* type = atom->type;
    int nlocal = atom->nlocal;
    double* special_coul = force->special_coul; // 1-2, 1-3, 1-4 prefactors for Coulombics
    double* special_lj = force->special_lj; // 1-2, 1-3, 1-4 prefactors for LJ
    int newton_pair = force->newton_pair; // Newton's 3rd law settings
    double qqrd2e = force->qqrd2e; // q^2/r to energy w/ dielectric constant

    inum = list->inum;
    ilist = list->ilist;

    numneigh = list->numneigh; // # of J neighbors for each I atom
    firstneigh = list->firstneigh; // ptr to 1st J int value of each I atom

    num_core_neigh = list->num_core_neigh;
    first_core_neigh = list->first_core_neigh;
    num_shell_neigh = list->num_shell_neigh;
    first_shell_neigh = list->first_shell_neigh;

    //cout << "num_core_neigh == " << num_core_neigh[0] << "   " << "num_shell_neigh == " << num_shell_neigh[0] << endl;

    double Total_Force[3] = { 0.00,0.00,0.00 };

    int total_no_comp = 0;

    /**************       寻找最大的邻居数量       *************/
    int Bin_Shell_Choice[int(P)];

    srand((unsigned int)time(0));
    time2 = MPI_Wtime();

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        qtmp = q[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];

        jlist = first_core_neigh[i];
        jnum = num_core_neigh[i];

        //cout << "ii = "<<ii<<"   "<<"jlist[0]==" << jlist[0] << "   " << "jnum = " << jnum << endl;

        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            factor_coul = special_coul[sbmask(j)];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            jtype = type[j];

            r2inv = 1.0 / rsq;

            if (!ncoultablebits || rsq <= tabinnersq) {
                r = sqrt(rsq);
                grij = g_ewald * r;
                expm2 = exp(-grij * grij);
                t = 1.0 / (1.0 + EWALD_P * grij);
                erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                prefactor = qqrd2e * qtmp * q[j] / r;
                forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
                if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
            }
            else {
                union_int_float_t rsq_lookup; // typedef union {int i; float f;} union_int_float_t; 共用体 改变float int就会变成相应二进制对应的整数
                rsq_lookup.f = rsq;
                itable = rsq_lookup.i & ncoulmask;
                itable >>= ncoulshiftbits;
                fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
                table = ftable[itable] + fraction * dftable[itable];
                forcecoul = qtmp * q[j] * table;
                if (factor_coul < 1.0) {
                    table = ctable[itable] + fraction * dctable[itable];
                    prefactor = qtmp * q[j] * table;
                    forcecoul -= (1.0 - factor_coul) * prefactor;
                }
            }

            if (rsq < cut_ljsq[itype][jtype]) {
                r6inv = r2inv * r2inv * r2inv;
                forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
            }
            else forcelj = 0.0;

            fpair = (forcecoul + factor_lj * forcelj) * r2inv;

            f[i][0] += delx * fpair;
            f[i][1] += dely * fpair;
            f[i][2] += delz * fpair;

        }
        
        //cout << "ii = " << ii << "   " << "jlist[0]==" << jlist[0] << "   " << "jnum = " << jnum << endl;

        jlist = first_shell_neigh[i];
        jnum = num_shell_neigh[i];

        //cout << "jnum == " << jnum << endl;

        int MAX_P;
        if (jnum <= P) {
            MAX_P = jnum;
        }
        else
        {
            MAX_P = P;
        }

        /*随机抽取带放回 可以重复
        for (int iii = 0; iii < MAX_P; iii++)
        {
            int ind = rand() % jnum;
            Bin_Shell_Choice[iii] = jlist[ind];
            //if (i == 0)
             //   cout << "The bin shell choice == " << ind << endl;
        }
        */
        
        /*        随机抽取不放回  不可以重复        */
        int old = MAX_P;
        int ind = 0;
        for (int iii = 0; iii < jnum; iii++) {
            if (rand() % (jnum - iii) < old) {
                Bin_Shell_Choice[ind] = jlist[iii];
                ind++;
                old--;
            }
        }
        

        //此处添加Shell的计算 
        double Para[3] = {0.00,0.00,0.00};
        for (int iii = 0; iii < MAX_P; iii++)
        {
            j = Bin_Shell_Choice[iii];

            //cout << "iii == " << iii << "   j=" << j << endl;

            factor_lj = special_lj[sbmask(j)];
            factor_coul = special_coul[sbmask(j)];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            jtype = type[j];

            r2inv = 1.0 / rsq;
            
            if (!ncoultablebits || rsq <= tabinnersq) {
                r = sqrt(rsq);
                grij = g_ewald * r;
                expm2 = exp(-grij * grij);
                t = 1.0 / (1.0 + EWALD_P * grij);
                erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                prefactor = qqrd2e * qtmp * q[j] / r;
                forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
                if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
            }
            else {
                union_int_float_t rsq_lookup; // typedef union {int i; float f;} union_int_float_t; 共用体 改变float int就会变成相应二进制对应的整数
                rsq_lookup.f = rsq;
                itable = rsq_lookup.i & ncoulmask;
                itable >>= ncoulshiftbits;
                fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
                table = ftable[itable] + fraction * dftable[itable];
                forcecoul = qtmp * q[j] * table;
                if (factor_coul < 1.0) {
                    table = ctable[itable] + fraction * dctable[itable];
                    prefactor = qtmp * q[j] * table;
                    forcecoul -= (1.0 - factor_coul) * prefactor;
                }
            }

            if (rsq < cut_ljsq[itype][jtype]) {
                r6inv = r2inv * r2inv * r2inv;
                forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
            }
            else forcelj = 0.0;

            fpair = (forcecoul + factor_lj * forcelj) * r2inv;

            Para[0] += delx * fpair;
            Para[1] += dely * fpair;
            Para[2] += delz * fpair;

            if (comm->me == 0 && i == 0)
            {
               // cout << i<<"   "<<j<<"   "<<fpair << "    " << forcecoul << "    " << forcelj << "     " << delx << "   " << dely << "   " << delz << endl;
               // cout << "The forces along each dimension are 2 " << Para[0] << "    " << Para[1] << "    " << Para[2] << "    " << endl;
            }
        }

        double S;
        if (MAX_P == 0)
        {
            S = 0.00;
        }
        else
        {
            S = (jnum + 0.00) / (MAX_P + 0.00);
        }
        f[i][0] += Para[0] * S;
        f[i][1] += Para[1] * S;
        f[i][2] += Para[2] * S;

        Total_Force[0] += Para[0] * S;
        Total_Force[1] += Para[1] * S;
        Total_Force[2] += Para[2] * S;

        if (comm->me == 0 && i == 0)
        {
            //cout << "The forces along each dimension are " << Para[0] << "    " << Para[1] << "    " << Para[2] << "    "<<S<<endl;
            //cout << "The forces along each dimension are 1 " << Total_Force[0] << "    " << Total_Force[1] << "    " << Total_Force[2] << "    " << S <<"   "<< Shell_List[ii] <<"   "<< endl;
        }
    }

    //if (comm->me == 0)
    //    cout << time2 << endl;
    time2 = MPI_Wtime() - time2;

    if (comm->me == 0&& (update->ntimestep%100==0))
        cout << time2 << endl;

    if ((update->ntimestep >= 0) && update->ntimestep < 2000) {
        Time_Compute[update->ntimestep - 0] = time2 * 1000;
        //cout << Time_Compute[update->ntimestep - 2000] << "    " << update->ntimestep - 2000<<endl;
    }


    //if (comm->me == 0)
        //cout << "The forces along each dimension are "<<f[0][0] << "    " << f[0][1] << "    " << f[0][2] << endl;

    /*
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        qtmp = q[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        double Choice[jnum][6];
        int index = 0;

        for (jj = 0; jj < jnum; jj++)
        {
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
                    // 如果在core list里 
                    //if (rsq <= core_size * core_size || j < nlocal)
                    if (rsq < core_size * core_size)
                    {
                        if (!ncoultablebits || rsq <= tabinnersq) {
                            r = sqrt(rsq);
                            grij = g_ewald * r;
                            expm2 = exp(-grij * grij);
                            t = 1.0 / (1.0 + EWALD_P * grij);
                            erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                            prefactor = qqrd2e * qtmp * q[j] / r;
                            forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
                            if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
                        }
                        else {
                            union_int_float_t rsq_lookup; // typedef union {int i; float f;} union_int_float_t; 共用体 改变float int就会变成相应二进制对应的整数
                            rsq_lookup.f = rsq;
                            itable = rsq_lookup.i & ncoulmask;
                            itable >>= ncoulshiftbits;
                            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
                            table = ftable[itable] + fraction * dftable[itable];
                            forcecoul = qtmp * q[j] * table;
                            if (factor_coul < 1.0) {
                                table = ctable[itable] + fraction * dctable[itable];
                                prefactor = qtmp * q[j] * table;
                                forcecoul -= (1.0 - factor_coul) * prefactor;
                            }
                        }
                    }
                    else
                    {
                        forcecoul = 0.0;

                        // 如果在shell list里 
                        Choice[index][0] = delx;
                        Choice[index][1] = dely;
                        Choice[index][2] = delz;
                        Choice[index][3] = rsq;
                        Choice[index][4] = qtmp * q[j];
                        Choice[index][5] = j;
                        index++;
                    }
                }
                else { forcecoul = 0.0; }

                if (rsq < cut_ljsq[itype][jtype]) {
                    r6inv = r2inv * r2inv * r2inv;
                    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
                }
                else forcelj = 0.0;

                fpair = (forcecoul + factor_lj * forcelj) * r2inv;

                f[i][0] += delx * fpair;
                f[i][1] += dely * fpair;
                f[i][2] += delz * fpair;
                
                //if (newton_pair || j < nlocal) {
                  //  f[j][0] -= delx * fpair;
                    //f[j][1] -= dely * fpair;
                    //f[j][2] -= delz * fpair;

                    //total_no_comp++;
                //}
            }
        }
               
        //在index个邻居里面抽P个
        int MAX_P;
        if (index <= P) { 
            MAX_P = index;
        }
        else
        {
            MAX_P = P;
        }
        int Choice_Number[MAX_P];
        int old = MAX_P;
        int ind = 0;

        srand((unsigned int)time(0));

        for (int iii = 0; iii < index; iii++) {
            if (rand() % (index - iii) < old) {
                Choice_Number[ind] = iii;
                ind++;
                old--;
            }
        }

        
        //if (i == nlocal - 1)
        //{
            //cout<< "index == " << index << "   MAX_P == " << MAX_P <<"   core_size * core_size == "<< core_size * core_size<< endl;
            //for (int iii = 0; iii < ind; iii++)
            //{
                //cout << endl;
                //cout << "   " << Choice_Number[iii] << "     " << index << "     " << ind <<"   "<< MAX_P << endl;
                ////cout << cutsq[type[i]][type[Choice[Choice_Number[iii]][5]]]<<endl;
                //cout << endl;
            //}
          //  
        //}
        

        //for (int i = 0; i < ind; i++)
        //{
            //cout << endl;
            //cout << "   " << Choice_Number[i] << "     " << index << "     " << ind << endl;
            //cout << endl;
        //}

        //if (comm->me == 0 && ii == 0)cout << "index == " << index <<"   MAX_P == "<<MAX_P<< endl; 

        double Para[3] = { 0.00,0.00,0.00 };
        double qtmp_qj;

        for (int iii = 0; iii < MAX_P; iii++)
        {
            int index_index = Choice_Number[iii];
            delx = Choice[index_index][0];
            dely = Choice[index_index][1];
            delz = Choice[index_index][2];
            rsq = Choice[index_index][3];
            qtmp_qj = Choice[index_index][4];
            j = Choice[index_index][5];
            r2inv = 1.0 / rsq;

            if (factor_coul != 1)
            {
                cout << "Error! factor_coul != 1" << endl;
            }

            if (!ncoultablebits || rsq <= tabinnersq) {
                r = sqrt(rsq);
                grij = g_ewald * r;
                expm2 = exp(-grij * grij);
                t = 1.0 / (1.0 + EWALD_P * grij);
                erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                prefactor = qqrd2e * qtmp_qj / r;
                forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
                if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
            }
            else {
                union_int_float_t rsq_lookup; // typedef union {int i; float f;} union_int_float_t; 共用体 改变float int就会变成相应二进制对应的整数
                rsq_lookup.f = rsq;
                itable = rsq_lookup.i & ncoulmask;
                itable >>= ncoulshiftbits;
                fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
                table = ftable[itable] + fraction * dftable[itable];
                forcecoul = qtmp_qj * table;
                if (factor_coul < 1.0) {
                    table = ctable[itable] + fraction * dctable[itable];
                    prefactor = qtmp_qj * table;
                    forcecoul -= (1.0 - factor_coul) * prefactor; 
                }
            }

            fpair = forcecoul * r2inv;

            Para[0] += delx * fpair;
            Para[1] += dely * fpair;
            Para[2] += delz * fpair;

            
            //if (j < nlocal)
            //{
                //f[j][0] -= delx * fpair;
                //f[j][1] -= dely * fpair;
                //f[j][2] -= delz * fpair;

                //Total_Force[0] -= delx * fpair;
                //Total_Force[1] -= dely * fpair;
                //Total_Force[2] -= delz * fpair;
            //}
            
        }
        double S;
        if (MAX_P == 0)
        {
            S = 0.00;
        }
        else
        {
            S = (index + 0.00) / (MAX_P + 0.00);
        }
        f[i][0] += Para[0] * S;
        f[i][1] += Para[1] * S;
        f[i][2] += Para[2] * S;

        Total_Force[0] += Para[0] * S;
        Total_Force[1] += Para[1] * S;
        Total_Force[2] += Para[2] * S;

        //if ((comm->me == 0)&&(i%1000==0))cout << "total_no_comp == " << total_no_comp << "  S =" << (index + 0.00) / (MAX_P + 0.00) << endl;
    }
    */


    double Total_All[3] = { 0.00 ,0.00,0.00 };
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Allreduce((double*)Total_Force, (double*)Total_All, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    double reduce[3] = { Total_All[0] / (atom->natoms + 0.00),Total_All[1] / (atom->natoms + 0.00),Total_All[2] / (atom->natoms + 0.00) };
    for (int iii = 0; iii < nlocal; iii++)
    {
        f[iii][0] = f[iii][0] - reduce[0];
        f[iii][1] = f[iii][1] - reduce[1]; 
        f[iii][2] = f[iii][2] - reduce[2];
    }

    //cout << f[0][0] << "    " << f[0][1] << "    " << f[0][2] << endl;

    if (comm->me == 0 && update->ntimestep == 2000) {
        ofstream outfile;
        string str1,str2,str3,str4;
        stringstream ss;
        ss << cut_lj_global;
        str1 = ss.str();
        ss.clear();
        ss.str("");
        ss << cut_coul;
        str2 = ss.str();
        ss.clear();
        ss.str("");
        ss << core_size;
        str3 = ss.str();
        ss.clear();
        ss.str("");
        ss << P;
        str4 = ss.str();
        ss.clear();
        ss.str("");
        string locate = "./Time_RBL_Sampling";
        string txt1 = locate + "_" + str1 + "_" + str2 + "_" + str3 + "_" + str4 + ".txt";
        string txt2 = "./Time_RBL_Compute_" + str1 + "_" + str2 + "_" + str3 + " " + str4 + ".txt";
        outfile.open(txt1.c_str());
        //outfile.open("/lustre/home/acct-matxzl/matxzl/jiuyang/InvAug/Test/Time.txt");
        for (int i = 0; i < 2000; i++)
            outfile << Time_Sampling[i] << endl;
        outfile.close();
        ofstream outfile1;
        outfile1.open(txt2.c_str());
        for (int i = 0; i < 2000; i++)
            outfile1 << Time_Compute[i] << endl;
        outfile1.close();
    }
   
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulRBL::allocate()
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
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulRBL::settings(int narg, char** arg)
{
    if (narg < 3 || narg > 4) error->all(FLERR, "Illegal pair_style command");

    cut_lj_global = utils::numeric(FLERR,arg[0],false,lmp);
  if (narg == 1) cut_coul = cut_lj_global;
  else cut_coul = utils::numeric(FLERR,arg[1],false,lmp);

    // reset cutoffs that have been explicitly set

    if (allocated) {
        int i, j;
        for (i = 1; i <= atom->ntypes; i++)
            for (j = i; j <= atom->ntypes; j++)
                if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
    }

    core_size= utils::numeric(FLERR,arg[2],false,lmp);
    P= utils::numeric(FLERR,arg[3],false,lmp);

    force->pair->cut_core = core_size;

    if (force->newton_pair == 1)
        error->all(FLERR, "Pair style pair/lj/cut/coul/rbl requires newton pair off");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutCoulRBL::coeff(int narg, char** arg)
{
    if (narg < 4 || narg > 5)
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);

    double cut_lj_one = cut_lj_global;
  if (narg == 5) cut_lj_one = utils::numeric(FLERR,arg[4],false,lmp);

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

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulRBL::init_style()
{
    if (!atom->q_flag)
        error->all(FLERR, "Pair style lj/cut/coul/long requires atom attribute q");

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

double PairLJCutCoulRBL::init_one(int i, int j)
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

void* PairLJCutCoulRBL::extract(const char* str, int& dim)
{
    dim = 0;
    if (strcmp(str, "cut_coul") == 0) return (void*)&cut_coul;
    dim = 2;
    if (strcmp(str, "epsilon") == 0) return (void*)epsilon;
    if (strcmp(str, "sigma") == 0) return (void*)sigma;
    return NULL;
}
#endif  // LMP_ENABLE_EXPERIMENTAL_DRBSOG
