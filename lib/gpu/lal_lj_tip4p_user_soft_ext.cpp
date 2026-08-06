/***************************************************************************
                            lj_tip4p_user_soft_ext.cpp
                             -------------------
                              V. Nikolskiy (HSE)

  Functions for LAMMPS access to lj/tip4p/user/soft acceleration functions
  (soft-core variant: adds lam1 per-type-pair scaling)

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : thevsevak@gmail.com
 ***************************************************************************/

#include <cassert>
#include <cmath>
#include <iostream>

#include "lal_lj_tip4p_user_soft.h"

using namespace std;
using namespace LAMMPS_AL;

static LJ_TIP4PUserSoft<PRECISION, ACC_PRECISION> LJTIP4PUSF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int ljtip4p_user_soft_gpu_init(const int ntypes, double **cutsq, double **host_lj1,
                          double **host_lj2, double **host_lj3, double **host_lj4,
                          double **offset, double *special_lj, const int inum,
                          const int tH, const int tO, const double alpha,
                          const double qdist, const int nall, const int max_nbors,
                          const int maxspecial, const double cell_size, int &gpu_mode,
                          FILE *screen, double **host_cut_ljsq,
                          const double host_cut_coulsq,
                          const double host_cut_coulsqplus,
                          double *host_special_coul, const double qqrd2e,
                          const double *host_taylor, const int host_taylor_terms,
                          const double host_b, const double host_sigma,
                          const int host_mmax, const double host_w0,
                          int map_size, int max_same,
                          double **host_lambda, const double host_nlambda,
                          const double host_lam_charge)
{
  LJTIP4PUSF.clear();
  gpu_mode = LJTIP4PUSF.device->gpu_mode();
  const double gpu_split = LJTIP4PUSF.device->particle_split();
  const int first_gpu = LJTIP4PUSF.device->first_device();
  const int last_gpu = LJTIP4PUSF.device->last_device();
  const int world_me = LJTIP4PUSF.device->world_me();
  const int gpu_rank = LJTIP4PUSF.device->gpu_rank();
  const int procs_per_gpu = LJTIP4PUSF.device->procs_per_gpu();

  LJTIP4PUSF.device->init_message(screen, "lj/cut/tip4p/user/soft/gpu", first_gpu, last_gpu);

  bool message = false;
  if (LJTIP4PUSF.device->replica_me() == 0 && screen) message = true;

  if (message) {
    fprintf(screen, "Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok = 0;
  if (world_me == 0)
    init_ok = LJTIP4PUSF.init(ntypes, cutsq, host_lj1, host_lj2, host_lj3, host_lj4, offset,
                             special_lj, inum, tH, tO, alpha, qdist, nall, max_nbors,
                             maxspecial, cell_size, gpu_split, screen, host_cut_ljsq,
                             host_cut_coulsq, host_cut_coulsqplus, host_special_coul, qqrd2e,
                             host_taylor, host_taylor_terms, host_b, host_sigma, host_mmax,
                             host_w0, map_size, max_same, host_lambda, host_nlambda,
                             host_lam_charge);

  LJTIP4PUSF.device->world_barrier();
  if (message) fprintf(screen, "Done.\n");

  for (int i = 0; i < procs_per_gpu; i++) {
    if (message) {
      if (last_gpu - first_gpu == 0)
        fprintf(screen, "Initializing Device %d on core %d...", first_gpu, i);
      else
        fprintf(screen, "Initializing Devices %d-%d on core %d...", first_gpu, last_gpu, i);
      fflush(screen);
    }
    if (gpu_rank == i && world_me != 0)
      init_ok = LJTIP4PUSF.init(ntypes, cutsq, host_lj1, host_lj2, host_lj3, host_lj4, offset,
                               special_lj, inum, tH, tO, alpha, qdist, nall, max_nbors,
                               maxspecial, cell_size, gpu_split, screen, host_cut_ljsq,
                               host_cut_coulsq, host_cut_coulsqplus, host_special_coul, qqrd2e,
                               host_taylor, host_taylor_terms, host_b, host_sigma, host_mmax,
                               host_w0, map_size, max_same, host_lambda, host_nlambda,
                               host_lam_charge);

    LJTIP4PUSF.device->serialize_init();
    if (message) fprintf(screen, "Done.\n");
  }
  if (message) fprintf(screen, "\n");

  if (init_ok == 0) LJTIP4PUSF.estimate_gpu_overhead(2);
  return init_ok;
}

void ljtip4p_user_soft_gpu_clear() { LJTIP4PUSF.clear(); }

int **ljtip4p_user_soft_gpu_compute_n(const int ago, const int inum_full, const int nall,
                                 double **host_x, int *host_type, double *sublo,
                                 double *subhi, tagint *tag, int *map_array, int map_size,
                                 int *sametag, int max_same, int **nspecial,
                                 tagint **special, const bool eflag, const bool vflag,
                                 const bool eatom, const bool vatom, int &host_start,
                                 int **ilist, int **jnum, const double cpu_time,
                                 bool &success, double *host_q, double *boxlo,
                                 double *prd, int *periodicity)
{
  return LJTIP4PUSF.compute(ago, inum_full, nall, host_x, host_type, sublo, subhi, tag,
                           map_array, map_size, sametag, max_same, nspecial, special,
                           eflag, vflag, eatom, vatom, host_start, ilist, jnum, cpu_time,
                           success, host_q, boxlo, prd, periodicity);
}

void ljtip4p_user_soft_gpu_compute(const int ago, const int inum_full, const int nall,
                              double **host_x, int *host_type, int *ilist, int *numj,
                              int **firstneigh, const bool eflag, const bool vflag,
                              const bool eatom, const bool vatom, int &host_start,
                              const double cpu_time, bool &success, double *host_q,
                              const int nlocal, double *boxlo, double *prd)
{
  LJTIP4PUSF.compute(ago, inum_full, nall, host_x, host_type, ilist, numj, firstneigh, eflag,
                    vflag, eatom, vatom, host_start, cpu_time, success, host_q, nlocal, boxlo,
                    prd);
}

double ljtip4p_user_soft_gpu_bytes() { return LJTIP4PUSF.host_memory_usage(); }

double ljtip4p_user_soft_gpu_get_du_dlam() { return LJTIP4PUSF.get_du_dlam(); }

void ljtip4p_user_soft_copy_molecule_data(int n, tagint *tag, int *map_array, int map_size,
                                     int *sametag, int max_same, int ago)
{
  LJTIP4PUSF.copy_relations_data(n, tag, map_array, map_size, sametag, max_same, ago);
}
