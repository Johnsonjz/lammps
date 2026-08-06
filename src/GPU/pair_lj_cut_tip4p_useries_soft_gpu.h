/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Vsevolod Nikolskiy (HSE)
   Soft-core variant: adds lam1 per-type-pair scaling
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/tip4p/user/soft/gpu,PairLJCutTIP4PUserSoftGPU);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_TIP4P_USER_SOFT_GPU_H
#define LMP_PAIR_LJ_TIP4P_USER_SOFT_GPU_H

#include "pair_lj_cut_tip4p_useries_soft.h"

namespace LAMMPS_NS {

class PairLJCutTIP4PUserSoftGPU : public PairLJCutTIP4PUserSoft {
 public:
  PairLJCutTIP4PUserSoftGPU(LAMMPS *lmp);
  ~PairLJCutTIP4PUserSoftGPU() override;
  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

  double du_dlam_accum;   // running sum of dU/dλ
  bigint du_dlam_count;   // number of timesteps accumulated
  void reset_dudlam() { du_dlam_accum = 0.0; du_dlam_count = 0; }

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif
