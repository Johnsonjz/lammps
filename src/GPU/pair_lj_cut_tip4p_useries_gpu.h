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
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/tip4p/user/gpu,PairLJCutTIP4PUserGPU);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_TIP4P_USER_GPU_H
#define LMP_PAIR_LJ_TIP4P_USER_GPU_H

#include "pair_lj_cut_tip4p_useries.h"

namespace LAMMPS_NS {

class PairLJCutTIP4PUserGPU : public PairLJCutTIP4PUser {
 public:
  PairLJCutTIP4PUserGPU(LAMMPS *lmp);
  ~PairLJCutTIP4PUserGPU() override;
  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif
