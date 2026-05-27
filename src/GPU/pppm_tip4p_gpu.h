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

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/tip4p/gpu,PPPMTIP4PGPU)
// clang-format on
#else

#ifndef LMP_PPPM_TIP4P_GPU_H
#define LMP_PPPM_TIP4P_GPU_H

#include "pppm_tip4p.h"

namespace LAMMPS_NS {

class PPPMTIP4PGPU : public PPPMTIP4P {
 public:
  explicit PPPMTIP4PGPU(class LAMMPS *);
  void init() override;
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
