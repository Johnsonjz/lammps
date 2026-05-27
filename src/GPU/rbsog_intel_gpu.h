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
KSpaceStyle(rbsog/gpu,RBSOGGPU)
// clang-format on
#else

#ifndef LMP_RBSOG_GPU_H
#define LMP_RBSOG_GPU_H

#include "rbsog_intel.h"

namespace LAMMPS_NS {

class RBSOGGPU : public RBSOG {
 public:
  explicit RBSOGGPU(class LAMMPS *);
  void init() override;
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
