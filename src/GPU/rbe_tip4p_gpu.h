/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(rbe/tip4p/gpu, RBETIP4PGPU)
// clang-format on
#else

#ifndef LMP_RBE_TIP4P_GPU_H
#define LMP_RBE_TIP4P_GPU_H

#include "rbe_tip4p.h"

namespace LAMMPS_NS {

class RBETIP4PGPU : public RBETIP4P {
 public:
  explicit RBETIP4PGPU(class LAMMPS *);
  void init() override;
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
