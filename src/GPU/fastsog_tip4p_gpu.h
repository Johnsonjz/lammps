/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   FastSOG/TIP4P GPU acceleration.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(fastsog/tip4p/gpu, FastSOGTIP4PGPU)
// clang-format on
#else

#ifndef LMP_FASTSOG_TIP4P_GPU_H
#define LMP_FASTSOG_TIP4P_GPU_H

#include "fastsog_tip4p.h"

namespace LAMMPS_NS {

class FastSOGTIP4PGPU : public FastSOGTIP4P {
 public:
  explicit FastSOGTIP4PGPU(class LAMMPS *lmp);
  void init() override;
  void compute(int eflag, int vflag) override;
};

}  // namespace LAMMPS_NS

#endif
#endif
