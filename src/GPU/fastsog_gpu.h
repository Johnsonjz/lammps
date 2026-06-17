/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   FastSOG GPU acceleration — inherits from FastSOG, offloads charge
   spreading and force interpolation to GPU via LAMMPS_AL library.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(fastsog/gpu, FastSOGGPU)
// clang-format on
#else

#ifndef LMP_FASTSOG_GPU_H
#define LMP_FASTSOG_GPU_H

#include "fastsog.h"

namespace LAMMPS_NS {

class FastSOGGPU : public FastSOG {
 public:
  explicit FastSOGGPU(class LAMMPS *lmp);
  ~FastSOGGPU() override;
  void init() override;
  void setup() override;
  void compute(int eflag, int vflag) override;

 protected:
  bool gpu_ready;
};

}  // namespace LAMMPS_NS

#endif
#endif
