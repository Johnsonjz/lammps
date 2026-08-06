/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(rbe/gpu,RbeGPU)
// clang-format on
#else

#ifndef LMP_RBE_GPU_H
#define LMP_RBE_GPU_H

#include "rbe.h"

namespace LAMMPS_NS {

class RbeGPU : public Rbe {
 public:
  explicit RbeGPU(class LAMMPS *);
  void init() override;
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
