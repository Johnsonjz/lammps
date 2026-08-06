/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "rbe_gpu.h"

#include "gpu_extra.h"
#include "lammps.h"

using namespace LAMMPS_NS;

RbeGPU::RbeGPU(LAMMPS *lmp) : Rbe(lmp)
{
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
  use_gpu_accel = true;
}

void RbeGPU::init()
{
  Rbe::init();
}

void RbeGPU::compute(int eflag, int vflag)
{
  Rbe::compute(eflag, vflag);
}
