/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#include "rbe_tip4p_gpu.h"

#include "gpu_extra.h"
#include "lammps.h"

using namespace LAMMPS_NS;

RBETIP4PGPU::RBETIP4PGPU(LAMMPS *lmp) : RBETIP4P(lmp)
{
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
  use_gpu_accel = true;
}

void RBETIP4PGPU::init()
{
  RBETIP4P::init();
}

void RBETIP4PGPU::compute(int eflag, int vflag)
{
  RBETIP4P::compute(eflag, vflag);
}
