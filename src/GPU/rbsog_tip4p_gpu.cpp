/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "rbsog_tip4p_gpu.h"

#include "gpu_extra.h"
#include "lammps.h"

using namespace LAMMPS_NS;

RBSOGTIP4PGPU::RBSOGTIP4PGPU(LAMMPS *lmp) : RBSOGTIP4P(lmp)
{
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
  use_gpu_accel = true;
}

void RBSOGTIP4PGPU::init()
{
  RBSOGTIP4P::init();
}

void RBSOGTIP4PGPU::compute(int eflag, int vflag)
{
  RBSOGTIP4P::compute(eflag, vflag);
}
