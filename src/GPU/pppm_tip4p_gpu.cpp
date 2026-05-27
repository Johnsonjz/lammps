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

#include "pppm_tip4p_gpu.h"

#include "gpu_extra.h"
#include "lammps.h"

using namespace LAMMPS_NS;

PPPMTIP4PGPU::PPPMTIP4PGPU(LAMMPS *lmp) : PPPMTIP4P(lmp)
{
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

void PPPMTIP4PGPU::init()
{
  PPPMTIP4P::init();
}

void PPPMTIP4PGPU::compute(int eflag, int vflag)
{
  // Keep TIP4P/M-site physics identical to PPPMTIP4P.
  // The /gpu style hooks GPU package readiness and creates a dedicated
  // style entry for benchmarking and future kernel offload work.
  PPPMTIP4P::compute(eflag, vflag);
}
