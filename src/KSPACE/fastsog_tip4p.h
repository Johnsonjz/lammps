/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Zhen Jiang (SJTU)
   FastSOGTIP4P: FastSOG kspace solver for TIP4P water models.
   Inherits from FastSOG; overrides charge spreading to use M-site
   position and force redistribution for the TIP4P geometry.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(fastsog/tip4p, FastSOGTIP4P)
// clang-format on
#else

#ifndef LMP_FASTSOG_TIP4P_H
#define LMP_FASTSOG_TIP4P_H

#include "fastsog.h"

namespace LAMMPS_NS {

class FastSOGTIP4P : public FastSOG {
 public:
  explicit FastSOGTIP4P(class LAMMPS *lmp);
  void init() override;
  void compute(int eflag, int vflag) override;

 protected:
  // TIP4P geometry
  int typeH, typeO;     // atom types of TIP4P water H and O atoms
  double qdist;         // distance from O site to negative charge (M site)
  double alpha;         // geometric factor for force redistribution

 protected:
  void find_M(int i, int &iH1, int &iH2, double *xM);
};

}  // namespace LAMMPS_NS

#endif
#endif
