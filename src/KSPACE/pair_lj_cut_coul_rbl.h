/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

#ifdef LMP_ENABLE_EXPERIMENTAL_DRBSOG
PairStyle(lj/cut/coul/rbl, PairLJCutCoulRBL)
#endif

#else

#ifndef LMP_PAIR_LJ_CUT_COUL_RBL_H
#define LMP_PAIR_LJ_CUT_COUL_RBL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutCoulRBL : public Pair {

public:
	PairLJCutCoulRBL(class LAMMPS*);
	virtual ~PairLJCutCoulRBL(); 
	virtual void compute(int, int);
	virtual void settings(int, char**);
	void coeff(int, char**);
	virtual void init_style();
	virtual double init_one(int, int);
	virtual void* extract(const char*, int&);

protected:
	double cut_lj_global; 
	double** cut_lj, ** cut_ljsq;
	double cut_coul, cut_coulsq;
	double** epsilon, ** sigma;
	double* cut_respa;
	double qdist;             // TIP4P distance from O site to negative charge
	double g_ewald;
	double** lj1, ** lj2, ** lj3, ** lj4, ** offset;
	double* TimeSet;
	double* Time_Sampling; 
	double* Time_Compute;
	double	core_size, P;

	virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/rbl requires atom attribute q

The atom style defined does not have this attribute.

E: Pair style requires a KSpace style

No kspace style is defined.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/