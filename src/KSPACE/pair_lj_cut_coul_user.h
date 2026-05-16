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

PairStyle(lj/cut/coul/user, PairLJCutCoulUser)

#else

#ifndef LMP_PAIR_LJ_CUT_COUL_USER_H
#define LMP_PAIR_LJ_CUT_COUL_USER_H

#include "pair.h"

namespace LAMMPS_NS {

	class PairLJCutCoulUser : public Pair {

	public:
		PairLJCutCoulUser(class LAMMPS*);
		~PairLJCutCoulUser() override;
		void compute(int, int) override;
		void settings(int, char**) override;
		void* extract(const char*, int&) override;
		void coeff(int, char**) override;
		void init_style() override;
		double init_one(int, int) override;
		//virtual void* extract(const char*, int&);

	protected:
		inline double G_Sigma1(double, double);
		double Compute_W01(double, double);
		double cut_lj_global;
		double** cut_lj, ** cut_ljsq;
		double cut_coul, cut_coulsq;
		double qdist;             // TIP4P distance from O site to negative charge
		double** lj1, ** lj2, ** lj3, ** lj4, ** offset;
		double* cut_respa;
		//double qdist;             // TIP4P distance from O site to negative charge
		double g_ewald;
		double** epsilon, ** sigma;

		double coef;

		double* BL, * BL3INV, * BL2SIGMA2INV;//Setting if applying vectorization
		
		double* TaylorCoeff;
		int TaylorTerms;

		int Step;
		double* TimeSet;
		virtual void allocate();
	};
}

#endif
#endif