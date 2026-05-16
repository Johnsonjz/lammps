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

PairStyle(lj/cut/coul/drbsog, PairLJCutCoulDRBSOG)
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_DRBSOG_H
#define LMP_PAIR_LJ_CUT_COUL_DRBSOG_H

#include "pair.h"

namespace LAMMPS_NS {

	class PairLJCutCoulDRBSOG : public Pair {

	public:
		PairLJCutCoulDRBSOG(class LAMMPS*);
		virtual ~PairLJCutCoulDRBSOG();
		virtual void compute(int, int);
		virtual void settings(int, char**);
		virtual void* extract(const char*, int&);
		virtual void allocate();
		void coeff(int, char**);
		virtual void init_style();
		virtual double init_one(int, int);
		//virtual void* extract(const char*, int&);

	protected:
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

		double* TaylorCoeff;
		int TaylorTerms;

		//float b, Sigma;
		//int Mmax;

		//float r0,w0;

		int Step;
		double* TimeSet;
		};
	}

#endif
#endif 