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

#ifdef KSPACE_CLASS

KSpaceStyle(rbsog, RBSOG)

#else

#ifndef LMP_RBSOG_H
#define LMP_RBSOG_H

#include "kspace.h"
#include <immintrin.h> 

namespace LAMMPS_NS {

	class RBSOG : public KSpace {
	public:
		RBSOG(class LAMMPS*);
		virtual ~RBSOG();
		void init() override;
		void setup() override;
		void settings(int, char**) override;
		virtual void compute(int, int) override;
		double memory_usage() override;
		double MH_D_Modify(int, double);
  		double randn_box_muller_linear_congruential(const double, const double);
		float Compute_W0(float, float);
		float Gaussian(int, int, int, float, float, float, float, float, float, int, float*);
		float Gaussian_Fourier_Plus(float, float, float, float, float, float, int, float*);
		float Gaussian_modify(int, int, int, float, float, float, float, float, float, int, float*);
		float Gaussian_Fourier_Plus_modify(float, float, float, float, float, float, int, float*);
		inline float G_sigma(float, float);
		__m512 Gaussian_Fourier_Plus_AVX(__m512, __m512, __m512, float, float, float, int, float*);
		// __m512 Gaussian_Fourier_Plus_modify_AVX(__m512, __m512, __m512, float, float, float, int, float*);

	protected:
		float b;
		float sigma;
		int Mmax;
		float r0;
		float w0;
		double S, S_npt;
		double S0, S_npt0;
		double xprd0, yprd0, zprd0;
		float rcut;
		float volume;
		int Kmax;
		int RankID;
		int P;
  		int triclinic;    // domain settings, orthog or triclinic
		int Kcut;
		int** K_Sample;
		int* idx_npt;
		bigint Step;
		float* sl;
		float* coef;
		float* coef_npt;
		double* TimeSet;
		double* TimeSet_sampling;
	};

}

#endif
#endif 