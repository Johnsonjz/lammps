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

KSpaceStyle(drbsog, DRBSOG)
#else

#ifndef LMP_DRBSOG_H
#define LMP_DRBSOG_H

#include "kspace.h"

namespace LAMMPS_NS {

	class DRBSOG : public KSpace {
	public:
		DRBSOG(class LAMMPS*);
		virtual ~DRBSOG();
		void init();
		void setup();
		virtual void settings(int, char**);
		virtual void compute(int, int);
		double memory_usage();
	protected:
		double h;
		double t_0;
		int M1,M2;
		double w_M2;
		double lambda;
		int P;

		double S;//归一化系数
		double* TimeSet;//算法计时
		double** K_Sample;//用于存储样本
		double rcut;//近场截断半径
		int Kmax;//注意这里在算S的时候很重要
		double V;

		int Kcut;//前面扔掉的Fourier区域
		double KbT;//记录能量计算
		int RankID;//当前的MPI

		double QsumQ;
		bigint Step;
	};
}
#endif
#endif