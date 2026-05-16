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

KSpaceStyle(rbsog/tip4p, RBSOGTIP4P)

#else

#ifndef LMP_RBSOG_TIP4P_H
#define LMP_RBSOG_TIP4P_H

// #include "kspace.h"
#include "rbsog_intel.h"

namespace LAMMPS_NS {

	class RBSOGTIP4P : public RBSOG {
	public:
		RBSOGTIP4P(class LAMMPS*);
		// virtual ~RBSOG();
		void init() override;
        void compute(int, int) override;
	protected:
		  // TIP4P settings
        int typeH, typeO;    // atom types of TIP4P water H and O atoms
        double qdist;        // distance from O site to negative charge
        double alpha;        // geometric factor
    private:
        void find_M(int, int &, int &, double *);
	};

}

#endif
#endif 