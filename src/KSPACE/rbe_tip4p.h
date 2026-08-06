#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(rbe/tip4p, RBETIP4P)
// clang-format on
#else

#ifndef LMP_RBETIP4P_H
#define LMP_RBETIP4P_H

#include "kspace.h"
#include<iostream>
#include <fstream>
#include <sstream>

namespace LAMMPS_NS {
	
class RBETIP4P : public KSpace {
public:
  RBETIP4P(class LAMMPS *);
  virtual ~RBETIP4P();
  void settings(int, char**);
  void init();
  void setup() ;
  void compute(int, int);
  double memory_usage();
  void slabcorr();
  double randn_box_muller_linear_congruential1(const double Mean, const double SquareMargin);
  inline double Prob(double m, double alpha, double L, double S1, double pi);
  double MH_D(double m, double alpha, double L, double pi);

protected:
  // TIP4P settings
  int typeH, typeO;    // atom types of TIP4P water H and O atoms
  double qdist;        // distance from O site to negative charge
  double alpha;        // geometric factor
  double beta; //
  int P;                    //
  double S0, S;
  double pi,pi_sqrt;
  double volume;
  double cutoff;
  std::streamoff pointer; 
  std::ifstream infile;
  int lx,ly,lz;
  int Step;
  int me;
  int (*K_All)[3];
   double* TimeSet;
  int RankID; //
  double S1_X, S1_Y, S1_Z;
  double S_X, S_Y, S_Z;
  double xprd0, yprd0, zprd0;
  float** K_Sample;
  bool use_gpu_accel;
  int triclinic;    // domain settings, orthog or triclinic
private:
  void find_M(int, int &, int &, double *);
};
}
#endif
#endif

