#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(rbe/4p, RBE4P)
// clang-format on
#else

#ifndef LMP_RBE4P_H
#define LMP_RBE4P_H

#include "kspace.h"
#include<iostream>
#include <fstream>
#include <sstream>

namespace LAMMPS_NS {
	
class RBE4P : public KSpace {
public:
  RBE4P(class LAMMPS *);
  virtual ~RBE4P();
  void settings(int, char**);
  void init();//在运行前初始化 
  void setup();//在跑第一个时间步前要做的东西
  void compute(int, int);//每个时间步里都要做的
  double memory_usage();//内存使用统计
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
  int triclinic;    // domain settings, orthog or triclinic
private:
  void find_M(int, int &, int &, double *);
};
}
#endif
#endif

