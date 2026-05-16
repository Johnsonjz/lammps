#ifdef KSPACE_CLASS

KSpaceStyle(rbe/disp, RbeDisp)

#else

#ifndef LMP_RBE_DISP_H
#define LMP_RBE_DISP_H

#include "kspace.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace LAMMPS_NS {
	
class RbeDisp : public KSpace {
public:
  RbeDisp(class LAMMPS *);
  virtual ~RbeDisp();
  void settings(int, char**);
  void init();//在运行前初始化 
  void setup();//在跑第一个时间步前要做的东西
  void compute(int, int);//每个时间步里都要做的
  double memory_usage();//内存使用统计
  void slabcorr();
  double randn_box_muller_linear_congruential1(const double Mean, const double SquareMargin);
  inline double Prob(double m, double alpha, double L, double S1, double pi);
  double MH_D(double m, double alpha, double L, double pi);

public:
  //double Lx, Ly, Lz, c_cut, alpha, alpha_sqrt;
  double alpha, alpha_sqrt, g_ewald;
  //int N, P; 
  int P;
  double S;
  double pi,pi_sqrt;
  double volume;
  double mumurd2e;
  double q2;
  double musum, musqsum, mu2;
  std::streamoff pointer;
  std::ifstream infile;
  int lx,ly,lz;
  //int Kspace_Index;
  int Step;
  int me;
  int (*K_All)[3];//[1000000][3];
  float *Time;
  int RankID;
  double S1_X, S1_Y, S1_Z;
  float** K_Sample;
  int function[2];
  int nfunctions;


protected:
  void musum_musq();
};
}

#endif
#endif

