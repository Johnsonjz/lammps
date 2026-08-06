/**************************************************************************
                               lj_tip4p_user_soft.h
                             -------------------
                              V. Nikolskiy (HSE)

  Class for acceleration of the lj/tip4p/user/soft pair style
  (soft-core variant: adds lam1 per-type-pair scaling)

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : thevsevak@gmail.com
***************************************************************************/

#ifndef LAL_LJ_TIP4P_USER_SOFT_H
#define LAL_LJ_TIP4P_USER_SOFT_H

#include "lal_base_charge.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class LJ_TIP4PUserSoft : public BaseCharge<numtyp, acctyp> {
public:
  LJ_TIP4PUserSoft();
  ~LJ_TIP4PUserSoft();

  /// Clear any previous data and set up for a new LAMMPS run
  int init(const int ntypes, double **host_cutsq,
           double **host_lj1, double **host_lj2, double **host_lj3,
           double **host_lj4, double **host_offset, double *host_special_lj,
           const int nlocal, const int tH, const int tO,
           const double alpha, const double qdist,
           const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen,
           double **host_cut_ljsq,
           const double host_cut_coulsq, const double host_cut_coulsqplus,
           double *host_special_coul, const double qqrd2e,
           const double *host_taylor, const int host_taylor_terms,
           const double host_b, const double host_sigma,
           const int host_mmax, const double host_w0,
           int map_size, int max_same,
           double **host_lambda, const double host_nlambda,
           const double host_lam_charge);

  /// Clear all host and device data
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  /// Copy data from LAMMPS_NS
  void copy_relations_data(int n, tagint *tag, int *map_array, int map_size,
                           int *sametag, int max_same, int ago);

  /// Reimplement BaseCharge pair loop with host neighboring
  void compute(const int f_ago, const int inum_full, const int nall,
               double **host_x, int *host_type, int *ilist, int *numj,
               int **firstneigh, const bool eflag, const bool vflag,
               const bool eatom, const bool vatom, int &host_start,
               const double cpu_time, bool &success, double *charge,
               const int nlocal, double *boxlo, double *prd);

  /// Reimplement BaseCharge pair loop with device neighboring
  int **compute(const int ago, const int inum_full, const int nall,
                double **host_x, int *host_type, double *sublo, double *subhi,
                tagint *tag, int *map_array, int map_size, int *sametag,
                int max_same, int **nspecial, tagint **special,
                const bool eflag, const bool vflag, const bool eatom,
                const bool vatom, int &host_start, int **ilist, int **jnum,
                const double cpu_time, bool &success, double *charge,
                double *boxlo, double *prd, int *periodicity);

  // --------------------------- TYPE DATA --------------------------

  /// lj1.x = lj1, lj1.y = lj2, lj1.z = cutsq_vdw
  UCL_D_Vec<numtyp4> lj1;
  /// lj3.x = lj3, lj3.y = lj4, lj3.z = offset
  UCL_D_Vec<numtyp4> lj3;
  /// cutsq
  UCL_D_Vec<numtyp> cutsq;
  /// User-series Taylor coefficients
  UCL_D_Vec<numtyp> taylor;
  /// Special LJ values [0-3] and Special Coul values [4-7]
  UCL_D_Vec<numtyp> sp_lj;
  /// Per-type-pair lam1 = pow(lambda, nlambda) for soft-core scaling
  UCL_D_Vec<numtyp> lam1;
  /// Per-type-pair soft-core shifts packed: .x = sc_lj, .y = sc_c
  UCL_D_Vec<numtyp4> sc;

  /// Soft-core parameters
  numtyp _alpha_lj, _alpha_c, _power_n;

  /// Charge-scaling lambda for ∂U/∂λ (separate from per-type-pair lam1;
  /// needed because lam1=0 for O-O/O-H but Coulomb ∂U/∂λ still needs λ)
  numtyp _lam_charge;

  /// Total dU/dλ for current timestep (soft-core corrected)
  acctyp du_dlam_total;

  /// Getter for du_dlam_total
  double get_du_dlam() const { return (double)du_dlam_total; }

  bool shared_types;

  /// Number of atom types
  int _lj_types;

  numtyp _qqrd2e;
  int _taylor_terms;
  int _mmax;
  numtyp4 _user_coeff;

  /// TIP4P water parameters
  int TypeO, TypeH;
  numtyp alpha, qdist;
  numtyp cut_coulsq, cut_coulsqplus;

  UCL_D_Vec<acctyp> du_dlam;  // single-element: total dU/dλ (atomicAdd in kernel)

  UCL_D_Vec<int> hneight;
  UCL_D_Vec<numtyp4> m;      // position and charge of virtual particle
  UCL_D_Vec<acctyp4> ansO;   // force applied to virtual particle

  UCL_D_Vec<tagint> tag;
  UCL_D_Vec<int> map_array;
  UCL_D_Vec<int> atom_sametag;

  UCL_Kernel k_pair_distrib, k_pair_reneigh, k_pair_newsite;
  UCL_Kernel k_pair_distrib_noev, *k_pair_dt_sel;

private:
  bool _allocated;
  int t_ago;
  int loop(const int eflag, const int vflag);
};

}    // namespace LAMMPS_AL

#endif
