/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Zhen Jiang (SJTU)
------------------------------------------------------------------------- */

#include "fastsog_tip4p.h"
#include "fastsog_spline.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "pair.h"
#include "angle.h"
#include "bond.h"
#include "memory.h"
#include "fft3d_wrap.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FastSOGTIP4P::FastSOGTIP4P(LAMMPS *lmp) : FastSOG(lmp)
{
  tip4pflag = 1;
  typeH = typeO = 0;
  qdist = 0.0;
  alpha = 0.0;
}

/* ---------------------------------------------------------------------- */

void FastSOGTIP4P::init()
{
  // TIP4P FastSOG requires newton on, b/c it computes forces on ghost atoms
  if (force->newton == 0)
    error->all(FLERR, "Kspace style fastsog/tip4p requires newton on");

  FastSOG::init();

  // Extract TIP4P geometry parameters from pair style
  int itmp;
  if (comm->me == 0)
    utils::logmesg(lmp, "  extracting TIP4P info from pair style\n");

  auto *p_qdist = (double *)force->pair->extract("qdist", itmp);
  int *p_typeO = (int *)force->pair->extract("typeO", itmp);
  int *p_typeH = (int *)force->pair->extract("typeH", itmp);
  int *p_typeA = (int *)force->pair->extract("typeA", itmp);
  int *p_typeB = (int *)force->pair->extract("typeB", itmp);

  if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
    error->all(FLERR, "Pair style is incompatible with TIP4P KSpace style");

  qdist = *p_qdist;
  typeO = *p_typeO;
  typeH = *p_typeH;
  int typeA = *p_typeA;
  int typeB = *p_typeB;

  if (force->angle == nullptr || force->bond == nullptr ||
      force->angle->setflag == nullptr || force->bond->setflag == nullptr)
    error->all(FLERR, "Bond and angle potentials must be defined for TIP4P");
  if (typeA < 1 || typeA > atom->nangletypes ||
      force->angle->setflag[typeA] == 0)
    error->all(FLERR, "Bad TIP4P angle type for fastsog/tip4p");
  if (typeB < 1 || typeB > atom->nbondtypes ||
      force->bond->setflag[typeB] == 0)
    error->all(FLERR, "Bad TIP4P bond type for fastsog/tip4p");

  double theta = force->angle->equilibrium_angle(typeA);
  double blen = force->bond->equilibrium_distance(typeB);
  alpha = qdist / (cos(0.5 * theta) * blen);

  if (comm->me == 0)
    utils::logmesg(lmp, fmt::format("  TIP4P qdist={:.6g} typeO={} typeH={} "
                   "alpha={:.6g}\n", qdist, typeO, typeH, alpha));
}

/* ----------------------------------------------------------------------
   find M-site position for TIP4P oxygen atom i
   iH1, iH2 = indices of bonded hydrogen atoms (closest images)
   xM = computed M-site position (3-vector, output)
------------------------------------------------------------------------- */

void FastSOGTIP4P::find_M(int i, int &iH1, int &iH2, double *xM)
{
  double **x = atom->x;

  // Find H1 and H2 atoms bonded to O (tag = i's tag + 1, +2)
  iH1 = atom->map(atom->tag[i] + 1);
  iH2 = atom->map(atom->tag[i] + 2);

  if (iH1 == -1 || iH2 == -1)
    error->one(FLERR, "TIP4P hydrogen is missing");
  if (atom->type[iH1] != typeH || atom->type[iH2] != typeH)
    error->one(FLERR, "TIP4P hydrogen has incorrect atom type");

  // Find closest images of H atoms to O
  iH1 = domain->closest_image(i, iH1);
  iH2 = domain->closest_image(i, iH2);

  // Compute M site: O + alpha * 0.5 * (OH1 + OH2)
  double delx1 = x[iH1][0] - x[i][0];
  double dely1 = x[iH1][1] - x[i][1];
  double delz1 = x[iH1][2] - x[i][2];

  double delx2 = x[iH2][0] - x[i][0];
  double dely2 = x[iH2][1] - x[i][1];
  double delz2 = x[iH2][2] - x[i][2];

  xM[0] = x[i][0] + alpha * 0.5 * (delx1 + delx2);
  xM[1] = x[i][1] + alpha * 0.5 * (dely1 + dely2);
  xM[2] = x[i][2] + alpha * 0.5 * (delz1 + delz2);
}

/* ----------------------------------------------------------------------
   Override compute: use M-site for charge spreading, redistribute forces
------------------------------------------------------------------------- */

void FastSOGTIP4P::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  if (qsqsum == 0.0) {
    energy = 0.0;
    for (int j = 0; j < 6; ++j) virial[j] = 0.0;
    return;
  }

  ensure_fft_plan();

  const bool want_energy = (eflag & ENERGY_GLOBAL);
  const bool want_virial = (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR));

  energy = 0.0;
  for (int j = 0; j < 6; ++j) virial[j] = 0.0;

  const int nlocal = atom->nlocal;
  if (nlocal <= 0) return;

  // ── Clear mesh arrays ──
  std::fill(mesh_rho.begin(), mesh_rho.end(), 0.0);
  std::fill(mesh_fft_work.begin(), mesh_fft_work.end(), 0.0);
  std::fill(mesh_gradx.begin(), mesh_gradx.end(), 0.0);
  std::fill(mesh_grady.begin(), mesh_grady.end(), 0.0);
  std::fill(mesh_gradz.begin(), mesh_gradz.end(), 0.0);

  const double volume = mesh_lx * mesh_ly * mesh_lz;
  const double rho_scale =
      static_cast<double>(mesh_nx * mesh_ny * mesh_nz) / volume;

  constexpr int assign_half = (kFastSOG_AssignOrder - 1) / 2;

  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;

  // ── Charge spreading with TIP4P M-site ──
  // Pre-find M-sites for all local O atoms
  std::vector<double> xM_vals(static_cast<size_t>(nlocal) * 3, 0.0);
  std::vector<int> iH1_vals(static_cast<size_t>(nlocal), -1);
  std::vector<int> iH2_vals(static_cast<size_t>(nlocal), -1);

  for (int i = 0; i < nlocal; ++i) {
    if (type[i] == typeO) {
      double xM[3];
      int iH1, iH2;
      find_M(i, iH1, iH2, xM);
      iH1_vals[static_cast<size_t>(i)] = iH1;
      iH2_vals[static_cast<size_t>(i)] = iH2;
      xM_vals[static_cast<size_t>(i) * 3 + 0] = xM[0];
      xM_vals[static_cast<size_t>(i) * 3 + 1] = xM[1];
      xM_vals[static_cast<size_t>(i) * 3 + 2] = xM[2];
    }
  }

  if (spline_type >= 4) {
    // CubeS₂ charge spreading with M-site
    const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;

    for (int i = 0; i < nlocal; ++i) {
      double xi_pos[3];
      if (type[i] == typeO) {
        xi_pos[0] = xM_vals[static_cast<size_t>(i) * 3 + 0];
        xi_pos[1] = xM_vals[static_cast<size_t>(i) * 3 + 1];
        xi_pos[2] = xM_vals[static_cast<size_t>(i) * 3 + 2];
      } else {
        xi_pos[0] = x[i][0];
        xi_pos[1] = x[i][1];
        xi_pos[2] = x[i][2];
      }

      const double fx = periodic_fraction(xi_pos[0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(xi_pos[1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(xi_pos[2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      const double q_scaled = rho_scale * q[i];

      if (spline_type == 4) {
        for (int k = 0; k < kCubes2NumNodes4; ++k) {
          const auto &node = kCubes2Nodes4[k];
          const double w = cubes2_weight_4(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          mesh_rho[idx] += static_cast<FFT_SCALAR>(q_scaled * w);
        }
      }
    }
  } else {
    // Legacy B-spline M-site spreading
    for (int i = 0; i < nlocal; ++i) {
      double xi_pos[3];
      if (type[i] == typeO) {
        xi_pos[0] = xM_vals[static_cast<size_t>(i) * 3 + 0];
        xi_pos[1] = xM_vals[static_cast<size_t>(i) * 3 + 1];
        xi_pos[2] = xM_vals[static_cast<size_t>(i) * 3 + 2];
      } else {
        xi_pos[0] = x[i][0];
        xi_pos[1] = x[i][1];
        xi_pos[2] = x[i][2];
      }

      const double fx = periodic_fraction(xi_pos[0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(xi_pos[1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(xi_pos[2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      std::array<double, kFastSOG_AssignOrder> wx, wy, wz;
      fastsog_bspline_weights_1d(tx, wx);
      fastsog_bspline_weights_1d(ty, wy);
      fastsog_bspline_weights_1d(tz, wz);

      std::array<int, kFastSOG_AssignOrder> ix, iy, iz;
      for (int a = 0; a < kFastSOG_AssignOrder; ++a) {
        ix[static_cast<size_t>(a)] =
            wrap_index(ix0 - assign_half + a, mesh_nx);
        iy[static_cast<size_t>(a)] =
            wrap_index(iy0 - assign_half + a, mesh_ny);
        iz[static_cast<size_t>(a)] =
            wrap_index(iz0 - assign_half + a, mesh_nz);
      }

      for (int a = 0; a < kFastSOG_AssignOrder; ++a) {
        for (int b = 0; b < kFastSOG_AssignOrder; ++b) {
          for (int c = 0; c < kFastSOG_AssignOrder; ++c) {
            const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
            mesh_rho[idx] += static_cast<FFT_SCALAR>(rho_scale * q[i] * wx[a] *
                                                      wy[b] * wz[c]);
          }
        }
      }
    }
  }

  // ── FFT forward ──
  const size_t ngrid = mesh_rho.size();
  for (size_t idx = 0; idx < ngrid; ++idx) {
    mesh_fft_work[2 * idx] = mesh_rho[idx];
    mesh_fft_work[2 * idx + 1] = 0.0;
  }
  mesh_fft->compute(mesh_fft_work.data(), mesh_fft_work.data(),
                    FFT3d::FORWARD);

  // ── k-space loop: multiply by Green functions, compute gradients ──
  const double scaleinv = 1.0 / static_cast<double>(ngrid);
  const double s2 = scaleinv * scaleinv;
  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;

  double energy_local = 0.0;
  double diag_sum_local = 0.0;
  std::array<double, 6> fv_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const size_t idx = mesh_index(ix, iy, iz);

        const double geff_energy = mesh_green_energy[idx];
        const double geff = mesh_green_force[idx];

        if (geff == 0.0 && geff_energy == 0.0) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        diag_sum_local += mesh_green_self[idx];

        const double rho_re =
            static_cast<double>(mesh_fft_work[2 * idx]);
        const double rho_im =
            static_cast<double>(mesh_fft_work[2 * idx + 1]);

        if (want_energy) {
          energy_local +=
              s2 * geff_energy * (rho_re * rho_re + rho_im * rho_im);
        }

        // Fourier-space virial (SOG-correct formula)
        // W_{αβ}(k) = s2 * |ρ|² * (ge * δ_{αβ} - gv * k_α * k_β)
        if (want_virial) {
          const double rho2 = rho_re * rho_re + rho_im * rho_im;
          const double gv = mesh_green_virial[idx];
          fv_local[0] += s2 * rho2 * (geff_energy - gv * kx * kx);
          fv_local[1] += s2 * rho2 * (geff_energy - gv * ky * ky);
          fv_local[2] += s2 * rho2 * (geff_energy - gv * kz * kz);
          fv_local[3] += s2 * rho2 * (-gv * kx * ky);
          fv_local[4] += s2 * rho2 * (-gv * kx * kz);
          fv_local[5] += s2 * rho2 * (-gv * ky * kz);
        }

        const double vk_re = scaleinv * geff * rho_re;
        const double vk_im = scaleinv * geff * rho_im;

        mesh_gradx[2 * idx] = static_cast<FFT_SCALAR>(-kx * vk_im);
        mesh_gradx[2 * idx + 1] = static_cast<FFT_SCALAR>(kx * vk_re);
        mesh_grady[2 * idx] = static_cast<FFT_SCALAR>(-ky * vk_im);
        mesh_grady[2 * idx + 1] = static_cast<FFT_SCALAR>(ky * vk_re);
        mesh_gradz[2 * idx] = static_cast<FFT_SCALAR>(-kz * vk_im);
        mesh_gradz[2 * idx + 1] = static_cast<FFT_SCALAR>(kz * vk_re);
      }
    }
  }

  // ── FFT backward (3 gradients) ──
  mesh_fft->compute(mesh_gradx.data(), mesh_gradx.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(), mesh_grady.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(), mesh_gradz.data(), FFT3d::BACKWARD);

  // ── Force interpolation with TIP4P redistribution ──
  const double qscale = force->qqrd2e * scale;
  std::array<double, 6> virial_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (spline_type >= 4) {
    const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;

    for (int i = 0; i < nlocal; ++i) {
      // Use M-site position for O atoms when interpolating forces
      double xi_pos[3];
      if (type[i] == typeO) {
        xi_pos[0] = xM_vals[static_cast<size_t>(i) * 3 + 0];
        xi_pos[1] = xM_vals[static_cast<size_t>(i) * 3 + 1];
        xi_pos[2] = xM_vals[static_cast<size_t>(i) * 3 + 2];
      } else {
        xi_pos[0] = x[i][0];
        xi_pos[1] = x[i][1];
        xi_pos[2] = x[i][2];
      }

      const double fx = periodic_fraction(xi_pos[0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(xi_pos[1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(xi_pos[2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      double gx = 0.0, gy = 0.0, gz = 0.0;

      if (spline_type == 4) {
        for (int k = 0; k < kCubes2NumNodes4; ++k) {
          const auto &node = kCubes2Nodes4[k];
          const double w = cubes2_weight_4(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          gx += w * static_cast<double>(mesh_gradx[2 * idx]);
          gy += w * static_cast<double>(mesh_grady[2 * idx]);
          gz += w * static_cast<double>(mesh_gradz[2 * idx]);
        }
      }

      const double qi = q[i];
      const double fxs = -qscale * qi * gx;
      const double fys = -qscale * qi * gy;
      const double fzs = -qscale * qi * gz;

      // TIP4P force redistribution
      if (type[i] == typeO) {
        int iH1 = iH1_vals[static_cast<size_t>(i)];
        int iH2 = iH2_vals[static_cast<size_t>(i)];

        atom->f[i][0] += fxs * (1.0 - alpha);
        atom->f[i][1] += fys * (1.0 - alpha);
        atom->f[i][2] += fzs * (1.0 - alpha);

        atom->f[iH1][0] += 0.5 * alpha * fxs;
        atom->f[iH1][1] += 0.5 * alpha * fys;
        atom->f[iH1][2] += 0.5 * alpha * fzs;

        atom->f[iH2][0] += 0.5 * alpha * fxs;
        atom->f[iH2][1] += 0.5 * alpha * fys;
        atom->f[iH2][2] += 0.5 * alpha * fzs;

        if (want_virial) {
          // Use M-site position for virial contribution (consistent with pppm_tip4p)
          double xM[3] = {xM_vals[static_cast<size_t>(i) * 3 + 0],
                          xM_vals[static_cast<size_t>(i) * 3 + 1],
                          xM_vals[static_cast<size_t>(i) * 3 + 2]};
          virial_local[0] += xM[0] * fxs;
          virial_local[1] += xM[1] * fys;
          virial_local[2] += xM[2] * fzs;
          virial_local[3] += xM[0] * fys;
          virial_local[4] += xM[0] * fzs;
          virial_local[5] += xM[1] * fzs;
        }
      } else {
        atom->f[i][0] += fxs;
        atom->f[i][1] += fys;
        atom->f[i][2] += fzs;

        if (want_virial) {
          virial_local[0] += x[i][0] * fxs;
          virial_local[1] += x[i][1] * fys;
          virial_local[2] += x[i][2] * fzs;
          virial_local[3] += x[i][0] * fys;
          virial_local[4] += x[i][0] * fzs;
          virial_local[5] += x[i][1] * fzs;
        }
      }
    }
  } else {
    // Legacy B-spline force interpolation with TIP4P redistribution
    for (int i = 0; i < nlocal; ++i) {
      double xi_pos[3];
      if (type[i] == typeO) {
        xi_pos[0] = xM_vals[static_cast<size_t>(i) * 3 + 0];
        xi_pos[1] = xM_vals[static_cast<size_t>(i) * 3 + 1];
        xi_pos[2] = xM_vals[static_cast<size_t>(i) * 3 + 2];
      } else {
        xi_pos[0] = x[i][0];
        xi_pos[1] = x[i][1];
        xi_pos[2] = x[i][2];
      }

      const double fx = periodic_fraction(xi_pos[0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(xi_pos[1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(xi_pos[2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      std::array<double, kFastSOG_AssignOrder> wx, wy, wz;
      fastsog_bspline_weights_1d(tx, wx);
      fastsog_bspline_weights_1d(ty, wy);
      fastsog_bspline_weights_1d(tz, wz);

      std::array<int, kFastSOG_AssignOrder> ix, iy, iz;
      for (int a = 0; a < kFastSOG_AssignOrder; ++a) {
        ix[static_cast<size_t>(a)] =
            wrap_index(ix0 - assign_half + a, mesh_nx);
        iy[static_cast<size_t>(a)] =
            wrap_index(iy0 - assign_half + a, mesh_ny);
        iz[static_cast<size_t>(a)] =
            wrap_index(iz0 - assign_half + a, mesh_nz);
      }

      double gx = 0.0, gy = 0.0, gz = 0.0;
      for (int a = 0; a < kFastSOG_AssignOrder; ++a) {
        for (int b = 0; b < kFastSOG_AssignOrder; ++b) {
          for (int c = 0; c < kFastSOG_AssignOrder; ++c) {
            const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
            const double w = wx[a] * wy[b] * wz[c];
            gx += w * static_cast<double>(mesh_gradx[2 * idx]);
            gy += w * static_cast<double>(mesh_grady[2 * idx]);
            gz += w * static_cast<double>(mesh_gradz[2 * idx]);
          }
        }
      }

      const double qi = q[i];
      const double fxs = -qscale * qi * gx;
      const double fys = -qscale * qi * gy;
      const double fzs = -qscale * qi * gz;

      // TIP4P force redistribution
      if (type[i] == typeO) {
        int iH1 = iH1_vals[static_cast<size_t>(i)];
        int iH2 = iH2_vals[static_cast<size_t>(i)];

        atom->f[i][0] += fxs * (1.0 - alpha);
        atom->f[i][1] += fys * (1.0 - alpha);
        atom->f[i][2] += fzs * (1.0 - alpha);

        atom->f[iH1][0] += 0.5 * alpha * fxs;
        atom->f[iH1][1] += 0.5 * alpha * fys;
        atom->f[iH1][2] += 0.5 * alpha * fzs;

        atom->f[iH2][0] += 0.5 * alpha * fxs;
        atom->f[iH2][1] += 0.5 * alpha * fys;
        atom->f[iH2][2] += 0.5 * alpha * fzs;

        if (want_virial) {
          double xM[3] = {xM_vals[static_cast<size_t>(i) * 3 + 0],
                          xM_vals[static_cast<size_t>(i) * 3 + 1],
                          xM_vals[static_cast<size_t>(i) * 3 + 2]};
          virial_local[0] += xM[0] * fxs;
          virial_local[1] += xM[1] * fys;
          virial_local[2] += xM[2] * fzs;
          virial_local[3] += xM[0] * fys;
          virial_local[4] += xM[0] * fzs;
          virial_local[5] += xM[1] * fzs;
        }
      } else {
        atom->f[i][0] += fxs;
        atom->f[i][1] += fys;
        atom->f[i][2] += fzs;

        if (want_virial) {
          virial_local[0] += x[i][0] * fxs;
          virial_local[1] += x[i][1] * fys;
          virial_local[2] += x[i][2] * fzs;
          virial_local[3] += x[i][0] * fys;
          virial_local[4] += x[i][0] * fzs;
          virial_local[5] += x[i][1] * fzs;
        }
      }
    }
  }

  // ── Energy ──
  if (want_energy) {
    double energy_all = 0.0;
    MPI_Allreduce(&energy_local, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);

    double diag_sum_all = 0.0;
    MPI_Allreduce(&diag_sum_local, &diag_sum_all, 1, MPI_DOUBLE, MPI_SUM,
                  world);

    double qsqsum_local = 0.0;
    for (int i = 0; i < nlocal; ++i) qsqsum_local += q[i] * q[i];
    double qsqsum_all = 0.0;
    MPI_Allreduce(&qsqsum_local, &qsqsum_all, 1, MPI_DOUBLE, MPI_SUM, world);

    energy = 0.5 * volume * energy_all;
    if (remove_self_interaction) {
      const double self_term = diag_sum_all / (2.0 * volume);
      energy -= qsqsum_all * self_term;
    }
    energy -= self_coeff * qsqsum_all;
    energy *= qscale;
  }

  // ── Virial ──
  if (want_virial) {
    double vr_all[6] = {0.0}, vf_all[6] = {0.0};

    // Force·r virial (diagnostic only)
    MPI_Allreduce(virial_local.data(), vr_all, 6, MPI_DOUBLE, MPI_SUM, world);

    // Fourier-space virial (primary, matches rbsog_intel & PPPM convention)
    MPI_Allreduce(fv_local.data(), vf_all, 6, MPI_DOUBLE, MPI_SUM, world);
    const double virial_scale = 0.5 * volume * qscale;
    for (int j = 0; j < 6; ++j) virial[j] = virial_scale * vf_all[j];

    // Diagnostic: compare force·r vs Fourier virial
    double max_rel_diff = 0.0;
    for (int j = 0; j < 6; ++j) {
      double denom = std::max(std::fabs(vr_all[j]), std::fabs(virial[j]));
      if (denom > 0.0) {
        double rd = std::fabs(vr_all[j] - virial[j]) / denom;
        if (rd > max_rel_diff) max_rel_diff = rd;
      }
    }
    if (comm->me == 0 && max_rel_diff > 0.0001) {
      std::string msg = fmt::format(
          "  FastSOG virial: max |force r - Fourier|/max = {:.6f}\n"
          "    force r: {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}\n"
          "    Fourier: {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}\n",
          max_rel_diff,
          vr_all[0], vr_all[1], vr_all[2], vr_all[3], vr_all[4], vr_all[5],
          virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
      utils::logmesg(lmp, msg);
    }
  }
}
