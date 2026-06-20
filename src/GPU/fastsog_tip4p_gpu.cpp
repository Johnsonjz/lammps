/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   FastSOG/TIP4P GPU acceleration — M-site aware charge spreading and
   force interpolation on GPU.
------------------------------------------------------------------------- */

#include "fastsog_tip4p_gpu.h"
#include "fastsog_spline.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "gpu_extra.h"
#include "math_const.h"
#include "modify.h"

#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace MathConst;

// ── External GPU functions ──
extern "C" int fastsog_gpu_spread(
    const int nlocal, const float *x, const float *y, const float *z,
    const float *q, const int nlocal_pad,
    float *rho, const float rho_scale,
    const float b_lo_x, const float b_lo_y, const float b_lo_z,
    const float delxinv, const float delyinv, const float delzinv,
    const int nx, const int ny, const int nz, const float xi_param);

extern "C" int fastsog_gpu_interp(
    const int nlocal, const float *x, const float *y, const float *z,
    const float *q, const int nlocal_pad,
    const float *gradx, const float *grady, const float *gradz,
    const float qscale, const float b_lo_x, const float b_lo_y,
    const float b_lo_z, const float delxinv, const float delyinv,
    const float delzinv, const int nx, const int ny, const int nz,
    const float xi_param, float *fx, float *fy, float *fz);

extern "C" void fastsog_gpu_clear();

/* ---------------------------------------------------------------------- */

FastSOGTIP4PGPU::FastSOGTIP4PGPU(LAMMPS *lmp) : FastSOGTIP4P(lmp)
{
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

void FastSOGTIP4PGPU::init()
{
  FastSOGTIP4P::init();
}

/* ----------------------------------------------------------------------
   GPU-accelerated compute with TIP4P M-site handling.
   M-site positions are computed on CPU, then passed to GPU kernels.
------------------------------------------------------------------------- */

void FastSOGTIP4PGPU::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (atom->natoms != natoms_original) {
    qsum_qsq(); natoms_original = atom->natoms;
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

  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;

  // ── Compute M-site positions for O atoms (CPU) ──
  std::vector<float> h_x(nlocal), h_y(nlocal), h_z(nlocal), h_q(nlocal);
  std::vector<int> iH1_list(nlocal, -1), iH2_list(nlocal, -1);
  std::vector<double> xM_list(nlocal * 3, 0.0);

  for (int i = 0; i < nlocal; ++i) {
    if (type[i] == typeO) {
      double xM[3]; int iH1, iH2;
      find_M(i, iH1, iH2, xM);
      h_x[i] = static_cast<float>(xM[0]);
      h_y[i] = static_cast<float>(xM[1]);
      h_z[i] = static_cast<float>(xM[2]);
      iH1_list[i] = iH1; iH2_list[i] = iH2;
      xM_list[i*3+0] = xM[0]; xM_list[i*3+1] = xM[1]; xM_list[i*3+2] = xM[2];
    } else {
      h_x[i] = static_cast<float>(x[i][0]);
      h_y[i] = static_cast<float>(x[i][1]);
      h_z[i] = static_cast<float>(x[i][2]);
    }
    h_q[i] = static_cast<float>(q[i]);
  }

  const double volume = mesh_lx * mesh_ly * mesh_lz;
  const float rho_scale = static_cast<float>(
      static_cast<double>(mesh_nx * mesh_ny * mesh_nz) / volume);
  const float b_lo_x = static_cast<float>(domain->boxlo[0]);
  const float b_lo_y = static_cast<float>(domain->boxlo[1]);
  const float b_lo_z = static_cast<float>(domain->boxlo[2]);
  const float delxinv = static_cast<float>(static_cast<double>(mesh_nx) / mesh_lx);
  const float delyinv = static_cast<float>(static_cast<double>(mesh_ny) / mesh_ly);
  const float delzinv = static_cast<float>(static_cast<double>(mesh_nz) / mesh_lz);
  const float xi_param = static_cast<float>((spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6);

  const size_t ngrid = mesh_rho.size();

  // ── GPU charge spreading (M-site positions already computed) ──
  std::vector<float> h_rho(ngrid, 0.0f);
  int gpu_status = fastsog_gpu_spread(nlocal, h_x.data(), h_y.data(), h_z.data(),
                                       h_q.data(), nlocal, h_rho.data(), rho_scale,
                                       b_lo_x, b_lo_y, b_lo_z,
                                       delxinv, delyinv, delzinv,
                                       mesh_nx, mesh_ny, mesh_nz, xi_param);

  if (gpu_status != 0) {
    FastSOGTIP4P::compute(eflag, vflag);
    return;
  }

  // Copy to mesh_rho
  std::fill(mesh_rho.begin(), mesh_rho.end(), 0.0);
  for (size_t i = 0; i < ngrid; ++i)
    mesh_rho[i] = static_cast<FFT_SCALAR>(h_rho[i]);

  // ── FFT forward → Green multiply → FFT backward (CPU) ──
  std::fill(mesh_fft_work.begin(), mesh_fft_work.end(), 0.0);
  for (size_t idx = 0; idx < ngrid; ++idx)
    mesh_fft_work[2*idx] = mesh_rho[idx];
  mesh_fft->compute(mesh_fft_work.data(), mesh_fft_work.data(), FFT3d::FORWARD);

  const double scaleinv = 1.0 / static_cast<double>(ngrid);
  const double s2 = scaleinv * scaleinv;
  const double tpox = MY_2PI / mesh_lx, tpoy = MY_2PI / mesh_ly, tpoz = MY_2PI / mesh_lz;
  double energy_local = 0.0, diag_sum_local = 0.0;
  std::array<double,6> fv_local = {0.0};

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2*iz/mesh_nz);
    const double kz = tpoz * kz_mode;
    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2*iy/mesh_ny);
      const double ky = tpoy * ky_mode;
      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2*ix/mesh_nx);
        const double kx = tpox * kx_mode;
        const size_t idx = mesh_index(ix, iy, iz);
        const double ge = mesh_green_energy[idx], gf = mesh_green_force[idx];
        if (ge == 0.0 && gf == 0.0) {
          mesh_gradx[2*idx]=mesh_gradx[2*idx+1]=0.0;
          mesh_grady[2*idx]=mesh_grady[2*idx+1]=0.0;
          mesh_gradz[2*idx]=mesh_gradz[2*idx+1]=0.0;
          continue;
        }
        diag_sum_local += mesh_green_self[idx];
        const double rho_re = mesh_fft_work[2*idx], rho_im = mesh_fft_work[2*idx+1];
        if (want_energy) energy_local += s2 * ge * (rho_re*rho_re + rho_im*rho_im);
        if (want_virial) {
          double rho2=rho_re*rho_re+rho_im*rho_im;
          double gv=mesh_green_virial[idx];
          fv_local[0]+=s2*rho2*(ge - gv*kx*kx);
          fv_local[1]+=s2*rho2*(ge - gv*ky*ky);
          fv_local[2]+=s2*rho2*(ge - gv*kz*kz);
          fv_local[3]+=s2*rho2*(-gv*kx*ky);
          fv_local[4]+=s2*rho2*(-gv*kx*kz);
          fv_local[5]+=s2*rho2*(-gv*ky*kz);
        }
        const double vk_re = scaleinv * gf * rho_re, vk_im = scaleinv * gf * rho_im;
        mesh_gradx[2*idx] = static_cast<FFT_SCALAR>(-kx*vk_im);
        mesh_gradx[2*idx+1] = static_cast<FFT_SCALAR>(kx*vk_re);
        mesh_grady[2*idx] = static_cast<FFT_SCALAR>(-ky*vk_im);
        mesh_grady[2*idx+1] = static_cast<FFT_SCALAR>(ky*vk_re);
        mesh_gradz[2*idx] = static_cast<FFT_SCALAR>(-kz*vk_im);
        mesh_gradz[2*idx+1] = static_cast<FFT_SCALAR>(kz*vk_re);
      }
    }
  }

  mesh_fft->compute(mesh_gradx.data(), mesh_gradx.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(), mesh_grady.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(), mesh_gradz.data(), FFT3d::BACKWARD);

  // ── GPU force interpolation (M-site positions) ──
  std::vector<float> h_gradx(ngrid), h_grady(ngrid), h_gradz(ngrid);
  for (size_t i = 0; i < ngrid; ++i) {
    h_gradx[i] = static_cast<float>(mesh_gradx[2*i]);
    h_grady[i] = static_cast<float>(mesh_grady[2*i]);
    h_gradz[i] = static_cast<float>(mesh_gradz[2*i]);
  }

  std::vector<float> h_fx(nlocal), h_fy(nlocal), h_fz(nlocal);
  const float qscale_f = static_cast<float>(force->qqrd2e * scale);

  // NOTE: Pass M-site positions for O atoms, real positions for H atoms
  // This was already computed above in h_x, h_y, h_z
  gpu_status = fastsog_gpu_interp(nlocal, h_x.data(), h_y.data(), h_z.data(),
                                   h_q.data(), nlocal,
                                   h_gradx.data(), h_grady.data(), h_gradz.data(),
                                   qscale_f, b_lo_x, b_lo_y, b_lo_z,
                                   delxinv, delyinv, delzinv,
                                   mesh_nx, mesh_ny, mesh_nz, xi_param,
                                   h_fx.data(), h_fy.data(), h_fz.data());

  if (gpu_status != 0) {
    FastSOGTIP4P::compute(eflag, vflag);
    return;
  }

  // ── Accumulate forces with TIP4P redistribution ──
  std::array<double, 6> virial_local = {0.0};
  for (int i = 0; i < nlocal; ++i) {
    double fxs = static_cast<double>(h_fx[i]);
    double fys = static_cast<double>(h_fy[i]);
    double fzs = static_cast<double>(h_fz[i]);

    if (type[i] == typeO) {
      int iH1 = iH1_list[i], iH2 = iH2_list[i];
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
        double xM[3] = {xM_list[i*3+0], xM_list[i*3+1], xM_list[i*3+2]};
        virial_local[0] += xM[0]*fxs; virial_local[1] += xM[1]*fys;
        virial_local[2] += xM[2]*fzs; virial_local[3] += xM[0]*fys;
        virial_local[4] += xM[0]*fzs; virial_local[5] += xM[1]*fzs;
      }
    } else {
      atom->f[i][0] += fxs;
      atom->f[i][1] += fys;
      atom->f[i][2] += fzs;
      if (want_virial) {
        virial_local[0] += x[i][0]*fxs; virial_local[1] += x[i][1]*fys;
        virial_local[2] += x[i][2]*fzs; virial_local[3] += x[i][0]*fys;
        virial_local[4] += x[i][0]*fzs; virial_local[5] += x[i][1]*fzs;
      }
    }
  }

  // ── Energy ──
  if (want_energy) {
    double ea=0, da=0;
    MPI_Allreduce(&energy_local, &ea, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&diag_sum_local, &da, 1, MPI_DOUBLE, MPI_SUM, world);
    double qsl=0;
    for (int i=0;i<nlocal;++i) qsl+=q[i]*q[i];
    double qsa; MPI_Allreduce(&qsl,&qsa,1,MPI_DOUBLE,MPI_SUM,world);
    energy = 0.5*volume*ea;
    if (remove_self_interaction) energy -= qsa*da/(2.0*volume);
    energy -= self_coeff*qsa;
    energy *= force->qqrd2e*scale;
  }
  if (want_virial) {
    double va[6];
    MPI_Allreduce(virial_local.data(),va,6,MPI_DOUBLE,MPI_SUM,world);
    // force·r retained for diagnostic (va), primary virial is Fourier
    double vf_all[6];
    MPI_Allreduce(fv_local.data(),vf_all,6,MPI_DOUBLE,MPI_SUM,world);
    double vs=0.5*volume*(force->qqrd2e*scale);
    for (int j=0;j<6;++j) virial[j]=vs*vf_all[j];
  }
}
