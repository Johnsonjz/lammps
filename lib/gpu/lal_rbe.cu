// ***************************************************************************
//                                 rbe.cu
//                             -------------------
//                     Custom kernels for RBE long-range terms
//
// RBE uses single Gaussian screening exp(-k^2/(4*alpha)) with
// Monte Carlo k-point sampling (no SOG multi-scale sum).
// Kernels are simpler than RBSOG — no fac/k2 division.
// ***************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_preprocessor.h"
#endif

#define RBE_BLOCK 256

// ── Rho kernel: compute structure factor for each k-point ──
//   rho_cos[k] = sum_i q_i * cos(kx_k*x_i + ky_k*y_i + kz_k*z_i)
//   rho_sin[k] = sum_i q_i * sin(kx_k*x_i + ky_k*y_i + kz_k*z_i)
// 1 block per k-point, parallel reduction within block
__kernel void rbe_rho_kernel(const __global float *restrict x,
                             const __global float *restrict y,
                             const __global float *restrict z,
                             const __global float *restrict q,
                             const __global float *restrict kx,
                             const __global float *restrict ky,
                             const __global float *restrict kz,
                             const int nlocal, const int pcount,
                             __global float *restrict rho_cos,
                             __global float *restrict rho_sin)
{
  __local float sh_cos[RBE_BLOCK];
  __local float sh_sin[RBE_BLOCK];

  const int k = BLOCK_ID_X;
  const int tid = THREAD_ID_X;

  if (k >= pcount) return;

  const float kxv = kx[k];
  const float kyv = ky[k];
  const float kzv = kz[k];

  float sum_cos = 0.0f;
  float sum_sin = 0.0f;

  for (int i = tid; i < nlocal; i += BLOCK_SIZE_X) {
    const float moment = kxv * x[i] + kyv * y[i] + kzv * z[i];
    float s, c;
    sincosf(moment, &s, &c);
    const float qi = q[i];
    sum_cos += qi * c;
    sum_sin += qi * s;
  }

  sh_cos[tid] = sum_cos;
  sh_sin[tid] = sum_sin;
  __syncthreads();

  for (int offset = BLOCK_SIZE_X >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_cos[tid] += sh_cos[tid + offset];
      sh_sin[tid] += sh_sin[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rho_cos[k] = sh_cos[0];
    rho_sin[k] = sh_sin[0];
  }
}

// ── Force kernel: accumulate k-space forces on each atom ──
//   fx[i] = sum_k q_i * imag_k * kx_k
//   imag_k = (cos(moment) * rho_sin[k] + sin(moment) * rho_cos[k]) * midterm * fac[k]
//   moment = -(kx_k * x_i + ky_k * y_i + kz_k * z_i)
// 1 thread per atom, each loops over P k-points
__kernel void rbe_force_kernel(const __global float *restrict x,
                               const __global float *restrict y,
                               const __global float *restrict z,
                               const __global float *restrict q,
                               const __global float *restrict kx,
                               const __global float *restrict ky,
                               const __global float *restrict kz,
                               const __global float *restrict fac,
                               const __global float *restrict midterm_arr,
                               const __global float *restrict rho_cos,
                               const __global float *restrict rho_sin,
                               const int nlocal, const int pcount,
                               const float midterm,
                               __global float *restrict fx,
                               __global float *restrict fy,
                               __global float *restrict fz)
{
  const int i = BLOCK_ID_X * BLOCK_SIZE_X + THREAD_ID_X;
  if (i >= nlocal) return;

  const float xi = x[i];
  const float yi = y[i];
  const float zi = z[i];
  const float qi = q[i];

  float fxi = 0.0f;
  float fyi = 0.0f;
  float fzi = 0.0f;

  for (int k = 0; k < pcount; ++k) {
    const float kxv = kx[k];
    const float kyv = ky[k];
    const float kzv = kz[k];
    const float moment = -(kxv * xi + kyv * yi + kzv * zi);
    float s, c;
    sincosf(moment, &s, &c);

    // RBE force formula: imag = (c * rho_sin + s * rho_cos) * midterm * fac[k]
    const float imag = (c * rho_sin[k] + s * rho_cos[k]) * midterm * fac[k];

    // Force components: F += q_i * imag * (kx/k^2) component from midterm_arr
    const float qimag = qi * imag;
    fxi += qimag * midterm_arr[3*k + 0];
    fyi += qimag * midterm_arr[3*k + 1];
    fzi += qimag * midterm_arr[3*k + 2];
  }

  fx[i] = fxi;
  fy[i] = fyi;
  fz[i] = fzi;
}
