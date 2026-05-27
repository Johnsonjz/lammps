// ***************************************************************************
//                                 rbsog.cu
//                             -------------------
//                     Custom kernels for RBSOG long-range terms
// ***************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_preprocessor.h"
#endif

#define RBSOG_BLOCK 256

__kernel void rbsog_rho_kernel(const __global float *restrict x,
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
  __local float sh_cos[RBSOG_BLOCK];
  __local float sh_sin[RBSOG_BLOCK];

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

__kernel void rbsog_force_sampled_kernel(const __global float *restrict x,
                                         const __global float *restrict y,
                                         const __global float *restrict z,
                                         const __global float *restrict q,
                                         const __global float *restrict kx,
                                         const __global float *restrict ky,
                                         const __global float *restrict kz,
                                         const __global float *restrict fac,
                                         const __global float *restrict rho_cos,
                                         const __global float *restrict rho_sin,
                                         const int nlocal, const int pcount,
                                         const float midterm,
                                         __global float *restrict fx,
                                         __global float *restrict fy,
                                         __global float *restrict fz)
{
  const int i = GLOBAL_ID_X;
  if (i >= nlocal) return;

  const float xi = x[i];
  const float yi = y[i];
  const float zi = z[i];
  const float qi = q[i];

  float fxi = 0.0f;
  float fyi = 0.0f;
  float fzi = 0.0f;

  for (int k = 0; k < pcount; k++) {
    const float kxv = kx[k];
    const float kyv = ky[k];
    const float kzv = kz[k];
    const float k2 = kxv * kxv + kyv * kyv + kzv * kzv;
    if (k2 == 0.0f) continue;

    const float moment = -(kxv * xi + kyv * yi + kzv * zi);
    float s, c;
    sincosf(moment, &s, &c);

    const float imag = (c * rho_sin[k] + s * rho_cos[k]) * (midterm * fac[k]) / k2;
    const float pref = qi * imag;

    fxi += pref * kxv;
    fyi += pref * kyv;
    fzi += pref * kzv;
  }

  fx[i] = fxi;
  fy[i] = fyi;
  fz[i] = fzi;
}

__kernel void rbsog_force_direct_kernel(const __global float *restrict x,
                                        const __global float *restrict y,
                                        const __global float *restrict z,
                                        const __global float *restrict q,
                                        const __global float *restrict kx,
                                        const __global float *restrict ky,
                                        const __global float *restrict kz,
                                        const __global float *restrict coeff,
                                        const __global float *restrict rho_cos,
                                        const __global float *restrict rho_sin,
                                        const int nlocal, const int pcount,
                                        const float midterm,
                                        __global float *restrict fx,
                                        __global float *restrict fy,
                                        __global float *restrict fz)
{
  const int i = GLOBAL_ID_X;
  if (i >= nlocal) return;

  const float xi = x[i];
  const float yi = y[i];
  const float zi = z[i];
  const float qi = q[i];

  float fxi = 0.0f;
  float fyi = 0.0f;
  float fzi = 0.0f;

  for (int k = 0; k < pcount; k++) {
    const float kxv = kx[k];
    const float kyv = ky[k];
    const float kzv = kz[k];

    const float moment = -(kxv * xi + kyv * yi + kzv * zi);
    float s, c;
    sincosf(moment, &s, &c);

    const float imag = (c * rho_sin[k] + s * rho_cos[k]) * (midterm * coeff[k]);
    const float pref = qi * imag;

    fxi += pref * kxv;
    fyi += pref * kyv;
    fzi += pref * kzv;
  }

  fx[i] = fxi;
  fy[i] = fyi;
  fz[i] = fzi;
}
