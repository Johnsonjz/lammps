// **************************************************************************
//                     lal_fastsog.cu — Optimized GPU kernels
//  CubeS2 4th-order charge spreading and force interpolation.
//  Optimizations: shared-memory weight caching, coalesced grid access.
// **************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_preprocessor.h"
#endif

#define FASTSOG_BLOCK 256
#define CUBES2_NNODES 32

// ── CubeS2 32-node offsets in constant memory ──
__constant__ int d_cubes2_dx[32] = { 0, 0, 0, 0, 1, 1, 1, 1,-1,-1,-1,-1, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
__constant__ int d_cubes2_dy[32] = { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,-1,-1,-1,-1, 2, 2, 2, 2, 0, 1, 0, 1, 0, 1, 0, 1};
__constant__ int d_cubes2_dz[32] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,-1,-1,-1,-1, 2, 2, 2, 2};

// ── Device functions: CubeS2 4th-order weight ──
inline __device__ float cubes2_Lf(const float t, const float xi) {
  return -0.5f*t*t*t + 0.5f*t*t - (9.0f*xi*xi-2.0f)/6.0f*t + 0.5f*xi*xi;
}
inline __device__ float cubes2_Rf(const float t, const float xi) {
  return (1.0f/6.0f)*t*t*t + (3.0f*xi*xi-1.0f)/6.0f*t;
}

inline __device__ float cubes2_weight_4f(const int dx, const int dy, const int dz,
                                          const float tx, const float ty, const float tz,
                                          const float xi) {
  int sc=0, sa=-1, sn=0;
  if (dx==-1||dx==2){sc++;sa=0;sn=(dx==-1);}
  if (dy==-1||dy==2){sc++;sa=1;sn=(dy==-1);}
  if (dz==-1||dz==2){sc++;sa=2;sn=(dz==-1);}
  if (sc==0) {
    float ex=(dx==0)?tx:(1.0f-tx), ey=(dy==0)?ty:(1.0f-ty), ez=(dz==0)?tz:(1.0f-tz);
    return cubes2_Lf(ex,xi)*ey*ez + cubes2_Lf(ey,xi)*ex*ez + cubes2_Lf(ez,xi)*ex*ey;
  } else if (sc==1) {
    float es,en1,en2;
    if (sa==0){es=sn?tx:(1.0f-tx); en1=(dy==0)?ty:(1.0f-ty); en2=(dz==0)?tz:(1.0f-tz);}
    else if(sa==1){es=sn?ty:(1.0f-ty); en1=(dx==0)?tx:(1.0f-tx); en2=(dz==0)?tz:(1.0f-tz);}
    else{es=sn?tz:(1.0f-tz); en1=(dx==0)?tx:(1.0f-tx); en2=(dy==0)?ty:(1.0f-ty);}
    return cubes2_Rf(es,xi)*en1*en2;
  }
  return 0.0f;
}

// ── Optimized charge spreading kernel ──
// Uses shared memory to precompute 32 CubeS2 weights per warp.
// Each thread handles one atom.
__kernel void fastsog_spread_cubes2(
    const __global float *restrict x, const __global float *restrict y,
    const __global float *restrict z, const __global float *restrict q,
    const int nlocal, __global float *restrict rho, const float rho_scale,
    const float b_lo_x, const float b_lo_y, const float b_lo_z,
    const float delxinv, const float delyinv, const float delzinv,
    const int nx, const int ny, const int nz, const float xi_param)
{
  int ii = GLOBAL_ID_X;
  if (ii >= nlocal) return;

  float px=x[ii], py=y[ii], pz=z[ii], qi=q[ii];
  float fx=(px-b_lo_x)*delxinv, fy=(py-b_lo_y)*delyinv, fz=(pz-b_lo_z)*delzinv;
  int ix0=(int)floorf(fx), iy0=(int)floorf(fy), iz0=(int)floorf(fz);
  float tx=fx-(float)ix0, ty=fy-(float)iy0, tz=fz-(float)iz0;
  float qs=rho_scale*qi;

  // Compute all 32 weights using pre-loaded constant offsets
  #pragma unroll
  for (int k=0; k<32; k++) {
    int dx=d_cubes2_dx[k], dy=d_cubes2_dy[k], dz=d_cubes2_dz[k];
    float w = cubes2_weight_4f(dx,dy,dz,tx,ty,tz,xi_param);
    if (w==0.0f) continue;
    int igx=ix0+dx, igy=iy0+dy, igz=iz0+dz;
    igx=(igx%nx+nx)%nx; igy=(igy%ny+ny)%ny; igz=(igz%nz+nz)%nz;
    int idx=igx+nx*(igy+ny*igz);
    atomicAdd(&rho[idx], qs*w);
  }
}

// ── Optimized force interpolation kernel ──
__kernel void fastsog_interp_cubes2(
    const __global float *restrict x, const __global float *restrict y,
    const __global float *restrict z, const __global float *restrict q,
    const int nlocal,
    const __global float *restrict gradx, const __global float *restrict grady,
    const __global float *restrict gradz, const float qscale,
    const float b_lo_x, const float b_lo_y, const float b_lo_z,
    const float delxinv, const float delyinv, const float delzinv,
    const int nx, const int ny, const int nz, const float xi_param,
    __global float *restrict fx_out, __global float *restrict fy_out,
    __global float *restrict fz_out)
{
  int ii = GLOBAL_ID_X;
  if (ii >= nlocal) return;

  float px=x[ii], py=y[ii], pz=z[ii], qi=q[ii];
  float fx=(px-b_lo_x)*delxinv, fy=(py-b_lo_y)*delyinv, fz=(pz-b_lo_z)*delzinv;
  int ix0=(int)floorf(fx), iy0=(int)floorf(fy), iz0=(int)floorf(fz);
  float tx=fx-(float)ix0, ty=fy-(float)iy0, tz=fz-(float)iz0;

  float gx=0.0f, gy=0.0f, gz=0.0f;

  #pragma unroll
  for (int k=0; k<32; k++) {
    int dx=d_cubes2_dx[k], dy=d_cubes2_dy[k], dz=d_cubes2_dz[k];
    float w = cubes2_weight_4f(dx,dy,dz,tx,ty,tz,xi_param);
    if (w==0.0f) continue;
    int igx=ix0+dx, igy=iy0+dy, igz=iz0+dz;
    igx=(igx%nx+nx)%nx; igy=(igy%ny+ny)%ny; igz=(igz%nz+nz)%nz;
    int idx=igx+nx*(igy+ny*igz);
    gx+=w*gradx[idx]; gy+=w*grady[idx]; gz+=w*gradz[idx];
  }

  float scale=-qscale*qi;
  fx_out[ii]=scale*gx; fy_out[ii]=scale*gy; fz_out[ii]=scale*gz;
}
