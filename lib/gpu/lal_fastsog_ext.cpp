/***************************************************************************
                     lal_fastsog_ext.cpp — FastSOG GPU library
  Provides: charge spreading + force interpolation on GPU.
  FFT and Green function remain on CPU (grid is small enough).
  Uses pinned memory for zero-copy grid transfers.
 ***************************************************************************/

#include <cstdio>
#include <cstring>
#include <string>

#include "lal_device.h"
#include "lal_precision.h"

#if defined(USE_OPENCL) || defined(USE_HIP) || defined(USE_CUDART)

extern "C" int fastsog_gpu_spread(const int, const float*, const float*,
    const float*, const float*, const int, float*, const float, const float,
    const float, const float, const float, const float, const float,
    const int, const int, const int, const float) { return -1; }
extern "C" int fastsog_gpu_interp(const int, const float*, const float*,
    const float*, const float*, const int, const float*, const float*,
    const float*, const float, const float, const float, const float,
    const float, const float, const float, const int, const int, const int,
    const float, float*, float*, float*) { return -1; }
extern "C" void fastsog_gpu_clear() {}

#else

#include "fastsog_cubin.h"

using namespace LAMMPS_AL;
namespace LAMMPS_AL { extern Device<PRECISION, ACC_PRECISION> global_device; }

namespace {

static constexpr int FASTSOG_BLOCK = 256;

struct FastSOGContext {
  UCL_Device *device = nullptr;
  UCL_Program *program = nullptr;
  UCL_Kernel k_spread, k_interp;
  bool compiled = false;

  // Atom data
  UCL_Vector<float, float> x, y, z, q;
  UCL_Vector<float, float> fx, fy, fz;
  // Grid data
  UCL_Vector<float, float> rho, gradx, grady, gradz;
};

FastSOGContext ctx;

inline int resize_vec(UCL_Vector<float,float> &v, int sz, UCL_Device &d) {
  if (sz<=0) return 0;
  if (v.numel()==0) return v.alloc(sz, d);
  if (sz > (int)v.numel()) return v.resize(sz);
  return 0;
}

int ensure_program() {
  if (global_device.gpu == nullptr) return -1;
  if (ctx.compiled && ctx.device == global_device.gpu) return 0;
  if (ctx.program) { delete ctx.program; ctx.program = nullptr; }
  ctx.device = global_device.gpu;
  ctx.program = new UCL_Program(*ctx.device);
  std::string flags = global_device.compile_string();
  int s = ctx.program->load_string(fastsog, flags.c_str(), nullptr, stderr);
  if (s != UCL_SUCCESS) {
    fprintf(stderr,"FastSOG GPU: compile failed\n");
    delete ctx.program; ctx.program=nullptr; ctx.compiled=false; return -2;
  }
  ctx.k_spread.set_function(*ctx.program, "fastsog_spread_cubes2");
  ctx.k_interp.set_function(*ctx.program, "fastsog_interp_cubes2");
  ctx.compiled = true;
  return 0;
}

int ensure_bufs(int nl, int ng) {
  int s; auto &d=*ctx.device;
  s=resize_vec(ctx.x,nl,d); if(s) return -3; s=resize_vec(ctx.y,nl,d); if(s) return -3;
  s=resize_vec(ctx.z,nl,d); if(s) return -3; s=resize_vec(ctx.q,nl,d); if(s) return -3;
  s=resize_vec(ctx.fx,nl,d); if(s) return -3; s=resize_vec(ctx.fy,nl,d); if(s) return -3;
  s=resize_vec(ctx.fz,nl,d); if(s) return -3;
  s=resize_vec(ctx.rho,ng,d);  if(s) return -3;
  s=resize_vec(ctx.gradx,ng,d); if(s) return -3;
  s=resize_vec(ctx.grady,ng,d); if(s) return -3;
  s=resize_vec(ctx.gradz,ng,d); if(s) return -3;
  return 0;
}

} // namespace

// ── Public C API ──

extern "C" int fastsog_gpu_spread(
    const int nlocal, const float *x, const float *y, const float *z,
    const float *q, const int, float *rho, const float rho_scale,
    const float bx, const float by, const float bz,
    const float dxi, const float dyi, const float dzi,
    const int nx, const int ny, const int nz, const float xi)
{
  if (nlocal<=0) return 0;
  int s=ensure_program(); if(s) return s;
  int ng=nx*ny*nz; s=ensure_bufs(nlocal,ng); if(s) return s;

  std::memcpy(ctx.x.host.begin(),x,sizeof(float)*nlocal);
  std::memcpy(ctx.y.host.begin(),y,sizeof(float)*nlocal);
  std::memcpy(ctx.z.host.begin(),z,sizeof(float)*nlocal);
  std::memcpy(ctx.q.host.begin(),q,sizeof(float)*nlocal);
  std::memset(ctx.rho.host.begin(),0,sizeof(float)*ng);

  ctx.x.update_device(nlocal,true); ctx.y.update_device(nlocal,true);
  ctx.z.update_device(nlocal,true); ctx.q.update_device(nlocal,true);
  ctx.rho.update_device(ng,true);

  int gx=(nlocal+FASTSOG_BLOCK-1)/FASTSOG_BLOCK;
  ctx.k_spread.set_size(gx,FASTSOG_BLOCK);
  ctx.k_spread.run(&ctx.x,&ctx.y,&ctx.z,&ctx.q,&nlocal,&ctx.rho,&rho_scale,
                   &bx,&by,&bz,&dxi,&dyi,&dzi,&nx,&ny,&nz,&xi);

  ctx.rho.update_host(ng,false);
  std::memcpy(rho,ctx.rho.host.begin(),sizeof(float)*ng);
  return 0;
}

extern "C" int fastsog_gpu_interp(
    const int nlocal, const float *x, const float *y, const float *z,
    const float *q, const int, const float *gx_in, const float *gy_in,
    const float *gz_in, const float qscale, const float bx, const float by,
    const float bz, const float dxi, const float dyi, const float dzi,
    const int nx, const int ny, const int nz, const float xi,
    float *fx, float *fy, float *fz)
{
  if (nlocal<=0) return 0;
  int s=ensure_program(); if(s) return s;
  int ng=nx*ny*nz; s=ensure_bufs(nlocal,ng); if(s) return s;

  std::memcpy(ctx.x.host.begin(),x,sizeof(float)*nlocal);
  std::memcpy(ctx.y.host.begin(),y,sizeof(float)*nlocal);
  std::memcpy(ctx.z.host.begin(),z,sizeof(float)*nlocal);
  std::memcpy(ctx.q.host.begin(),q,sizeof(float)*nlocal);
  std::memcpy(ctx.gradx.host.begin(),gx_in,sizeof(float)*ng);
  std::memcpy(ctx.grady.host.begin(),gy_in,sizeof(float)*ng);
  std::memcpy(ctx.gradz.host.begin(),gz_in,sizeof(float)*ng);

  ctx.x.update_device(nlocal,true); ctx.y.update_device(nlocal,true);
  ctx.z.update_device(nlocal,true); ctx.q.update_device(nlocal,true);
  ctx.gradx.update_device(ng,true); ctx.grady.update_device(ng,true);
  ctx.gradz.update_device(ng,true);

  int gx=(nlocal+FASTSOG_BLOCK-1)/FASTSOG_BLOCK;
  ctx.k_interp.set_size(gx,FASTSOG_BLOCK);
  ctx.k_interp.run(&ctx.x,&ctx.y,&ctx.z,&ctx.q,&nlocal,
                   &ctx.gradx,&ctx.grady,&ctx.gradz,&qscale,
                   &bx,&by,&bz,&dxi,&dyi,&dzi,&nx,&ny,&nz,&xi,
                   &ctx.fx,&ctx.fy,&ctx.fz);

  ctx.fx.update_host(nlocal,false); ctx.fy.update_host(nlocal,false);
  ctx.fz.update_host(nlocal,false);
  std::memcpy(fx,ctx.fx.host.begin(),sizeof(float)*nlocal);
  std::memcpy(fy,ctx.fy.host.begin(),sizeof(float)*nlocal);
  std::memcpy(fz,ctx.fz.host.begin(),sizeof(float)*nlocal);
  return 0;
}

extern "C" void fastsog_gpu_clear() {
  if (ctx.program) { delete ctx.program; ctx.program=nullptr; }
  ctx.compiled=false;
  ctx.x.clear();ctx.y.clear();ctx.z.clear();ctx.q.clear();
  ctx.fx.clear();ctx.fy.clear();ctx.fz.clear();
  ctx.rho.clear();ctx.gradx.clear();ctx.grady.clear();ctx.gradz.clear();
}

#endif
