/***************************************************************************
                              rbsog_ext.cpp
***************************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#include "lal_device.h"
#include "lal_precision.h"

#if defined(USE_OPENCL) || defined(USE_HIP) || defined(USE_CUDART)

int rbsog_gpu_compute_rho(const int, const int, const float *, const float *, const float *,
                          const float *, const float *, const float *, const float *,
                          float *, float *)
{
  return -1;
}

int rbsog_gpu_compute_force_sampled(const int, const int, const float *, const float *,
                                    const float *, const float *, const float *, const float *,
                                    const float *, const float *, const float *, const float *,
                                    const float, float *, float *, float *)
{
  return -1;
}

int rbsog_gpu_compute_force_direct_single_rank(const int, const int, const float *,
                                               const float *, const float *, const float *,
                                               const float *, const float *, const float *,
                                               const float *, const float, float *, float *,
                                               float *, float *, float *)
{
  return -1;
}

void rbsog_gpu_clear() {}

#else

#include "rbsog_cubin.h"

using namespace LAMMPS_AL;
namespace LAMMPS_AL {
extern Device<PRECISION, ACC_PRECISION> global_device;
}

namespace {

static constexpr int RBSOG_BLOCK = 256;

struct RBSOGContext {
  UCL_Device *device = nullptr;
  UCL_Program *program = nullptr;
  UCL_Kernel k_rho;
  UCL_Kernel k_force;
  UCL_Kernel k_force_direct;
  bool compiled = false;

  UCL_Vector<float, float> x;
  UCL_Vector<float, float> y;
  UCL_Vector<float, float> z;
  UCL_Vector<float, float> q;

  UCL_Vector<float, float> kx;
  UCL_Vector<float, float> ky;
  UCL_Vector<float, float> kz;
  UCL_Vector<float, float> fac;
  UCL_Vector<float, float> rho_cos;
  UCL_Vector<float, float> rho_sin;

  UCL_Vector<float, float> fx;
  UCL_Vector<float, float> fy;
  UCL_Vector<float, float> fz;
};

RBSOGContext ctx;

int resize_vec(UCL_Vector<float, float> &vec, const int size, UCL_Device &dev)
{
  if (size <= 0) return UCL_SUCCESS;
  if (vec.numel() == 0) return vec.alloc(size, dev);
  if (size > static_cast<int>(vec.numel())) return vec.resize(size);
  return UCL_SUCCESS;
}

int ensure_program()
{
  if (global_device.gpu == nullptr) return -1;

  if (ctx.compiled && ctx.device == global_device.gpu) return 0;

  if (ctx.program) {
    delete ctx.program;
    ctx.program = nullptr;
  }

  ctx.device = global_device.gpu;
  ctx.program = new UCL_Program(*ctx.device);

  std::string flags = global_device.compile_string();
  int status = ctx.program->load_string(rbsog, flags.c_str(), nullptr, stderr);
  if (status != UCL_SUCCESS) {
    delete ctx.program;
    ctx.program = nullptr;
    ctx.compiled = false;
    return -2;
  }

  ctx.k_rho.set_function(*ctx.program, "rbsog_rho_kernel");
  ctx.k_force.set_function(*ctx.program, "rbsog_force_sampled_kernel");
  ctx.k_force_direct.set_function(*ctx.program, "rbsog_force_direct_kernel");
  ctx.compiled = true;
  return 0;
}

int ensure_buffers(const int nlocal, const int pcount)
{
  int status = UCL_SUCCESS;

  status = resize_vec(ctx.x, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.y, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.z, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.q, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;

  status = resize_vec(ctx.fx, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.fy, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.fz, nlocal, *ctx.device); if (status != UCL_SUCCESS) return -3;

  status = resize_vec(ctx.kx, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.ky, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.kz, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.fac, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;

  status = resize_vec(ctx.rho_cos, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;
  status = resize_vec(ctx.rho_sin, pcount, *ctx.device); if (status != UCL_SUCCESS) return -3;

  return 0;
}

}    // namespace

int rbsog_gpu_compute_rho(const int nlocal, const int pcount, const float *x, const float *y,
                          const float *z, const float *q, const float *kx, const float *ky,
                          const float *kz, float *rho_cos, float *rho_sin)
{
  if (nlocal <= 0 || pcount <= 0) return 0;

  int status = ensure_program();
  if (status != 0) return status;

  status = ensure_buffers(nlocal, pcount);
  if (status != 0) return status;

  std::memcpy(ctx.x.host.begin(), x, sizeof(float) * nlocal);
  std::memcpy(ctx.y.host.begin(), y, sizeof(float) * nlocal);
  std::memcpy(ctx.z.host.begin(), z, sizeof(float) * nlocal);
  std::memcpy(ctx.q.host.begin(), q, sizeof(float) * nlocal);

  std::memcpy(ctx.kx.host.begin(), kx, sizeof(float) * pcount);
  std::memcpy(ctx.ky.host.begin(), ky, sizeof(float) * pcount);
  std::memcpy(ctx.kz.host.begin(), kz, sizeof(float) * pcount);

  ctx.x.update_device(nlocal, true);
  ctx.y.update_device(nlocal, true);
  ctx.z.update_device(nlocal, true);
  ctx.q.update_device(nlocal, true);
  ctx.kx.update_device(pcount, true);
  ctx.ky.update_device(pcount, true);
  ctx.kz.update_device(pcount, true);

  ctx.k_rho.set_size(pcount, RBSOG_BLOCK);
  ctx.k_rho.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q, &ctx.kx, &ctx.ky, &ctx.kz,
                &nlocal, &pcount, &ctx.rho_cos, &ctx.rho_sin);

  ctx.rho_cos.update_host(pcount, false);
  ctx.rho_sin.update_host(pcount, false);

  std::memcpy(rho_cos, ctx.rho_cos.host.begin(), sizeof(float) * pcount);
  std::memcpy(rho_sin, ctx.rho_sin.host.begin(), sizeof(float) * pcount);

  return 0;
}

int rbsog_gpu_compute_force_sampled(const int nlocal, const int pcount, const float *x,
                                    const float *y, const float *z, const float *q,
                                    const float *kx, const float *ky, const float *kz,
                                    const float *fac, const float *rho_all_cos,
                                    const float *rho_all_sin, const float midterm, float *fx,
                                    float *fy, float *fz)
{
  if (nlocal <= 0 || pcount <= 0) return 0;

  int status = ensure_program();
  if (status != 0) return status;

  status = ensure_buffers(nlocal, pcount);
  if (status != 0) return status;

  std::memcpy(ctx.x.host.begin(), x, sizeof(float) * nlocal);
  std::memcpy(ctx.y.host.begin(), y, sizeof(float) * nlocal);
  std::memcpy(ctx.z.host.begin(), z, sizeof(float) * nlocal);
  std::memcpy(ctx.q.host.begin(), q, sizeof(float) * nlocal);

  std::memcpy(ctx.kx.host.begin(), kx, sizeof(float) * pcount);
  std::memcpy(ctx.ky.host.begin(), ky, sizeof(float) * pcount);
  std::memcpy(ctx.kz.host.begin(), kz, sizeof(float) * pcount);
  std::memcpy(ctx.fac.host.begin(), fac, sizeof(float) * pcount);
  std::memcpy(ctx.rho_cos.host.begin(), rho_all_cos, sizeof(float) * pcount);
  std::memcpy(ctx.rho_sin.host.begin(), rho_all_sin, sizeof(float) * pcount);

  ctx.x.update_device(nlocal, true);
  ctx.y.update_device(nlocal, true);
  ctx.z.update_device(nlocal, true);
  ctx.q.update_device(nlocal, true);

  ctx.kx.update_device(pcount, true);
  ctx.ky.update_device(pcount, true);
  ctx.kz.update_device(pcount, true);
  ctx.fac.update_device(pcount, true);
  ctx.rho_cos.update_device(pcount, true);
  ctx.rho_sin.update_device(pcount, true);

  const int gx = (nlocal + RBSOG_BLOCK - 1) / RBSOG_BLOCK;
  ctx.k_force.set_size(gx, RBSOG_BLOCK);
  ctx.k_force.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q, &ctx.kx, &ctx.ky, &ctx.kz,
                  &ctx.fac, &ctx.rho_cos, &ctx.rho_sin, &nlocal, &pcount,
                  &midterm, &ctx.fx, &ctx.fy, &ctx.fz);

  ctx.fx.update_host(nlocal, false);
  ctx.fy.update_host(nlocal, false);
  ctx.fz.update_host(nlocal, false);

  std::memcpy(fx, ctx.fx.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fy, ctx.fy.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fz, ctx.fz.host.begin(), sizeof(float) * nlocal);

  return 0;
}

int rbsog_gpu_compute_force_direct_single_rank(const int nlocal, const int pcount, const float *x,
                                               const float *y, const float *z, const float *q,
                                               const float *kx, const float *ky, const float *kz,
                                               const float *coeff, const float midterm,
                                               float *fx, float *fy, float *fz,
                                               float *rho_cos, float *rho_sin)
{
  if (nlocal <= 0 || pcount <= 0) return 0;

  int status = ensure_program();
  if (status != 0) return status;

  status = ensure_buffers(nlocal, pcount);
  if (status != 0) return status;

  std::memcpy(ctx.x.host.begin(), x, sizeof(float) * nlocal);
  std::memcpy(ctx.y.host.begin(), y, sizeof(float) * nlocal);
  std::memcpy(ctx.z.host.begin(), z, sizeof(float) * nlocal);
  std::memcpy(ctx.q.host.begin(), q, sizeof(float) * nlocal);

  std::memcpy(ctx.kx.host.begin(), kx, sizeof(float) * pcount);
  std::memcpy(ctx.ky.host.begin(), ky, sizeof(float) * pcount);
  std::memcpy(ctx.kz.host.begin(), kz, sizeof(float) * pcount);
  std::memcpy(ctx.fac.host.begin(), coeff, sizeof(float) * pcount);

  ctx.x.update_device(nlocal, true);
  ctx.y.update_device(nlocal, true);
  ctx.z.update_device(nlocal, true);
  ctx.q.update_device(nlocal, true);
  ctx.kx.update_device(pcount, true);
  ctx.ky.update_device(pcount, true);
  ctx.kz.update_device(pcount, true);
  ctx.fac.update_device(pcount, true);

  ctx.k_rho.set_size(pcount, RBSOG_BLOCK);
  ctx.k_rho.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q, &ctx.kx, &ctx.ky, &ctx.kz,
                &nlocal, &pcount, &ctx.rho_cos, &ctx.rho_sin);

  ctx.rho_cos.update_host(pcount, false);
  ctx.rho_sin.update_host(pcount, false);
  std::memcpy(rho_cos, ctx.rho_cos.host.begin(), sizeof(float) * pcount);
  std::memcpy(rho_sin, ctx.rho_sin.host.begin(), sizeof(float) * pcount);

  const int gx = (nlocal + RBSOG_BLOCK - 1) / RBSOG_BLOCK;
  ctx.k_force_direct.set_size(gx, RBSOG_BLOCK);
  ctx.k_force_direct.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q, &ctx.kx, &ctx.ky, &ctx.kz,
                         &ctx.fac, &ctx.rho_cos, &ctx.rho_sin, &nlocal, &pcount,
                         &midterm, &ctx.fx, &ctx.fy, &ctx.fz);

  ctx.fx.update_host(nlocal, false);
  ctx.fy.update_host(nlocal, false);
  ctx.fz.update_host(nlocal, false);

  std::memcpy(fx, ctx.fx.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fy, ctx.fy.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fz, ctx.fz.host.begin(), sizeof(float) * nlocal);

  return 0;
}

void rbsog_gpu_clear()
{
  if (ctx.program) {
    delete ctx.program;
    ctx.program = nullptr;
  }
  ctx.compiled = false;

  ctx.x.clear();
  ctx.y.clear();
  ctx.z.clear();
  ctx.q.clear();

  ctx.kx.clear();
  ctx.ky.clear();
  ctx.kz.clear();
  ctx.fac.clear();
  ctx.rho_cos.clear();
  ctx.rho_sin.clear();

  ctx.fx.clear();
  ctx.fy.clear();
  ctx.fz.clear();
}

#endif
