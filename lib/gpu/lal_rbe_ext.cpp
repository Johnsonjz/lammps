/***************************************************************************
                              rbe_ext.cpp
    C bridge for RBE GPU acceleration — follows RBSOG GPU pattern
***************************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#include "lal_device.h"
#include "lal_precision.h"

#if defined(USE_OPENCL) || defined(USE_HIP) || defined(USE_CUDART)

int rbe_gpu_compute_rho(const int, const int, const float *, const float *, const float *,
                        const float *, const float *, const float *, const float *,
                        float *, float *)
{ return -1; }

int rbe_gpu_compute_force(const int, const int, const float *, const float *, const float *,
                          const float *, const float *, const float *, const float *,
                          const float *, const float *, const float *, const float *,
                          const float, float *, float *, float *)
{ return -1; }

void rbe_gpu_clear() {}

#else

#include "rbe_cubin.h"

using namespace LAMMPS_AL;
namespace LAMMPS_AL {
extern Device<PRECISION, ACC_PRECISION> global_device;
}

namespace {

static constexpr int RBE_BLOCK = 256;

struct RBEContext {
  UCL_Device *device = nullptr;
  UCL_Program *program = nullptr;
  UCL_Kernel k_rho;
  UCL_Kernel k_force;
  bool compiled = false;

  UCL_Vector<float, float> x, y, z, q;
  UCL_Vector<float, float> kx, ky, kz, fac;
  UCL_Vector<float, float> midterm_arr;
  UCL_Vector<float, float> rho_cos, rho_sin;
  UCL_Vector<float, float> fx, fy, fz;
};

RBEContext ctx;

int resize_vec(UCL_Vector<float, float> &vec, const int size, UCL_Device &dev)
{
  if (size <= 0) return UCL_SUCCESS;
  if (vec.cols() > 0 && size > static_cast<int>(vec.cols())) return vec.resize(size);
  if (vec.cols() == 0) return vec.alloc(size, dev, UCL_READ_WRITE);
  return UCL_SUCCESS;
}

int ensure_program()
{
  if (global_device.gpu == nullptr) return -1;
  if (ctx.compiled && ctx.device == global_device.gpu) return 0;

  if (ctx.program) { delete ctx.program; ctx.program = nullptr; }
  ctx.device = global_device.gpu;
  ctx.program = new UCL_Program(*ctx.device);

  std::string flags = global_device.compile_string();
  int status = ctx.program->load_string(rbe, flags.c_str(), nullptr, stderr);
  if (status != UCL_SUCCESS) {
    delete ctx.program;
    ctx.program = nullptr;
    ctx.compiled = false;
    return -2;
  }

  ctx.k_rho.set_function(*ctx.program, "rbe_rho_kernel");
  ctx.k_force.set_function(*ctx.program, "rbe_force_kernel");
  ctx.compiled = true;
  return 0;
}

int ensure_buffers(const int nlocal, const int pcount)
{
  if (ensure_program() < 0) return -1;
  UCL_Device &dev = *global_device.gpu;

  int rc = UCL_SUCCESS;
  rc |= resize_vec(ctx.x, nlocal, dev);
  rc |= resize_vec(ctx.y, nlocal, dev);
  rc |= resize_vec(ctx.z, nlocal, dev);
  rc |= resize_vec(ctx.q, nlocal, dev);
  rc |= resize_vec(ctx.kx, pcount, dev);
  rc |= resize_vec(ctx.ky, pcount, dev);
  rc |= resize_vec(ctx.kz, pcount, dev);
  rc |= resize_vec(ctx.fac, pcount, dev);
  rc |= resize_vec(ctx.midterm_arr, 3 * pcount, dev);
  rc |= resize_vec(ctx.rho_cos, pcount, dev);
  rc |= resize_vec(ctx.rho_sin, pcount, dev);
  rc |= resize_vec(ctx.fx, nlocal, dev);
  rc |= resize_vec(ctx.fy, nlocal, dev);
  rc |= resize_vec(ctx.fz, nlocal, dev);
  return rc;
}

}  // anonymous namespace

int rbe_gpu_compute_rho(const int nlocal, const int pcount,
                        const float *x, const float *y, const float *z,
                        const float *q, const float *kx, const float *ky, const float *kz,
                        float *rho_cos, float *rho_sin)
{
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

  ctx.k_rho.set_size(pcount, RBE_BLOCK);
  ctx.k_rho.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q,
                &ctx.kx, &ctx.ky, &ctx.kz,
                &nlocal, &pcount,
                &ctx.rho_cos, &ctx.rho_sin);

  ctx.rho_cos.update_host(pcount, false);
  ctx.rho_sin.update_host(pcount, false);
  std::memcpy(rho_cos, ctx.rho_cos.host.begin(), sizeof(float) * pcount);
  std::memcpy(rho_sin, ctx.rho_sin.host.begin(), sizeof(float) * pcount);
  return 0;
}

int rbe_gpu_compute_force(const int nlocal, const int pcount,
                          const float *x, const float *y, const float *z,
                          const float *q, const float *kx, const float *ky, const float *kz,
                          const float *fac, const float *midterm_arr,
                          const float *rho_all_cos, const float *rho_all_sin,
                          const float midterm_scalar,
                          float *fx, float *fy, float *fz)
{
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
  std::memcpy(ctx.midterm_arr.host.begin(), midterm_arr, sizeof(float) * 3 * pcount);
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
  ctx.midterm_arr.update_device(3 * pcount, true);
  ctx.rho_cos.update_device(pcount, true);
  ctx.rho_sin.update_device(pcount, true);

  const int block = RBE_BLOCK;
  const int grid = (nlocal + block - 1) / block;
  ctx.k_force.set_size(grid, block);
  ctx.k_force.run(&ctx.x, &ctx.y, &ctx.z, &ctx.q,
                  &ctx.kx, &ctx.ky, &ctx.kz,
                  &ctx.fac, &ctx.midterm_arr,
                  &ctx.rho_cos, &ctx.rho_sin,
                  &nlocal, &pcount, &midterm_scalar,
                  &ctx.fx, &ctx.fy, &ctx.fz);

  ctx.fx.update_host(nlocal, false);
  ctx.fy.update_host(nlocal, false);
  ctx.fz.update_host(nlocal, false);
  std::memcpy(fx, ctx.fx.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fy, ctx.fy.host.begin(), sizeof(float) * nlocal);
  std::memcpy(fz, ctx.fz.host.begin(), sizeof(float) * nlocal);
  return 0;
}

#endif
