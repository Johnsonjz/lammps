/* ----------------------------------------------------------------------
   Scalar compatibility helpers for former AVX-512 code paths in RBSOG.
   This keeps algorithmic structure unchanged while removing oneAPI/SVML
   dependencies so the style can build on standard toolchains.
------------------------------------------------------------------------- */

#ifndef LMP_RBSOG_SIMD_COMPAT_H
#define LMP_RBSOG_SIMD_COMPAT_H

#include <cmath>
#include <cstring>

namespace LAMMPS_NS {

struct RBSOGVec {
  float lane[16];
};

inline RBSOGVec rbsog_setzero_ps()
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = 0.0f;
  return out;
}

inline RBSOGVec rbsog_set1_ps(const float value)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = value;
  return out;
}

inline RBSOGVec rbsog_load_ps(const float *ptr)
{
  RBSOGVec out;
  std::memcpy(out.lane, ptr, sizeof(out.lane));
  return out;
}

inline void rbsog_store_ps(float *ptr, const RBSOGVec &value)
{
  std::memcpy(ptr, value.lane, sizeof(value.lane));
}

inline RBSOGVec operator+(const RBSOGVec &a, const RBSOGVec &b)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = a.lane[i] + b.lane[i];
  return out;
}

inline RBSOGVec operator-(const RBSOGVec &a, const RBSOGVec &b)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = a.lane[i] - b.lane[i];
  return out;
}

inline RBSOGVec operator-(const RBSOGVec &a)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = -a.lane[i];
  return out;
}

inline RBSOGVec operator*(const RBSOGVec &a, const RBSOGVec &b)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = a.lane[i] * b.lane[i];
  return out;
}

inline RBSOGVec operator/(const RBSOGVec &a, const RBSOGVec &b)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = a.lane[i] / b.lane[i];
  return out;
}

inline RBSOGVec &operator+=(RBSOGVec &a, const RBSOGVec &b)
{
  for (int i = 0; i < 16; ++i) a.lane[i] += b.lane[i];
  return a;
}

inline RBSOGVec &operator-=(RBSOGVec &a, const RBSOGVec &b)
{
  for (int i = 0; i < 16; ++i) a.lane[i] -= b.lane[i];
  return a;
}

inline RBSOGVec &operator*=(RBSOGVec &a, const RBSOGVec &b)
{
  for (int i = 0; i < 16; ++i) a.lane[i] *= b.lane[i];
  return a;
}

inline RBSOGVec &operator/=(RBSOGVec &a, const RBSOGVec &b)
{
  for (int i = 0; i < 16; ++i) a.lane[i] /= b.lane[i];
  return a;
}

inline RBSOGVec rbsog_mul_ps(const RBSOGVec &a, const RBSOGVec &b)
{
  return a * b;
}

inline RBSOGVec rbsog_exp_ps(const RBSOGVec &x)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = std::exp(x.lane[i]);
  return out;
}

inline RBSOGVec rbsog_pow_ps(const RBSOGVec &base, const RBSOGVec &exponent)
{
  RBSOGVec out;
  for (int i = 0; i < 16; ++i) out.lane[i] = std::pow(base.lane[i], exponent.lane[i]);
  return out;
}

inline RBSOGVec rbsog_sincos_ps(RBSOGVec *cos_out, const RBSOGVec &x)
{
  RBSOGVec sin_out;
  RBSOGVec local_cos;
  for (int i = 0; i < 16; ++i) {
    sin_out.lane[i] = std::sin(x.lane[i]);
    local_cos.lane[i] = std::cos(x.lane[i]);
  }
  if (cos_out) *cos_out = local_cos;
  return sin_out;
}

inline float rbsog_reduce_add_ps(const RBSOGVec &x)
{
  float sum = 0.0f;
  for (int i = 0; i < 16; ++i) sum += x.lane[i];
  return sum;
}

}    // namespace LAMMPS_NS

#endif
