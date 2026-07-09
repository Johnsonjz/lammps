/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   FastSOG shared spline definitions (B-spline + CubeS₂ Midtown splines)
------------------------------------------------------------------------- */

#ifndef LMP_FASTSOG_SPLINE_H
#define LMP_FASTSOG_SPLINE_H

#include <array>
#include <cmath>

namespace LAMMPS_NS {

// ── B-spline order (legacy) ──
constexpr int kFastSOG_BSplineOrder = 5;
constexpr int kFastSOG_AssignOrder = kFastSOG_BSplineOrder;  // used by both .cpp files

inline void fastsog_bspline_weights_1d(const double frac,
                                        std::array<double, 5> &w) {
  w.fill(0.0);
  w[0] = 1.0 - frac;
  w[1] = frac;
  for (int k = 3; k <= 5; ++k) {
    const double inv = 1.0 / static_cast<double>(k - 1);
    w[static_cast<size_t>(k - 1)] =
        frac * w[static_cast<size_t>(k - 2)] * inv;
    for (int j = 1; j <= k - 2; ++j) {
      w[static_cast<size_t>(k - 1 - j)] =
          ((frac + static_cast<double>(j)) *
               w[static_cast<size_t>(k - 2 - j)] +
           (static_cast<double>(k - j) - frac) *
               w[static_cast<size_t>(k - 1 - j)]) *
          inv;
    }
    w[0] = (1.0 - frac) * w[0] * inv;
  }
}

// ── CubeS₂ Midtown splines ──
// Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)

// Optimal ξ values (paper Table I)
constexpr double kCubes2Xi4 = 0.5773502691896258;     // 1/√3, C² continuous
constexpr double kCubes2Xi4_alt = 0.5163977794943222; // 2/√15, general purpose
constexpr double kCubes2Xi6 = 0.6503998764035732;      // CubeS₂ 6th order optimal (SI notebook 2)

// Node counts
constexpr int kCubes2NumNodes4 = 32;
constexpr int kCubes2NumNodes6 = 88;

// ── CubeS₂ 4th-order node definition ──
struct CubeS2Node4 {
  int dx, dy, dz;  // offset from floor grid index
  int cls;         // 0 = (1,1,1) symmetry class, 1 = (2,1,1) class
  int sp_axis;     // for cls==1: which axis (0=x,1=y,2=z) has the special component
  int sp_is_neg;   // for cls==1: 1 if d=-1, 0 if d=2
};

constexpr CubeS2Node4 kCubes2Nodes4[32] = {
  // ── Class 0: offsets in {0,1}³ (8 nodes) ──
  { 0, 0, 0, 0, -1, 0}, { 0, 0, 1, 0, -1, 0}, { 0, 1, 0, 0, -1, 0}, { 0, 1, 1, 0, -1, 0},
  { 1, 0, 0, 0, -1, 0}, { 1, 0, 1, 0, -1, 0}, { 1, 1, 0, 0, -1, 0}, { 1, 1, 1, 0, -1, 0},
  // ── Class 1: one component is -1 or 2, others in {0,1} (24 nodes) ──
  // x-axis special
  {-1, 0, 0, 1, 0, 1}, {-1, 0, 1, 1, 0, 1}, {-1, 1, 0, 1, 0, 1}, {-1, 1, 1, 1, 0, 1},
  { 2, 0, 0, 1, 0, 0}, { 2, 0, 1, 1, 0, 0}, { 2, 1, 0, 1, 0, 0}, { 2, 1, 1, 1, 0, 0},
  // y-axis special
  { 0,-1, 0, 1, 1, 1}, { 0,-1, 1, 1, 1, 1}, { 1,-1, 0, 1, 1, 1}, { 1,-1, 1, 1, 1, 1},
  { 0, 2, 0, 1, 1, 0}, { 0, 2, 1, 1, 1, 0}, { 1, 2, 0, 1, 1, 0}, { 1, 2, 1, 1, 1, 0},
  // z-axis special
  { 0, 0,-1, 1, 2, 1}, { 0, 1,-1, 1, 2, 1}, { 1, 0,-1, 1, 2, 1}, { 1, 1,-1, 1, 2, 1},
  { 0, 0, 2, 1, 2, 0}, { 0, 1, 2, 1, 2, 0}, { 1, 0, 2, 1, 2, 0}, { 1, 1, 2, 1, 2, 0},
};

// Weight functions for CubeS₂ 4th order
// Eq. 15: L(θ,ξ) = -½θ³ + ½θ² - (9ξ²-2)/6·θ + ξ²/2
//          R(θ,ξ) =  ⅙θ³ + (3ξ²-1)/6·θ
inline double cubes2_L(const double theta, const double xi) {
  return -0.5 * theta * theta * theta + 0.5 * theta * theta
         - (9.0 * xi * xi - 2.0) / 6.0 * theta + 0.5 * xi * xi;
}

inline double cubes2_R(const double theta, const double xi) {
  return (1.0 / 6.0) * theta * theta * theta
         + (3.0 * xi * xi - 1.0) / 6.0 * theta;
}

// Compute CubeS₂ 4th-order weight for a single node at fractional (tx,ty,tz)
inline double cubes2_weight_4(const double tx, const double ty, const double tz,
                               const CubeS2Node4 &node, const double xi) {
  if (node.cls == 0) {
    // η = θ when d==1, η = 1−θ when d==0 (paper Eq. A8 + reflection Eq. 49).
    // The opposite mapping mirrors the charge to 1−θ within the cell.
    const double eta_x = (node.dx == 1) ? tx : (1.0 - tx);
    const double eta_y = (node.dy == 1) ? ty : (1.0 - ty);
    const double eta_z = (node.dz == 1) ? tz : (1.0 - tz);
    return cubes2_L(eta_x, xi) * eta_y * eta_z +
           cubes2_L(eta_y, xi) * eta_x * eta_z +
           cubes2_L(eta_z, xi) * eta_x * eta_y;
  } else {
    // Representative node (2,1,1): η_special=θ (d=2), η_normal=θ (d=1).
    // Reflection d=2→d=−1 sends η_special→1−θ (sp_is_neg).
    double eta_special, eta_n1, eta_n2;
    if (node.sp_axis == 0) {
      eta_special = node.sp_is_neg ? (1.0 - tx) : tx;
      eta_n1 = (node.dy == 1) ? ty : (1.0 - ty);
      eta_n2 = (node.dz == 1) ? tz : (1.0 - tz);
    } else if (node.sp_axis == 1) {
      eta_special = node.sp_is_neg ? (1.0 - ty) : ty;
      eta_n1 = (node.dx == 1) ? tx : (1.0 - tx);
      eta_n2 = (node.dz == 1) ? tz : (1.0 - tz);
    } else {
      eta_special = node.sp_is_neg ? (1.0 - tz) : tz;
      eta_n1 = (node.dx == 1) ? tx : (1.0 - tx);
      eta_n2 = (node.dy == 1) ? ty : (1.0 - ty);
    }
    return cubes2_R(eta_special, xi) * eta_n1 * eta_n2;
  }
}

// ──────────────────────────────────────────────────────────────────────
// CubeS₂ 6th-order (88-node, 5 classes) — verbatim from SI notebook 2.
// Node struct is shared with order 4; for cls==3 sp_is_neg is a 2-bit sign
// code (special axes), for cls==4 it is a 3-bit code (x,y,z). No ×27 factor.
// ──────────────────────────────────────────────────────────────────────

constexpr CubeS2Node4 kCubes2Nodes6[88] = {
  // ── Class 0 ──
  { 0, 0, 0, 0, -1, 0}, { 1, 0, 0, 0, -1, 0}, { 0, 1, 0, 0, -1, 0}, { 1, 1, 0, 0, -1, 0},
  { 0, 0, 1, 0, -1, 0}, { 1, 0, 1, 0, -1, 0}, { 0, 1, 1, 0, -1, 0}, { 1, 1, 1, 0, -1, 0},
  // ── Class 1 ──
  { 0, 0,-1, 1,  2, 1}, { 1, 0,-1, 1,  2, 1}, { 0, 1,-1, 1,  2, 1}, { 1, 1,-1, 1,  2, 1},
  { 0,-1, 0, 1,  1, 1}, { 1,-1, 0, 1,  1, 1}, {-1, 0, 0, 1,  0, 1}, { 2, 0, 0, 1,  0, 0},
  {-1, 1, 0, 1,  0, 1}, { 2, 1, 0, 1,  0, 0}, { 0, 2, 0, 1,  1, 0}, { 1, 2, 0, 1,  1, 0},
  { 0,-1, 1, 1,  1, 1}, { 1,-1, 1, 1,  1, 1}, {-1, 0, 1, 1,  0, 1}, { 2, 0, 1, 1,  0, 0},
  {-1, 1, 1, 1,  0, 1}, { 2, 1, 1, 1,  0, 0}, { 0, 2, 1, 1,  1, 0}, { 1, 2, 1, 1,  1, 0},
  { 0, 0, 2, 1,  2, 0}, { 1, 0, 2, 1,  2, 0}, { 0, 1, 2, 1,  2, 0}, { 1, 1, 2, 1,  2, 0},
  // ── Class 2 ──
  { 0, 0,-2, 2,  2, 1}, { 1, 0,-2, 2,  2, 1}, { 0, 1,-2, 2,  2, 1}, { 1, 1,-2, 2,  2, 1},
  { 0,-2, 0, 2,  1, 1}, { 1,-2, 0, 2,  1, 1}, {-2, 0, 0, 2,  0, 1}, { 3, 0, 0, 2,  0, 0},
  {-2, 1, 0, 2,  0, 1}, { 3, 1, 0, 2,  0, 0}, { 0, 3, 0, 2,  1, 0}, { 1, 3, 0, 2,  1, 0},
  { 0,-2, 1, 2,  1, 1}, { 1,-2, 1, 2,  1, 1}, {-2, 0, 1, 2,  0, 1}, { 3, 0, 1, 2,  0, 0},
  {-2, 1, 1, 2,  0, 1}, { 3, 1, 1, 2,  0, 0}, { 0, 3, 1, 2,  1, 0}, { 1, 3, 1, 2,  1, 0},
  { 0, 0, 3, 2,  2, 0}, { 1, 0, 3, 2,  2, 0}, { 0, 1, 3, 2,  2, 0}, { 1, 1, 3, 2,  2, 0},
  // ── Class 3 ──
  { 0,-1,-1, 3,  0, 3}, { 1,-1,-1, 3,  0, 3}, {-1, 0,-1, 3,  1, 3}, { 2, 0,-1, 3,  1, 2},
  {-1, 1,-1, 3,  1, 3}, { 2, 1,-1, 3,  1, 2}, { 0, 2,-1, 3,  0, 2}, { 1, 2,-1, 3,  0, 2},
  {-1,-1, 0, 3,  2, 3}, { 2,-1, 0, 3,  2, 2}, {-1, 2, 0, 3,  2, 1}, { 2, 2, 0, 3,  2, 0},
  {-1,-1, 1, 3,  2, 3}, { 2,-1, 1, 3,  2, 2}, {-1, 2, 1, 3,  2, 1}, { 2, 2, 1, 3,  2, 0},
  { 0,-1, 2, 3,  0, 1}, { 1,-1, 2, 3,  0, 1}, {-1, 0, 2, 3,  1, 1}, { 2, 0, 2, 3,  1, 0},
  {-1, 1, 2, 3,  1, 1}, { 2, 1, 2, 3,  1, 0}, { 0, 2, 2, 3,  0, 0}, { 1, 2, 2, 3,  0, 0},
  // ── Class 4 ──
  {-1,-1,-1, 4, -1, 7}, { 2,-1,-1, 4, -1, 6}, {-1, 2,-1, 4, -1, 5}, { 2, 2,-1, 4, -1, 4},
  {-1,-1, 2, 4, -1, 3}, { 2,-1, 2, 4, -1, 2}, {-1, 2, 2, 4, -1, 1}, { 2, 2, 2, 4, -1, 0},
};

// Order-6 degree-5 auxiliaries (SI notebook 2, order-6 Euclidean block).
inline double cubes2_L111_6(const double t, const double xi) {
  const double x2 = xi * xi, x4 = x2 * x2;
  return (1.0 / 12.0) * t * t * t * t * t
         - (1.0 / 6.0) * t * t * t * t
         + (10.0 * x2 - 1.0) / 12.0 * t * t * t
         - (6.0 * x2 - 1.0) / 6.0 * t * t
         + (5.0 * x4 - x2) / 4.0 * t
         - (3.0 * x4 - x2) / 6.0;
}

inline double cubes2_L311_6(const double t, const double xi) {
  const double x2 = xi * xi, x4 = x2 * x2;
  return (1.0 / 120.0) * t * t * t * t * t
         + (2.0 * x2 - 1.0) / 24.0 * t * t * t
         + (15.0 * x4 - 15.0 * x2 + 4.0) / 120.0 * t;
}

inline double cubes2_L211_6(const double t, const double xi) {
  return -0.25 * cubes2_L111_6(t, xi) - 2.5 * cubes2_L311_6(t, xi);
}

// Order-6 S is a DISTINCT degree-3 auxiliary (NOT the order-4 cubes2_L):
//   S(θ,ξ) = -½θ³ + ½θ² - (3ξ²-2)/2·θ + ξ²/2
inline double cubes2_S_6(const double t, const double xi) {
  return -0.5 * t * t * t + 0.5 * t * t
         - (3.0 * xi * xi - 2.0) / 2.0 * t + 0.5 * xi * xi;
}

// Compute CubeS₂ 6th-order weight for a single node at fractional (tx,ty,tz).
inline double cubes2_weight_6(const double tx, const double ty, const double tz,
                              const CubeS2Node4 &node, const double xi) {
  const double coords[3] = {tx, ty, tz};
  const int offs[3] = {node.dx, node.dy, node.dz};
  auto eta_pos = [&](int a) { return (offs[a] == 1) ? coords[a] : (1.0 - coords[a]); };

  if (node.cls == 0) {
    const double ex = (node.dx == 1) ? tx : (1.0 - tx);
    const double ey = (node.dy == 1) ? ty : (1.0 - ty);
    const double ez = (node.dz == 1) ? tz : (1.0 - tz);
    return cubes2_L111_6(ex, xi) * ey * ez
         + cubes2_L111_6(ey, xi) * ez * ex
         + cubes2_L111_6(ez, xi) * ex * ey
         + cubes2_S_6(ex, xi) * cubes2_S_6(ey, xi) * cubes2_S_6(ez, xi);
  } else if (node.cls == 1) {
    const int sp = node.sp_axis;
    const double es = node.sp_is_neg ? (1.0 - coords[sp]) : coords[sp];
    int n1 = (sp + 1) % 3, n2 = (sp + 2) % 3;
    if (n1 > n2) { int t = n1; n1 = n2; n2 = t; }
    return cubes2_L211_6(es, xi) * eta_pos(n1) * eta_pos(n2)
         + cubes2_R(es, xi) * cubes2_S_6(eta_pos(n1), xi) * cubes2_S_6(eta_pos(n2), xi);
  } else if (node.cls == 2) {
    const int sp = node.sp_axis;
    const double es = node.sp_is_neg ? (1.0 - coords[sp]) : coords[sp];
    int n1 = (sp + 1) % 3, n2 = (sp + 2) % 3;
    if (n1 > n2) { int t = n1; n1 = n2; n2 = t; }
    return cubes2_L311_6(es, xi) * eta_pos(n1) * eta_pos(n2);
  } else if (node.cls == 3) {
    const int nm = node.sp_axis;  // NORMAL axis for cls=3
    int s1 = (nm + 1) % 3, s2 = (nm + 2) % 3;
    if (s1 > s2) { int t = s1; s1 = s2; s2 = t; }
    const double en = eta_pos(nm);
    const double es1 = (node.sp_is_neg & 1) ? (1.0 - coords[s1]) : coords[s1];
    const double es2 = (node.sp_is_neg & 2) ? (1.0 - coords[s2]) : coords[s2];
    return cubes2_S_6(en, xi) * cubes2_R(es1, xi) * cubes2_R(es2, xi);
  } else {  // cls == 4
    const double ex = (node.sp_is_neg & 1) ? (1.0 - tx) : tx;
    const double ey = (node.sp_is_neg & 2) ? (1.0 - ty) : ty;
    const double ez = (node.sp_is_neg & 4) ? (1.0 - tz) : tz;
    return cubes2_R(ex, xi) * cubes2_R(ey, xi) * cubes2_R(ez, xi);
  }
}

}  // namespace LAMMPS_NS

#endif
