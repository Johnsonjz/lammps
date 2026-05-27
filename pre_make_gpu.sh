#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build_gpu"
MPI_ENV_NAME="${MPI_ENV_NAME:-lmp_mpi}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# Optional HPC environment module
if command -v module >/dev/null 2>&1; then
  module load oneapi || true
fi

CC_BIN="${CC:-}"
CXX_BIN="${CXX:-}"

# 1) user-provided CC/CXX  2) current PATH  3) fallback conda env (default: lmp_mpi)
if [[ -z "${CC_BIN}" ]]; then
  CC_BIN="$(command -v mpicc || true)"
fi
if [[ -z "${CXX_BIN}" ]]; then
  CXX_BIN="$(command -v mpicxx || true)"
fi

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" ]]; then
      CONDA_MPI_BIN="${CONDA_BASE}/envs/${MPI_ENV_NAME}/bin"
      if [[ -z "${CC_BIN}" && -x "${CONDA_MPI_BIN}/mpicc" ]]; then
        CC_BIN="${CONDA_MPI_BIN}/mpicc"
      fi
      if [[ -z "${CXX_BIN}" && -x "${CONDA_MPI_BIN}/mpicxx" ]]; then
        CXX_BIN="${CONDA_MPI_BIN}/mpicxx"
      fi
    fi
  fi
fi

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  echo "[ERROR] mpicc/mpicxx not found."
  echo "        Try: conda activate ${MPI_ENV_NAME}"
  echo "        Or set: CC=/path/to/mpicc CXX=/path/to/mpicxx"
  exit 1
fi

GPU_ARCH_VAL="${GPU_ARCH_VAL:-sm_90}"
GPU_PREC_VAL="${GPU_PREC_VAL:-mixed}"

rm -rf "${BUILD_DIR}"

cmake -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC_BIN}" \
  -D CMAKE_CXX_COMPILER="${CXX_BIN}" \
  -D BUILD_MPI=on \
  -D BUILD_OMP=off \
  -D PKG_GPU=on \
  -D PKG_KSPACE=on \
  -D PKG_MOLECULE=on \
  -D PKG_MANYBODY=on \
  -D PKG_REPLICA=on \
  -D PKG_RIGID=on \
  -D PKG_DIPOLE=on \
  -D PKG_EXTRA-DUMP=on \
  -D GPU_API=cuda \
  -D GPU_ARCH="${GPU_ARCH_VAL}" \
  -D GPU_PREC="${GPU_PREC_VAL}"

if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo "[OK] Configure complete (SKIP_BUILD=1): ${BUILD_DIR}"
  exit 0
fi

cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo "[OK] Build complete: ${BUILD_DIR}/lmp"
