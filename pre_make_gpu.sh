#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build_gpu"
MPI_ENV_NAME="${MPI_ENV_NAME:-lmp_mpi}"
CMAKE_ENV_NAME="${CMAKE_ENV_NAME:-lmp}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# Optional HPC environment module
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load oneapi >/dev/null 2>&1 || true
fi

CC_BIN="${CC:-}"
CXX_BIN="${CXX:-}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
CONDA_BASE=""
CONDA_BIN="$(type -P conda || true)"

if [[ -z "${CONDA_BIN}" && -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
fi

if [[ -z "${CONDA_BIN}" ]]; then
  for CONDA_CANDIDATE in \
    "/opt/anaconda3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "${CONDA_CANDIDATE}" ]]; then
      CONDA_BIN="${CONDA_CANDIDATE}"
      break
    fi
  done
fi

if [[ -n "${CONDA_BIN}" ]]; then
  CONDA_BASE="$("${CONDA_BIN}" info --base 2>/dev/null || true)"
fi

# 1) user-provided CC/CXX  2) current PATH  3) fallback conda env (default: lmp_mpi)
if [[ -z "${CC_BIN}" ]]; then
  CC_BIN="$(command -v mpicc || true)"
fi
if [[ -z "${CXX_BIN}" ]]; then
  CXX_BIN="$(command -v mpicxx || true)"
fi

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  for CONDA_ENV_NAME in "${MPI_ENV_NAME}" "${CMAKE_ENV_NAME}"; do
    for CONDA_ENV_BIN in \
      "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin" \
      "${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin"; do
      if [[ -z "${CC_BIN}" && -x "${CONDA_ENV_BIN}/mpicc" ]]; then
        CC_BIN="${CONDA_ENV_BIN}/mpicc"
      fi
      if [[ -z "${CXX_BIN}" && -x "${CONDA_ENV_BIN}/mpicxx" ]]; then
        CXX_BIN="${CONDA_ENV_BIN}/mpicxx"
      fi
    done
  done
fi

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  echo "[ERROR] mpicc/mpicxx not found."
  echo "        Try: conda activate ${MPI_ENV_NAME}"
  echo "        Or set: CC=/path/to/mpicc CXX=/path/to/mpicxx"
  exit 1
fi

if [[ -z "${CMAKE_BIN}" ]]; then
  for CONDA_CMAKE_BIN in \
    "${HOME}/.conda/envs/${CMAKE_ENV_NAME}/bin/cmake" \
    "${CONDA_BASE}/envs/${CMAKE_ENV_NAME}/bin/cmake"; do
    if [[ -x "${CONDA_CMAKE_BIN}" ]]; then
      CMAKE_BIN="${CONDA_CMAKE_BIN}"
      break
    fi
  done
fi

if [[ -z "${CMAKE_BIN}" ]]; then
  echo "[ERROR] cmake not found."
  echo "        Try: conda activate ${CMAKE_ENV_NAME}"
  echo "        Or set: CMAKE_BIN=/path/to/cmake"
  exit 1
fi

GPU_ARCH_VAL="${GPU_ARCH_VAL:-sm_90}"
GPU_PREC_VAL="${GPU_PREC_VAL:-mixed}"

rm -rf "${BUILD_DIR}"

"${CMAKE_BIN}" -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
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

"${CMAKE_BIN}" --build "${BUILD_DIR}" -j "$(nproc)"

echo "[OK] Build complete: ${BUILD_DIR}/lmp"
