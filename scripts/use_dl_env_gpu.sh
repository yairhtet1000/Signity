#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate this project's virtual environment or Conda environment first." >&2
  return 1 2>/dev/null || exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
NVIDIA_SITE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia")')"

CUDA_LIB_PATHS=(
  "$NVIDIA_SITE/cublas/lib"
  "$NVIDIA_SITE/cuda_cupti/lib"
  "$NVIDIA_SITE/cuda_nvrtc/lib"
  "$NVIDIA_SITE/cuda_runtime/lib"
  "$NVIDIA_SITE/cudnn/lib"
  "$NVIDIA_SITE/cufft/lib"
  "$NVIDIA_SITE/curand/lib"
  "$NVIDIA_SITE/cusolver/lib"
  "$NVIDIA_SITE/cusparse/lib"
  "$NVIDIA_SITE/nccl/lib"
  "$NVIDIA_SITE/nvjitlink/lib"
)

CUDA_LD_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}")"
export LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

echo "Configured TensorFlow CUDA library paths for $NVIDIA_SITE"
