#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${CONDA_PREFIX:-/home/mango/.conda/envs/dl_env}"
NVIDIA_SITE="$ENV_PREFIX/lib/python3.10/site-packages/nvidia"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"

echo "Configured TensorFlow CUDA library paths for $ENV_PREFIX"
