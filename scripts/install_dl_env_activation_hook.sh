#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the target Conda environment before running this script." >&2
  exit 1
fi

ENV_PREFIX="$CONDA_PREFIX"
PYTHON_VERSION="$("$ENV_PREFIX/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_SITE="$ENV_PREFIX/lib/python$PYTHON_VERSION/site-packages/nvidia"
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/signity_cuda.sh" <<EOF
#!/usr/bin/env bash
export _SIGNITY_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_cupti/lib:$NVIDIA_SITE/cuda_nvrtc/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cudnn/lib:$NVIDIA_SITE/cufft/lib:$NVIDIA_SITE/curand/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:$NVIDIA_SITE/nccl/lib:$NVIDIA_SITE/nvjitlink/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-/tmp/matplotlib-\$USER}"
EOF

cat > "$DEACTIVATE_DIR/signity_cuda.sh" <<'EOF'
#!/usr/bin/env bash
if [ -n "${_SIGNITY_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_SIGNITY_OLD_LD_LIBRARY_PATH"
  unset _SIGNITY_OLD_LD_LIBRARY_PATH
fi
EOF

chmod +x "$ACTIVATE_DIR/signity_cuda.sh" "$DEACTIVATE_DIR/signity_cuda.sh"
echo "Installed CUDA activation hook in $ENV_PREFIX"
