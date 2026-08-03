"""Report whether the active Python environment can use TensorFlow on NVIDIA."""

import sys
from pathlib import Path

# Running ``python scripts/check_gpu.py`` makes ``scripts`` the import root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_config import configure_cuda_library_path

cuda_libraries = configure_cuda_library_path()

import tensorflow as tf


print(f"TensorFlow version: {tf.__version__}")
print("Pip CUDA library directories:")
for library_dir in cuda_libraries:
    print(f"  {library_dir}")

gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow GPU devices: {gpus or 'none'}")
if not gpus:
    raise SystemExit(
        "No GPU detected. Confirm `nvidia-smi` works, then reinstall "
        '`tensorflow[and-cuda]` in this environment.'
    )
