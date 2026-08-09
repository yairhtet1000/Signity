"""Report whether the active Python environment can use TensorFlow on NVIDIA."""

import shutil
import subprocess
import sys
from pathlib import Path

# Running ``python scripts/check_gpu.py`` makes ``scripts`` the import root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_config import prepare_tensorflow_cuda

cuda_libraries = prepare_tensorflow_cuda()

import tensorflow as tf


print(f"TensorFlow version: {tf.__version__}")
print("Pip CUDA library directories:")
for library_dir in cuda_libraries:
    print(f"  {library_dir}")

gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow GPU devices: {gpus or 'none'}")
if not gpus:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi], text=True, capture_output=True, check=False
        )
        if result.returncode:
            print("\nNVIDIA driver diagnostic:")
            print(result.stderr or result.stdout)
            raise SystemExit(
                "The NVIDIA driver is installed but is not responding. Reboot Pop!_OS, "
                "then run this check again. Do not reinstall TensorFlow until `nvidia-smi` works."
            )
    raise SystemExit(
        "No GPU detected. Install or repair the Pop!_OS NVIDIA driver until "
        "`nvidia-smi` works, then reinstall `tensorflow[and-cuda]` if needed."
    )
