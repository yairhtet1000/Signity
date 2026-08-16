"""Make TensorFlow's pip-provided NVIDIA libraries discoverable on Linux.

TensorFlow must import *after* this module is configured. This is useful on
Pop!_OS when TensorFlow is installed with ``tensorflow[and-cuda]``: CUDA and
cuDNN are supplied by pip inside the virtual environment, while the NVIDIA
kernel driver is supplied by the operating system.
"""

import os
import site
import sys
import sysconfig
from pathlib import Path


NVIDIA_PACKAGES = (
    "cublas",
    "cuda_cupti",
    "cuda_nvrtc",
    "cuda_runtime",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvjitlink",
)


def configure_cuda_library_path():
    """Prepend installed pip CUDA libraries to ``LD_LIBRARY_PATH``."""
    candidate_roots = {
        Path(sysconfig.get_paths()["purelib"]) / "nvidia",
        Path(site.getusersitepackages()) / "nvidia",
    }
    for site_dir in site.getsitepackages():
        candidate_roots.add(Path(site_dir) / "nvidia")

    lib_dirs = [
        str(root / package / "lib")
        for root in candidate_roots
        for package in NVIDIA_PACKAGES
        if (root / package / "lib").is_dir()
    ]
    if not lib_dirs:
        return []

    existing = [
        part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part
    ]
    ordered_dirs = list(dict.fromkeys(lib_dirs + existing))
    os.environ["LD_LIBRARY_PATH"] = ":".join(ordered_dirs)
    return lib_dirs


def prepare_tensorflow_cuda():
    """Set CUDA paths and restart Python once so the dynamic linker sees them.

    Updating ``os.environ`` after Python starts is too late for some Linux
    dynamic-loader lookups. Re-executing the current command starts Python with
    the pip-installed NVIDIA libraries already in ``LD_LIBRARY_PATH``.
    """
    lib_dirs = configure_cuda_library_path()
    if not lib_dirs or os.environ.get("_SIGNITY_CUDA_REEXEC") == "1":
        _configure_gpu_memory_growth()
        return lib_dirs

    os.environ["_SIGNITY_CUDA_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def _configure_gpu_memory_growth():
    """Enable memory growth on all visible GPUs after TensorFlow is importable."""
    try:
        import tensorflow as tf
    except ImportError:
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow GPU devices: {gpus}")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print("TensorFlow GPU devices: none. Training will use CPU.")
