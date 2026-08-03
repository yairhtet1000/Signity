"""Make TensorFlow's pip-provided NVIDIA libraries discoverable on Linux.

TensorFlow must import *after* this module is configured.  This is useful on
Pop!_OS when TensorFlow is installed with ``tensorflow[and-cuda]``: CUDA and
cuDNN are supplied by pip inside the virtual environment, while the NVIDIA
kernel driver is supplied by the operating system.
"""

import os
import site
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
    """Prepend installed pip CUDA libraries to ``LD_LIBRARY_PATH``.

    The paths are discovered from the active Python interpreter, so this works
    with either ``venv`` or Conda and does not depend on a particular Python
    version or user home directory.
    """
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

    existing = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    ordered_dirs = list(dict.fromkeys(lib_dirs + existing))
    os.environ["LD_LIBRARY_PATH"] = ":".join(ordered_dirs)
    return lib_dirs
