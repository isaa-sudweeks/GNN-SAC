"""Fail early when an MJX worker receives a GPU unsupported by its CUDA build."""
from importlib import metadata

import torch


def require_supported_mjx_gpu(cfg: object) -> None:
    """CUDA 13 dropped pre-SM-7.5 devices, including Volta V100 GPUs."""
    if (str(getattr(cfg, "device", "cpu")).split(":", 1)[0] != "cuda"
            or str(getattr(cfg, "mujoco_backend", "mujoco")).lower() != "mjx"):
        return
    cuda_version = torch.version.cuda
    cuda13 = cuda_version is not None and int(cuda_version.split(".", 1)[0]) >= 13
    try:
        metadata.version("jax-cuda13-plugin")
        cuda13 = True
    except metadata.PackageNotFoundError:
        pass
    if not cuda13:
        return
    device = torch.device(str(cfg.device))
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        name = torch.cuda.get_device_name(device)
        raise RuntimeError(
            f"CUDA 13 MJX requires compute capability >= 7.5; allocated GPU {name!r} "
            f"has {capability[0]}.{capability[1]}. Request a compatible GPU partition "
            "(A100, H100/H200, B200, or L40S on ORC). V100/P100 nodes cannot run "
            "this environment; upgrading ptxas or bypassing version checks will not fix it."
        )
