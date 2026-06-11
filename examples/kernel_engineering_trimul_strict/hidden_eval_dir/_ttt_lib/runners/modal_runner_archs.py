'''
Deploy the Modal app for a specific task.
cd gpu_mode
modal deploy runners/modal_runner_archs.py
'''
# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
import modal  # pyright: ignore[reportMissingImports]
from modal import App, Image  # pyright: ignore[reportMissingImports]

from modal_runner import modal_run_config

TASK = "mla_decode_nvidia"

if TASK == "trimul":
    ## TRIMUL Image
    app = App("discord-bot-runner")
    cuda_version = "12.8.0"
    flavor = "devel"
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    cuda_image = (
        Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
        .apt_install(
            "git",
            "gcc-13",
            "g++-13",
            "clang-18",
        )
        .pip_install(
            "ninja~=1.11",
            "wheel~=0.45",
            "requests~=2.32.4",
            "packaging~=25.0",
            "numpy~=2.3",
            "pytest",
            "PyYAML",
        )
        .pip_install(
            "torch>=2.7.0,<2.8.0",
            "torchvision~=0.22",
            "torchaudio>=2.7.0,<2.8.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        # other frameworks
        .pip_install(
            "jax[cuda12]==0.5.3",  # 0.6 want's cudnn 9.8 in conflict with torch 2.7
            "jax2torch==0.0.7",
            "tinygrad~=0.10",
        )
        # nvidia cuda packages
        .pip_install(
            "nvidia-cupynumeric~=25.3",
            "nvidia-cutlass-dsl~=4.0",
            "cuda-core[cu12]~=0.3",
            "cuda-python[all]==12.8",
            # "nvmath-python[cu12]~=0.4",
            # "numba-cuda[cu12]~=0.15",
        )
    )

elif TASK == "mla_decode_nvidia":
    app = App("discord-bot-runner-mla-decode-nvidia")
    cuda_version = "12.9.0"
    flavor = "devel"
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    cuda_image = (
        Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .apt_install(
            "git",
            "gcc",
            "g++",
            "clang",
            "lld",
            "cmake",
            "ninja-build",
            "wget",
            "curl",
            "ca-certificates",
            "python3-dev",
            "python3-pip",
            "python3-setuptools",
            "python3-wheel",
            "libpython3.12",
        )
        .pip_install(
            "pip",
            extra_options="--upgrade",
        )
        .pip_install(
            "ninja",
            "wheel",
            "packaging",
            "numpy",
            "tinygrad",
        )
        .pip_install(
            "torch==2.8.0",
            index_url="https://download.pytorch.org/whl/cu129",
        )
        # nvidia cuda packages
        .pip_install(
            "nvidia-cupynumeric~=25.3",
            "nvidia-cutlass-dsl~=4.0",
            "cuda-core[cu12]~=0.3",
            "cuda-python[all]==12.8",
        )
    )

else:
    raise ValueError(f"Invalid task: {TASK}")

cuda_image = cuda_image.add_local_python_source(
    "libkernelbot",
    "modal_runner",
    "modal_runner_archs",
)

gpus = ["A100-80GB", "H100!", "B200"]

# We intentionally raise `ModalRequeueRequest` (from modal_runner.py) on banned GPU form factors.
# This must propagate as an exception for Modal to retry/requeue the call, so we configure retries here.
_REQUEUE_RETRIES = modal.Retries(
    max_retries=5,
    initial_delay=20.0,
    backoff_coefficient=2.0,
    max_delay=60,
)

for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_cuda_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_pytorch_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
