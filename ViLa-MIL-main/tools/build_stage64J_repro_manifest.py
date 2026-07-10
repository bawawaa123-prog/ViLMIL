#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def run_text(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    output = proc.stdout if proc.stdout else proc.stderr
    return proc.returncode, output


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"import_failed: {exc!r}"


def scipy_numpy_compat_note() -> dict[str, object]:
    result: dict[str, object] = {"has_warning": False, "warnings": []}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import numpy as np
            import scipy  # noqa: F401
        result["has_warning"] = bool(caught)
        result["warnings"] = [str(item.message) for item in caught]
        numpy_version = Version(np.__version__)
        requires = importlib.metadata.requires("scipy") or []
        numpy_requirements = []
        satisfies = []
        for raw in requires:
            req = Requirement(raw)
            if req.name.lower() != "numpy":
                continue
            numpy_requirements.append(str(req.specifier))
            satisfies.append(numpy_version in req.specifier)
        result["installed_numpy_version"] = str(numpy_version)
        result["scipy_numpy_requirements"] = numpy_requirements
        result["numpy_satisfies_all_scipy_requirements"] = all(satisfies) if satisfies else None
        return result
    except Exception as exc:
        return {"has_warning": True, "warnings": [repr(exc)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pip_rc, pip_freeze = run_text([sys.executable, "-m", "pip", "freeze"])
    nvidia_rc, nvidia_smi = run_text(["nvidia-smi"])

    import torch

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "project_root": str(args.project_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "git_commit": run_text(["git", "rev-parse", "HEAD"], cwd=args.project_root)[1].strip(),
        "git_branch": run_text(["git", "branch", "--show-current"], cwd=args.project_root)[1].strip(),
        "git_dirty": bool(run_text(["git", "status", "--short"], cwd=args.project_root)[1].strip()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu_driver_from_nvidia_smi_available": nvidia_rc == 0,
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "sklearn": package_version("sklearn"),
        "h5py": package_version("h5py"),
        "pandas": package_version("pandas"),
        "open_clip": package_version("open_clip"),
        "transformers": package_version("transformers"),
        "huggingface_hub": package_version("huggingface_hub"),
        "tokenizers": package_version("tokenizers"),
        "ml_collections": package_version("ml_collections"),
        "env_vars": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        },
        "pip_freeze_command_rc": pip_rc,
        "nvidia_smi_rc": nvidia_rc,
        "scipy_numpy_compatibility": scipy_numpy_compat_note(),
    }

    (args.output_dir / "current_environment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "current_pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")
    (args.output_dir / "current_nvidia_smi.txt").write_text(nvidia_smi, encoding="utf-8")


if __name__ == "__main__":
    main()
