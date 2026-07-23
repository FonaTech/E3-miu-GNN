#!/usr/bin/env python3
"""Check an E3-miu-GNN runtime before an agent starts a calculation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


CORE_PACKAGES = ("numpy", "torch", "torch-geometric", "ase", "scipy")
OPTIONAL_PACKAGES = ("h5py", "phonopy", "seekpath", "torch-pme", "tad-dftd4")


def _find_repo(value: Optional[str]) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[3]])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "E3_miu_GNN.py").is_file() and (resolved / "e3mu").is_dir():
            return resolved
    raise FileNotFoundError("Could not locate a repository containing E3_miu_GNN.py and e3mu/")


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_preflight(
    *,
    repo: Path,
    checkpoint: Optional[str],
    device: str,
) -> Dict[str, Any]:
    packages = {
        name: {"installed": (version := _version(name)) is not None, "version": version}
        for name in (*CORE_PACKAGES, *OPTIONAL_PACKAGES)
    }
    core_ready = all(packages[name]["installed"] for name in CORE_PACKAGES)
    device_report: Dict[str, Any] = {"requested": device}
    manifest = None
    errors = []
    if core_ready:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            import torch

            mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            cuda = bool(torch.cuda.is_available())
            resolved = (
                "cuda" if device == "auto" and cuda
                else "mps" if device == "auto" and mps
                else "cpu" if device == "auto"
                else device
            )
            available = resolved == "cpu" or (resolved == "mps" and mps) or (resolved == "cuda" and cuda)
            device_report.update({
                "resolved": resolved,
                "available": available,
                "mps_available": mps,
                "cuda_available": cuda,
            })
            if not available:
                errors.append(f"Requested device {resolved!r} is unavailable")
        except Exception as exc:
            errors.append(f"Device probe failed: {type(exc).__name__}: {exc}")
        if checkpoint:
            try:
                from e3mu import inspect_checkpoint

                manifest = inspect_checkpoint(checkpoint)
            except Exception as exc:
                errors.append(f"Checkpoint inspection failed: {type(exc).__name__}: {exc}")
    else:
        missing = [name for name in CORE_PACKAGES if not packages[name]["installed"]]
        errors.append("Missing core packages: " + ", ".join(missing))

    return {
        "schema": "e3mu-preflight-v1",
        "ok": not errors,
        "repository": str(repo),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "device": device_report,
        "checkpoint_manifest": manifest,
        "errors": errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        repo = _find_repo(args.repo)
        report = run_preflight(repo=repo, checkpoint=args.checkpoint, device=args.device)
    except Exception as exc:
        report = {
            "schema": "e3mu-preflight-v1",
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=args.pretty, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
