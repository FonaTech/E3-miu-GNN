#!/usr/bin/env python3
"""Execute one versioned E3-miu-GNN task and emit deterministic JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


ACTIONS = {"inspect", "predict", "relax", "evaluate", "phonon", "export_sevennet"}


def _find_repo(value: Optional[str]) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[3]])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "E3_miu_GNN.py").is_file() and (resolved / "e3mu").is_dir():
            return resolved
    raise FileNotFoundError("Could not locate the E3-miu-GNN repository; pass --repo")


def validate_task(task: Any) -> Dict[str, Any]:
    if not isinstance(task, Mapping):
        raise TypeError("Task JSON must be an object")
    if str(task.get("schema", "")) != "e3mu-task-v1":
        raise ValueError("Task schema must be 'e3mu-task-v1'")
    action = str(task.get("action", "")).strip()
    if action not in ACTIONS:
        raise ValueError("Unsupported action: " + action)
    if not str(task.get("checkpoint", "")).strip():
        raise ValueError("Task requires checkpoint")
    if action in {"predict", "relax", "phonon"} and not str(task.get("structure", "")).strip():
        raise ValueError(f"{action} requires structure")
    if action == "evaluate" and not str(task.get("dataset", "")).strip():
        raise ValueError("evaluate requires dataset")
    if "options" in task and not isinstance(task["options"], Mapping):
        raise TypeError("options must be an object")
    return dict(task)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--repo")
    parser.add_argument("--output")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        task_path = Path(args.task).expanduser().resolve()
        task = validate_task(json.loads(task_path.read_text(encoding="utf-8")))
        if args.validate_only:
            result: Dict[str, Any] = {
                "schema": "e3mu-task-validation-v1",
                "ok": True,
                "task": str(task_path),
                "action": task["action"],
            }
        else:
            repo = _find_repo(args.repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from e3mu import execute_task, write_json

            result = execute_task(task)
            if args.output:
                write_json(args.output, result, pretty=True)
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty, allow_nan=False))
        return 0
    except Exception as exc:
        error = {
            "schema": "e3mu-error-v1",
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(error, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
