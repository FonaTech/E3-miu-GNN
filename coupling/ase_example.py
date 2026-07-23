#!/usr/bin/env python3
"""Minimal ASE coupling example using a native E3-miu-GNN checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from e3mu import E3MUCalculator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("structure")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-mode", default="auto")
    args = parser.parse_args()

    atoms = read(args.structure)
    atoms.calc = E3MUCalculator(
        args.checkpoint,
        device=args.device,
        model_mode=args.model_mode,
        log=lambda _message: None,
    )
    print(f"energy_eV={atoms.get_potential_energy():.12g}")
    max_force = float(((atoms.get_forces() ** 2).sum(axis=1).max()) ** 0.5)
    print(f"max_force_eV_per_A={max_force:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
