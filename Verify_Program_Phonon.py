#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes

from torch_geometric.data import Batch as _TGBatch

# The public repository name is E3_miu_GNN. Keep the historical module name as
# a fallback so existing local checkouts and checkpoints remain usable.
try:
    from E3_miu_GNN import (
        AtomicData,
        AtomicNumberTable,
        Configuration,
        DualLayerFieldModel,
        MixedGranularityE3GNN,
        resolve_device,
        set_default_dtype,
    )
except ImportError:  # pragma: no cover - compatibility with older checkouts
    from Dual_Layer_Atomic_E3_GNN import (  # type: ignore[no-redef]
        AtomicData,
        AtomicNumberTable,
        Configuration,
        DualLayerFieldModel,
        MixedGranularityE3GNN,
        resolve_device,
        set_default_dtype,
    )


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_periodic_box(atoms: Atoms, *, vacuum_padding_A: float = 8.0, min_box_size_A: float = 10.0) -> None:
    try:
        cell = atoms.get_cell()
        vol = float(cell.volume) if cell is not None else 0.0
    except Exception:
        vol = 0.0

    if vol > 1e-8:
        try:
            atoms.set_pbc([True, True, True])
        except Exception:
            pass
        return

    pos = np.asarray(atoms.get_positions(), dtype=float)
    if pos.size == 0:
        box = np.diag([float(min_box_size_A)] * 3)
        atoms.set_cell(box, scale_atoms=False)
        atoms.set_pbc([True, True, True])
        return

    min_pos = np.min(pos, axis=0)
    max_pos = np.max(pos, axis=0)
    size = max_pos - min_pos
    box = size + 2.0 * float(vacuum_padding_A)
    box = np.maximum(box, float(min_box_size_A))
    atoms.set_cell(np.diag(box), scale_atoms=False)
    atoms.set_pbc([True, True, True])
    shift = box / 2.0 - (min_pos + max_pos) / 2.0
    atoms.set_positions(pos + shift)


def _ase_to_phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    cell = np.array(atoms.get_cell().array, dtype=float)
    try:
        scaled = atoms.get_scaled_positions(wrap=True)
    except TypeError:
        scaled = atoms.get_scaled_positions()
        scaled = np.asarray(scaled, dtype=float)
        scaled = scaled - np.floor(scaled)
    return PhonopyAtoms(symbols=list(atoms.get_chemical_symbols()), cell=cell, scaled_positions=np.asarray(scaled, dtype=float))


def _phonopy_to_ase_atoms(ph_atoms: Any) -> Atoms:
    symbols = list(getattr(ph_atoms, "symbols", []))
    cell = np.asarray(getattr(ph_atoms, "cell", None), dtype=float)
    scaled = np.asarray(getattr(ph_atoms, "scaled_positions", None), dtype=float)
    return Atoms(symbols=symbols, cell=cell, scaled_positions=scaled, pbc=True)


_MODEL_MODES = ("full_coupled", "ground_only")
_SPIN_POLICIES = ("auto", "off", "required")


def _normalise_choice(value: str, choices: Sequence[str], *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if selected not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    return selected


def _normalise_field(field: Sequence[float]) -> np.ndarray:
    values = np.asarray(field, dtype=float).reshape(-1)
    if values.size != 3 or not np.isfinite(values).all():
        raise ValueError("electric_field must contain three finite values")
    return values.reshape(3)


def _spin_vectors_from_atoms(
    atoms: Atoms,
    *,
    policy: str,
    threshold: float = 1e-10,
) -> Optional[np.ndarray]:
    """Return unit spin directions while preserving explicitly non-magnetic sites."""
    policy = _normalise_choice(policy, _SPIN_POLICIES, name="spin_policy")
    if policy == "off":
        return None

    raw: Optional[np.ndarray] = None
    for key in ("e3mu_spins", "spins"):
        if key in atoms.arrays:
            raw = np.asarray(atoms.arrays[key], dtype=float)
            break
    if raw is None and "initial_magmoms" in atoms.arrays:
        raw = np.asarray(atoms.arrays["initial_magmoms"], dtype=float)

    if raw is None or raw.size == 0:
        if policy == "required":
            raise ValueError(
                "spin_policy='required' but the structure has no spin vectors or "
                "ASE initial magnetic moments"
            )
        return None
    if not np.isfinite(raw).all():
        raise ValueError("Structure spin data contains non-finite values")

    atom_count = len(atoms)
    if raw.shape == (atom_count,) or raw.shape == (atom_count, 1):
        scalar = raw.reshape(atom_count)
        spins = np.zeros((atom_count, 3), dtype=float)
        active = np.abs(scalar) > float(threshold)
        spins[active, 2] = np.sign(scalar[active])
    elif raw.shape == (atom_count, 3):
        norm = np.linalg.norm(raw, axis=1)
        active = norm > float(threshold)
        spins = np.zeros((atom_count, 3), dtype=float)
        spins[active] = raw[active] / norm[active, None]
    else:
        raise ValueError(
            f"Structure spin data has shape {raw.shape}; expected ({atom_count},) "
            f"or ({atom_count}, 3)"
        )

    if not np.any(active):
        if policy == "required":
            raise ValueError("spin_policy='required' but all structure spin moments are zero")
        return None
    return spins


def _read_structure_with_metadata(
    structure_path: str,
    *,
    log: Callable[[str], None] = print,
) -> Atoms:
    atoms = ase_read(structure_path)
    if isinstance(atoms, list):
        atoms = atoms[0]
    atoms = atoms.copy()

    # POSCAR does not store magnetic moments. VASP jobs generated by this
    # project keep the frozen spin state in a sibling metadata.json file.
    if _spin_vectors_from_atoms(atoms, policy="auto") is None:
        metadata_path = Path(structure_path).expanduser().resolve().parent / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                spins = np.asarray(metadata.get("spins", []), dtype=float)
                if spins.shape == (len(atoms), 3) and np.isfinite(spins).all():
                    atoms.set_array("e3mu_spins", spins.copy())
                    log(f"[{_now()}] Loaded frozen spin state from {metadata_path}")
            except Exception as exc:
                log(f"[{_now()}] WARN: could not read spin metadata {metadata_path}: {exc}")
    return atoms


def _map_unitcell_spins_to_supercell(
    phonon: Any,
    unitcell_spins: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if unitcell_spins is None:
        return None
    supercell = phonon.supercell
    s2u_map = np.asarray(getattr(supercell, "s2u_map", []), dtype=int).reshape(-1)
    unit_count = int(unitcell_spins.shape[0])
    if s2u_map.size != len(supercell):
        raise RuntimeError("Phonopy did not expose a valid supercell-to-unit-cell map")

    # In Phonopy, s2u_map stores representative supercell indices. u2u_map
    # converts those representatives to contiguous input-unit-cell indices.
    u2u_map = getattr(supercell, "u2u_map", None)
    if isinstance(u2u_map, dict):
        try:
            indices = np.asarray([int(u2u_map[int(value)]) for value in s2u_map], dtype=int)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid Phonopy u2u_map for frozen spins") from exc
    else:
        u2s_map = np.asarray(getattr(supercell, "u2s_map", []), dtype=int).reshape(-1)
        representative_to_unit = {int(value): index for index, value in enumerate(u2s_map)}
        try:
            indices = np.asarray([representative_to_unit[int(value)] for value in s2u_map], dtype=int)
        except KeyError as exc:
            raise RuntimeError("Could not map frozen spins onto the Phonopy supercell") from exc
    if indices.size == 0 or np.min(indices) < 0 or np.max(indices) >= unit_count:
        raise RuntimeError("Phonopy spin mapping points outside the input unit cell")
    return np.asarray(unitcell_spins, dtype=float)[indices]


def _parse_float_sequence_spec(
    text: str,
    *,
    name: str = "values",
    min_len: int = 1,
    max_points: int = 10000,
) -> List[float]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError(f"{name} is empty")

    m = re.match(r"^\s*(range)\s*\(\s*(.*?)\s*\)\s*$", raw, flags=re.IGNORECASE)
    if m:
        inside = m.group(2)
        parts = [p.strip() for p in inside.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"{name}: Range(start, step, stop) needs 3 values")
        start = float(parts[0])
        step = float(parts[1])
        stop = float(parts[2])
        if not (np.isfinite(start) and np.isfinite(step) and np.isfinite(stop)):
            raise ValueError(f"{name}: Range values must be finite")
        if abs(step) <= 0:
            raise ValueError(f"{name}: step must be non-zero")
        if (stop - start) * step < 0:
            raise ValueError(f"{name}: step sign does not reach stop")
        vals: List[float] = []
        x = float(start)
        tol = 1e-9
        while (x <= stop + tol) if step > 0 else (x >= stop - tol):
            vals.append(float(x))
            x += step
            if len(vals) > int(max_points):
                raise ValueError(f"{name}: too many points (>{max_points})")
        if len(vals) < int(min_len):
            raise ValueError(f"{name}: need at least {min_len} points")
        return vals

    m = re.match(r"^\s*(linspace)\s*\(\s*(.*?)\s*\)\s*$", raw, flags=re.IGNORECASE)
    if m:
        inside = m.group(2)
        parts = [p.strip() for p in inside.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"{name}: Linspace(start, stop, n) needs 3 values")
        start = float(parts[0])
        stop = float(parts[1])
        n = int(float(parts[2]))
        if n < int(min_len):
            raise ValueError(f"{name}: Linspace n must be >= {min_len}")
        if n > int(max_points):
            raise ValueError(f"{name}: Linspace n too large (>{max_points})")
        return np.linspace(start, stop, n).astype(float).tolist()

    work = raw
    for ch in "[](){}":
        work = work.replace(ch, " ")
    work = work.replace(";", " ").replace("\n", " ").replace("\t", " ").replace(",", " ")
    toks = [t for t in work.split() if t]
    if not toks:
        raise ValueError(f"{name}: no numeric values found")
    out: List[float] = []
    for t in toks:
        out.append(float(t))
        if len(out) > int(max_points):
            raise ValueError(f"{name}: too many points (>{max_points})")
    if len(out) < int(min_len):
        raise ValueError(f"{name}: need at least {min_len} points")
    return out


class DualLayerPESCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        model: DualLayerFieldModel,
        *,
        device: str = "auto",
        model_mode: str = "full_coupled",
        total_charge: float = 0.0,
        electric_field: Sequence[float] = (0.0, 0.0, 0.0),
        spin_policy: str = "auto",
        compile_inference: bool = False,
        log: Callable[[str], None] = print,
    ):
        super().__init__()
        self.log = log
        self.model_mode = _normalise_choice(model_mode, _MODEL_MODES, name="model_mode")
        self.spin_policy = _normalise_choice(spin_policy, _SPIN_POLICIES, name="spin_policy")
        self.total_charge = float(total_charge)
        if not np.isfinite(self.total_charge):
            raise ValueError("total_charge must be finite")
        self.electric_field = _normalise_field(electric_field)
        self._is_mixed = isinstance(model, MixedGranularityE3GNN)
        if self.spin_policy == "required" and (
            self.model_mode != "full_coupled"
            or not self._is_mixed
            or not bool(getattr(model.cfg, "enable_spin", False))
        ):
            raise ValueError(
                "spin_policy='required' needs a full_coupled mixed-granularity "
                "checkpoint with enable_spin=True"
            )

        requested_dtype = str(getattr(getattr(model, "cfg", None), "dtype", "float32"))
        self.device, runtime_dtype = resolve_device(device, dtype=requested_dtype)
        set_default_dtype(runtime_dtype)
        self.dtype = torch.get_default_dtype()
        self._source_model = model
        self.model = model
        self._z_to_index = {int(z): i for i, z in enumerate(getattr(model, "z_table_zs", []) or [])}
        self._z_table = AtomicNumberTable(list(self._z_to_index))
        self._compiled = False
        self._compile_requested = bool(compile_inference)
        self._warned_missing_spin = False
        self.calculation_count = 0
        self.last_components: Dict[str, float] = {}

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if self._compile_requested:
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                self.log(f"[{_now()}] WARN: torch.compile is unavailable; using eager inference")
            else:
                try:
                    self.model = compile_fn(self.model, dynamic=True, fullgraph=False)
                    self._compiled = True
                    self.log(f"[{_now()}] Native model inference scheduled for torch.compile")
                except Exception as exc:
                    self.model = self._source_model
                    self.log(f"[{_now()}] WARN: torch.compile setup failed; using eager inference: {exc}")

        flags = [
            name.removeprefix("enable_")
            for name in ("enable_qeq", "enable_pme", "enable_deq", "enable_d4", "enable_spin", "enable_film", "enable_dmi")
            if bool(getattr(model.cfg, name, False))
        ]
        self.log(
            f"[{_now()}] Native PES: class={type(model).__name__} mode={self.model_mode} "
            f"device={self.device.type} dtype={runtime_dtype} physics={flags or ['short_range']}"
        )

    def _forward(self, batch: Any, *, use_spin: bool) -> Dict[str, torch.Tensor]:
        use_coupled = self.model_mode == "full_coupled"
        kwargs: Dict[str, Any] = {
            "training": False,
            "compute_forces": True,
            "compute_bec": False,
            "use_response_terms": use_coupled,
            "retain_graph": False,
        }
        if self._is_mixed:
            kwargs["use_domain_terms"] = bool(
                use_coupled
                and (getattr(self._source_model.cfg, "enable_qeq", False) or getattr(self._source_model.cfg, "enable_pme", False))
            )
            kwargs["use_spin_terms"] = bool(use_coupled and use_spin)
        return self.model(batch, **kwargs)

    def summary(self) -> Dict[str, Any]:
        cfg = getattr(self._source_model, "cfg", None)
        configured = [
            name
            for name in ("qeq", "pme", "deq", "d4", "spin", "film", "dmi")
            if bool(getattr(cfg, f"enable_{name}", False))
        ]
        return {
            "backend": "native_e3mu",
            "model_class": type(self._source_model).__name__,
            "mode": self.model_mode,
            "device": self.device.type,
            "dtype": str(self.dtype).replace("torch.", ""),
            "configured_physics": configured,
            "spin_policy": self.spin_policy,
            "total_charge": self.total_charge,
            "electric_field_V_per_A": self.electric_field.tolist(),
            "compiled": self._compiled,
            "force_evaluations": self.calculation_count,
            "last_components_eV": dict(self.last_components),
        }

    def calculate(self, atoms: Optional[Atoms] = None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms is None")

        at = atoms
        cell = np.asarray(at.get_cell().array, dtype=float)
        pbc = tuple(bool(x) for x in at.get_pbc())
        pos = np.asarray(at.get_positions(), dtype=float)
        zs = np.asarray(at.get_atomic_numbers(), dtype=int)
        if zs.size == 0:
            raise ValueError("Empty structure")
        if any(pbc) and (not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1e-10):
            raise ValueError("Periodic phonon inference requires a finite, non-singular cell")

        atom_types = np.asarray([self._z_to_index.get(int(z), -1) for z in zs], dtype=int)
        if np.any(atom_types < 0):
            missing = sorted({int(z) for z, t in zip(zs.tolist(), atom_types.tolist()) if int(t) < 0})
            raise ValueError(f"Structure contains elements not in checkpoint z_table_zs: {missing}")

        spins = _spin_vectors_from_atoms(at, policy=self.spin_policy)
        use_spin = bool(
            self.model_mode == "full_coupled"
            and self._is_mixed
            and getattr(self._source_model.cfg, "enable_spin", False)
            and spins is not None
        )
        if (
            self.model_mode == "full_coupled"
            and self._is_mixed
            and getattr(self._source_model.cfg, "enable_spin", False)
            and spins is None
            and not self._warned_missing_spin
        ):
            self.log(
                f"[{_now()}] No magnetic state found; spin energy is disabled while "
                "the remaining coupled physics stays active"
            )
            self._warned_missing_spin = True

        model_properties: Dict[str, Any] = {
            "field": self.electric_field,
            "total_charge": self.total_charge,
        }
        if spins is not None:
            model_properties["spins"] = spins
        config = Configuration(
            atomic_numbers=zs,
            positions=pos,
            properties=model_properties,
            property_weights={},
            cell=cell,
            pbc=pbc,
        )
        data = AtomicData.from_config(
            config,
            z_table=self._z_table,
            cutoff=float(getattr(self._source_model.cfg, "r_max", 5.0)),
        )
        batch = _TGBatch.from_data_list([data]).to(self.device)

        with torch.enable_grad():
            try:
                out = self._forward(batch, use_spin=use_spin)
            except Exception as exc:
                if not self._compiled:
                    raise
                self.log(
                    f"[{_now()}] WARN: compiled inference failed; retrying eagerly: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.model = self._source_model
                self._compiled = False
                out = self._forward(batch, use_spin=use_spin)

        energy_tensor = out.get("energy")
        forces_tensor = out.get("forces")
        if not isinstance(energy_tensor, torch.Tensor) or energy_tensor.numel() != 1:
            raise RuntimeError("Native model returned an invalid scalar energy")
        if not isinstance(forces_tensor, torch.Tensor):
            raise RuntimeError("Native model did not return forces")
        E = float(energy_tensor.detach().cpu().reshape(-1)[0].item())
        F = forces_tensor.detach().cpu().numpy().astype(float, copy=False)
        if not np.isfinite(E):
            raise FloatingPointError("Native model produced a non-finite phonon energy")
        if F.shape != (int(zs.size), 3):
            raise RuntimeError(f"Native model returned forces with shape {F.shape}; expected {(int(zs.size), 3)}")
        if not np.isfinite(F).all():
            raise FloatingPointError("Native model produced non-finite phonon forces")

        component_keys = (
            "energy_short",
            "energy_qeq",
            "energy_pme",
            "energy_d4",
            "energy_spin",
            "energy_response",
            "qeq_residual",
            "deq_residual",
            "coupling_residual",
        )
        components: Dict[str, float] = {}
        for key in component_keys:
            value = out.get(key)
            if isinstance(value, torch.Tensor) and value.numel():
                scalar = float(value.detach().cpu().reshape(-1)[0].item())
                if not np.isfinite(scalar):
                    raise FloatingPointError(f"Native model produced non-finite {key}")
                components[key] = scalar
        self.last_components = components
        self.calculation_count += 1
        self.results["energy"] = E
        self.results["forces"] = np.asarray(F, dtype=float).copy()


class SevenNetTSCalculator(Calculator):
    """ASE Calculator wrapping a SevenNet TorchScript model exported by E3MUAsSevenNetTorchScriptModel."""
    implemented_properties = ("energy", "forces")

    def __init__(self, model_path: str, *, device: str = "auto"):
        super().__init__()
        self.device, _runtime_dtype = resolve_device(device, dtype="float32")
        self.calculation_count = 0
        extra_files: Dict[str, Any] = {
            "chemical_symbols_to_index": b"",
            "cutoff": b"",
            "num_species": b"",
        }
        self._model = torch.jit.load(model_path, map_location=self.device, _extra_files=extra_files)
        self._model.eval()

        # Parse metadata from _extra_files
        raw_symbols = extra_files["chemical_symbols_to_index"]
        if isinstance(raw_symbols, bytes):
            raw_symbols = raw_symbols.decode("utf-8", errors="replace")
        self._symbols = [s for s in raw_symbols.split() if s]
        try:
            from ase.data import chemical_symbols as _ase_syms
            self._sym_to_z = {s: i for i, s in enumerate(_ase_syms)}
        except Exception:
            self._sym_to_z = {}
        self._sym_to_idx = {s: i for i, s in enumerate(self._symbols)}

        raw_cutoff = extra_files["cutoff"]
        if isinstance(raw_cutoff, bytes):
            raw_cutoff = raw_cutoff.decode("utf-8", errors="replace")
        try:
            self._cutoff = float(raw_cutoff.strip())
        except Exception:
            self._cutoff = 5.0

    def _z_to_idx(self, z: int) -> int:
        try:
            from ase.data import chemical_symbols as _ase_syms
            sym = _ase_syms[z]
        except Exception:
            sym = str(z)
        return self._sym_to_idx.get(sym, -1)

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": "sevennet_torchscript",
            "model_class": "SevenNetTorchScript",
            "mode": "ground_only",
            "device": self.device.type,
            "dtype": "float32",
            "configured_physics": ["short_range"],
            "spin_policy": "off",
            "total_charge": 0.0,
            "electric_field_V_per_A": [0.0, 0.0, 0.0],
            "compiled": True,
            "force_evaluations": self.calculation_count,
            "last_components_eV": {},
        }

    def calculate(self, atoms: Optional[Atoms] = None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms is None")

        cell = np.asarray(atoms.get_cell().array, dtype=float)
        pbc = tuple(bool(x) for x in atoms.get_pbc())
        pos = np.asarray(atoms.get_positions(), dtype=float)
        zs = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if zs.size == 0:
            raise ValueError("Empty structure")

        atom_idx = np.asarray([self._z_to_idx(int(z)) for z in zs], dtype=int)
        if np.any(atom_idx < 0):
            missing = sorted({int(z) for z, t in zip(zs.tolist(), atom_idx.tolist()) if int(t) < 0})
            raise ValueError(f"SevenNet model: elements not in chemical_symbols_to_index: {missing}")

        ase_atoms = Atoms(numbers=zs, positions=pos, cell=cell, pbc=pbc)
        i_idx, j_idx, S = neighbor_list("ijS", ase_atoms, cutoff=self._cutoff)
        edge_index = torch.tensor(np.stack([i_idx, j_idx], axis=0), dtype=torch.long, device=self.device)
        pbc_shift = torch.tensor(S, dtype=torch.float32, device=self.device)
        cell_t = torch.tensor(cell, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3,3)

        pos_t = torch.tensor(pos, dtype=torch.float32, device=self.device).requires_grad_(True)
        x_t = torch.tensor(atom_idx, dtype=torch.long, device=self.device)
        num_atoms_t = torch.tensor([int(zs.size)], dtype=torch.long, device=self.device)

        data_in: Dict[str, torch.Tensor] = {
            "pos": pos_t,
            "edge_index": edge_index,
            "pbc_shift": pbc_shift,
            "cell_lattice_vectors": cell_t,
            "x": x_t,
            "num_atoms": num_atoms_t,
        }

        with torch.enable_grad():
            out = self._model(data_in)

        E = float(out["inferred_total_energy"].detach().cpu().view(-1)[0].item())
        F = out["inferred_force"].detach().cpu().numpy().astype(float)
        if not np.isfinite(E):
            raise FloatingPointError("SevenNet model produced a non-finite phonon energy")
        if F.shape != (int(zs.size), 3):
            raise RuntimeError(f"SevenNet model returned forces with shape {F.shape}; expected {(int(zs.size), 3)}")
        if not np.isfinite(F).all():
            raise FloatingPointError("SevenNet model produced non-finite phonon forces")
        self.calculation_count += 1
        self.results["energy"] = E
        self.results["forces"] = F


def _make_pes_calculator(
    model,
    *,
    device: str,
    e3mu_compile_infer: bool = False,
    model_mode: str = "full_coupled",
    total_charge: float = 0.0,
    electric_field: Sequence[float] = (0.0, 0.0, 0.0),
    spin_policy: str = "auto",
    log: Callable[[str], None] = print,
):
    if isinstance(model, SevenNetTSCalculator):
        if spin_policy == "required":
            raise ValueError("SevenNet TorchScript exports do not contain the mixed spin Hamiltonian")
        if abs(float(total_charge)) > 1e-12 or np.linalg.norm(_normalise_field(electric_field)) > 1e-12:
            raise ValueError(
                "SevenNet TorchScript exports cannot apply total charge or external electric fields; "
                "use a native mixed-granularity checkpoint"
            )
        if model_mode != "ground_only":
            log(
                f"[{_now()}] WARN: SevenNet TorchScript exports contain only the "
                "short-range ground model; using ground_only mode"
            )
        if bool(e3mu_compile_infer):
            log(f"[{_now()}] SevenNet input is already TorchScript compiled")
        return model  # already an ASE Calculator
    return DualLayerPESCalculator(
        model,
        device=device,
        model_mode=model_mode,
        total_charge=total_charge,
        electric_field=electric_field,
        spin_policy=spin_policy,
        compile_inference=e3mu_compile_infer,
        log=log,
    )


def _load_model_for_phonon(model_ckpt: str, device: str, log: Callable[[str], None] = print):
    """Load either a DualLayerFieldModel .pt checkpoint or a SevenNet TorchScript .pt model."""
    # First try loading as our native DualLayerFieldModel checkpoint
    try:
        model = DualLayerFieldModel.load(model_ckpt, map_location="cpu")
        flags = [
            name.removeprefix("enable_")
            for name in ("enable_qeq", "enable_pme", "enable_deq", "enable_d4", "enable_spin", "enable_film", "enable_dmi")
            if bool(getattr(model.cfg, name, False))
        ]
        log(
            f"[{_now()}] Loaded {type(model).__name__} from {model_ckpt}; "
            f"elements={len(getattr(model, 'z_table_zs', []))} "
            f"configured_physics={flags or ['short_range']}"
        )
        return model
    except Exception as e_dual:
        log(f"[{_now()}] Not a DualLayerFieldModel ({type(e_dual).__name__}: {e_dual}); trying SevenNet TorchScript...")

    # Fall back to SevenNet TorchScript
    try:
        calc = SevenNetTSCalculator(model_ckpt, device=device)
        log(f"[{_now()}] Loaded SevenNet TorchScript model from {model_ckpt} (cutoff={calc._cutoff:.3g} Å, species={calc._symbols})")
        return calc
    except Exception as e_sn:
        raise RuntimeError(
            f"Could not load model from '{model_ckpt}'.\n"
            f"  DualLayerFieldModel error: {e_dual}\n"
            f"  SevenNet TorchScript error: {e_sn}"
        ) from e_sn


def compute_phonon_thermo_phonopy(
    *,
    model_ckpt: str,
    structure_path: str,
    device: str = "auto",
    e3mu_compile_infer: bool = False,
    model_mode: str = "full_coupled",
    total_charge: float = 0.0,
    electric_field: Sequence[float] = (0.0, 0.0, 0.0),
    spin_policy: str = "auto",
    supercell_matrix: Sequence[Sequence[int]] = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    displacement_amplitude_A: float = 0.01,
    subtract_equilibrium_forces: bool = True,
    dos_mesh: Tuple[int, int, int] = (10, 10, 10),
    dos_sigma_THz: Optional[float] = None,
    band_npoints: int = 101,
    thermal_temperatures_K: Optional[Sequence[float]] = None,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    from phonopy import Phonopy

    P = np.asarray(supercell_matrix, dtype=int)
    if P.shape != (3, 3):
        raise ValueError("supercell_matrix must be 3x3")
    determinant = int(round(float(np.linalg.det(P))))
    if determinant == 0:
        raise ValueError("supercell_matrix must be non-singular")
    supercell_multiplicity = abs(determinant)
    amp = float(displacement_amplitude_A)
    if not np.isfinite(amp) or amp <= 0:
        raise ValueError("displacement_amplitude_A must be > 0")
    model_mode = _normalise_choice(model_mode, _MODEL_MODES, name="model_mode")
    spin_policy = _normalise_choice(spin_policy, _SPIN_POLICIES, name="spin_policy")
    field_values = _normalise_field(electric_field)
    unitcell_charge = float(total_charge)
    if not np.isfinite(unitcell_charge):
        raise ValueError("total_charge must be finite")
    resolved_device, _ = resolve_device(device, dtype="float32")

    model = _load_model_for_phonon(model_ckpt, device=resolved_device.type, log=log)
    atoms = _read_structure_with_metadata(structure_path, log=log)
    _ensure_periodic_box(atoms)
    unitcell_spins = _spin_vectors_from_atoms(atoms, policy=spin_policy)

    unitcell = _ase_to_phonopy_atoms(atoms)
    try:
        phonon = Phonopy(unitcell, supercell_matrix=P, primitive_matrix="auto", log_level=1)
    except Exception as e:
        log(f"[{_now()}] WARN: primitive_matrix='auto' failed; falling back to primitive_matrix=None: {e}")
        phonon = Phonopy(unitcell, supercell_matrix=P, primitive_matrix=None, log_level=1)

    if stop_flag is not None and stop_flag():
        raise RuntimeError("Stopped")
    if progress is not None:
        progress({"type": "prop", "task": "Phonon/Thermo", "overall_frac": 0.01, "current": 0, "total": 1, "stage": "generate_displacements"})

    phonon.generate_displacements(distance=amp)
    displaced = list(phonon.supercells_with_displacements or [])
    if not displaced:
        raise RuntimeError("phonopy generated zero displaced supercells.")
    supercell_spins = _map_unitcell_spins_to_supercell(phonon, unitcell_spins)
    supercell_charge = unitcell_charge * float(supercell_multiplicity)
    calculator = _make_pes_calculator(
        model,
        device=resolved_device.type,
        e3mu_compile_infer=e3mu_compile_infer,
        model_mode=model_mode,
        total_charge=supercell_charge,
        electric_field=field_values,
        spin_policy=spin_policy,
        log=log,
    )
    spin_supported = bool(
        isinstance(calculator, DualLayerPESCalculator)
        and calculator.model_mode == "full_coupled"
        and calculator._is_mixed
        and getattr(calculator._source_model.cfg, "enable_spin", False)
    )
    active_supercell_spins = supercell_spins if spin_supported else None
    if supercell_spins is not None and not spin_supported:
        log(f"[{_now()}] Structure spin state is present but inactive for the selected model mode")
    log(
        f"[{_now()}] Phonopy finite displacement: amp={amp:.4g} Å "
        f"n_supercells={len(displaced)} atoms/supercell={len(phonon.supercell)} "
        f"charge/supercell={supercell_charge:.6g} e "
        f"spin_state={'frozen' if active_supercell_spins is not None else 'off'}"
    )

    reference_forces: Optional[np.ndarray] = None
    if bool(subtract_equilibrium_forces):
        if progress is not None:
            progress(
                {
                    "type": "prop",
                    "task": "Phonon/Thermo",
                    "overall_frac": 0.04,
                    "current": 0,
                    "total": int(len(displaced)),
                    "stage": "equilibrium_forces",
                }
            )
        reference_atoms = _phonopy_to_ase_atoms(phonon.supercell)
        if active_supercell_spins is not None:
            reference_atoms.set_array("e3mu_spins", active_supercell_spins.copy())
        reference_atoms.calc = calculator
        reference_forces = np.asarray(reference_atoms.get_forces(), dtype=float)
        if reference_forces.shape != (len(reference_atoms), 3) or not np.isfinite(reference_forces).all():
            raise FloatingPointError("Equilibrium supercell produced invalid reference forces")
        log(
            f"[{_now()}] Equilibrium-force correction: max|F0|="
            f"{float(np.max(np.linalg.norm(reference_forces, axis=1))):.6g} eV/Å"
        )

    forces_sets = []
    total = int(len(displaced))
    for i, ph_sc in enumerate(displaced, start=1):
        if stop_flag is not None and stop_flag():
            raise RuntimeError("Stopped")
        if progress is not None:
            progress(
                {
                    "type": "prop",
                    "task": "Phonon/Thermo",
                    "overall_frac": 0.05 + 0.55 * float(i) / float(max(1, total)),
                    "current": int(i),
                    "total": int(total),
                    "stage": "forces",
                }
            )
        sc_atoms = _phonopy_to_ase_atoms(ph_sc)
        if active_supercell_spins is not None:
            sc_atoms.set_array("e3mu_spins", active_supercell_spins.copy())
        sc_atoms.calc = calculator
        displaced_forces = np.asarray(sc_atoms.get_forces(), dtype=float)
        if reference_forces is not None:
            displaced_forces = displaced_forces - reference_forces
        if displaced_forces.shape != (len(sc_atoms), 3) or not np.isfinite(displaced_forces).all():
            raise FloatingPointError(f"Invalid finite-displacement forces for supercell {i}/{total}")
        forces_sets.append(displaced_forces)

    phonon.forces = np.asarray(forces_sets, dtype=float)
    phonon.produce_force_constants()
    try:
        phonon.symmetrize_force_constants()
    except Exception:
        pass
    force_constants = np.asarray(phonon.force_constants, dtype=float)
    if force_constants.size == 0 or not np.isfinite(force_constants).all():
        raise FloatingPointError("Phonopy produced non-finite force constants")

    if stop_flag is not None and stop_flag():
        raise RuntimeError("Stopped")
    if progress is not None:
        progress({"type": "prop", "task": "Phonon/Thermo", "overall_frac": 0.65, "current": 0, "total": 1, "stage": "band_structure"})

    try:
        phonon.auto_band_structure(
            npoints=int(max(5, band_npoints)),
            with_eigenvectors=False,
            with_group_velocities=False,
            plot=False,
            write_yaml=False,
        )
    except ModuleNotFoundError as exc:
        if "seekpath" in str(exc).lower():
            raise RuntimeError(
                "Automatic high-symmetry band paths require seekpath; install project requirements"
            ) from exc
        raise
    band = phonon.get_band_structure_dict()
    bs = phonon.band_structure
    labels = list(getattr(bs, "labels", []) or [])
    path_connections = list(getattr(bs, "path_connections", []) or [])

    if stop_flag is not None and stop_flag():
        raise RuntimeError("Stopped")
    if progress is not None:
        progress({"type": "prop", "task": "Phonon/Thermo", "overall_frac": 0.75, "current": 0, "total": 1, "stage": "dos"})

    mesh = tuple(int(x) for x in dos_mesh)
    if len(mesh) != 3 or any(value <= 0 for value in mesh):
        raise ValueError("dos_mesh must contain three positive integers")
    phonon.run_mesh(mesh=np.asarray(mesh, dtype=int))
    requested_dos_sigma = None if dos_sigma_THz in (None, "", "none") else float(dos_sigma_THz)
    if requested_dos_sigma is not None and (
        not np.isfinite(requested_dos_sigma) or requested_dos_sigma <= 0.0
    ):
        raise ValueError("dos_sigma_THz must be a positive finite value")
    effective_dos_sigma = requested_dos_sigma
    if effective_dos_sigma is None:
        mesh_frequencies = np.asarray(
            (phonon.get_mesh_dict() or {}).get("frequencies", []), dtype=float
        )
        if mesh_frequencies.size:
            frequency_span = float(np.nanmax(mesh_frequencies) - np.nanmin(mesh_frequencies))
            if not np.isfinite(frequency_span):
                raise FloatingPointError("Phonopy mesh contains non-finite frequencies")
            if frequency_span <= 1e-12:
                effective_dos_sigma = 0.05
                log(
                    f"[{_now()}] Degenerate mesh spectrum; using DOS sigma="
                    f"{effective_dos_sigma:g} THz"
                )
    phonon.run_total_dos(
        sigma=effective_dos_sigma,
        use_tetrahedron_method=effective_dos_sigma is None,
    )
    dos = phonon.get_total_dos_dict()

    thermal_out = None
    pretend_real_used = False
    if thermal_temperatures_K:
        if stop_flag is not None and stop_flag():
            raise RuntimeError("Stopped")
        if progress is not None:
            progress({"type": "prop", "task": "Phonon/Thermo", "overall_frac": 0.85, "current": 0, "total": 1, "stage": "thermal"})
        temps = np.sort(np.asarray(list(thermal_temperatures_K), dtype=float))
        if temps.size >= 2:
            t_min = float(temps[0])
            t_max = float(temps[-1])
            t_step = float(temps[1] - temps[0])
            if t_step <= 0:
                raise ValueError("thermal_temperatures_K must be increasing with constant step")
            if not np.allclose(np.diff(temps), t_step, rtol=1e-8, atol=1e-10):
                raise ValueError("thermal_temperatures_K must use a constant step")
            try:
                phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step, pretend_real=False)
                pretend_real_used = False
            except Exception as e:
                log(f"[{_now()}] WARN: thermal_properties failed; retry with pretend_real=True: {e}")
                phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step, pretend_real=True)
                pretend_real_used = True
            tp = phonon.get_thermal_properties_dict() or {}
            T = np.asarray(tp.get("temperatures", []), dtype=float)
            F = np.asarray(tp.get("free_energy", []), dtype=float)
            S = np.asarray(tp.get("entropy", []), dtype=float)
            Cv = np.asarray(tp.get("heat_capacity", []), dtype=float)
            n = min(int(T.size), int(F.size), int(S.size), int(Cv.size))
            T, F, S, Cv = T[:n], F[:n], S[:n], Cv[:n]
            H = F + (T * S) / 1000.0
            thermal_out = {
                "pretend_real": bool(pretend_real_used),
                "temperatures_K": T.tolist(),
                "free_energy_kJ_per_mol": F.tolist(),
                "entropy_J_per_K_mol": S.tolist(),
                "heat_capacity_J_per_K_mol": Cv.tolist(),
                "enthalpy_kJ_per_mol": H.tolist(),
            }

    # Frequency summary
    fmin = float("nan")
    fmax = float("nan")
    try:
        freqs_all = []
        for seg in (band.get("frequencies", []) or []):
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2:
                freqs_all.append(arr)
        if freqs_all:
            allf = np.concatenate(freqs_all)
            fmin = float(np.nanmin(allf))
            fmax = float(np.nanmax(allf))
    except Exception:
        pass

    if progress is not None:
        progress({"type": "prop", "task": "Phonon/Thermo", "overall_frac": 1.0, "current": 1, "total": 1, "stage": "done"})

    def _to_list(x: Any) -> Any:
        try:
            return np.array(x).tolist()
        except Exception:
            return x

    calculator_info = calculator.summary() if hasattr(calculator, "summary") else {}
    calculator_info["force_evaluations"] = int(getattr(calculator, "calculation_count", len(displaced)))
    maximum_force_drift = max(
        (float(np.linalg.norm(np.mean(values, axis=0))) for values in forces_sets),
        default=0.0,
    )
    return {
        "ok": True,
        "source": "phonopy_finite_displacement",
        "formula": str(getattr(atoms, "get_chemical_formula", lambda: "")()),
        "model": calculator_info,
        "input_unitcell_total_charge_e": unitcell_charge,
        "supercell_total_charge_e": supercell_charge,
        "electric_field_V_per_A": field_values.tolist(),
        "spin_treatment": "frozen" if active_supercell_spins is not None else "disabled",
        "subtract_equilibrium_forces": bool(subtract_equilibrium_forces),
        "max_corrected_force_drift_eV_per_A": maximum_force_drift,
        "displacement_amplitude_A": float(amp),
        "supercell_matrix": P.tolist(),
        "dos_mesh": list(mesh),
        "dos_sigma_THz": effective_dos_sigma,
        "requested_dos_sigma_THz": requested_dos_sigma,
        "band_npoints": int(max(5, band_npoints)),
        "band": {
            "distances": _to_list(band.get("distances", [])),
            "qpoints": _to_list(band.get("qpoints", [])),
            "frequencies_THz": _to_list(band.get("frequencies", [])),
            "labels": labels,
            "path_connections": _to_list(path_connections) if path_connections else [],
            "fmin_THz": fmin,
            "fmax_THz": fmax,
        },
        "dos": {
            "frequency_points_THz": _to_list(dos.get("frequency_points", [])),
            "total_dos": _to_list(dos.get("total_dos", [])),
        },
        "thermal": thermal_out,
    }


@dataclass
class _Defaults:
    supercell: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]] = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    amp_A: float = 0.01
    dos_mesh: Tuple[int, int, int] = (10, 10, 10)
    dos_sigma_THz: Optional[float] = None
    band_npoints: int = 101
    thermo_spec: str = "Range(0, 50, 1000)"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Run Phonopy (Band + DOS + Thermo)")
        self.geometry("1100x820")

        self.var_model = tk.StringVar(value="")
        self.var_structure = tk.StringVar(value="")
        self.var_device = tk.StringVar(value="auto")
        self.var_model_mode = tk.StringVar(value="full_coupled")
        self.var_total_charge = tk.StringVar(value="0.0")
        self.var_field_x = tk.StringVar(value="0.0")
        self.var_field_y = tk.StringVar(value="0.0")
        self.var_field_z = tk.StringVar(value="0.0")
        self.var_spin_policy = tk.StringVar(value="auto")
        self.var_compile = tk.BooleanVar(value=False)
        self.var_subtract_equilibrium = tk.BooleanVar(value=True)
        self._status = tk.StringVar(value="Idle")

        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._evt_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = False
        self._last: Optional[Dict[str, Any]] = None

        self._build_ui()
        self.after(100, self._tick)

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(root, highlightthickness=0)
        vsb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_evt=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(evt):
            try:
                canvas.itemconfigure(win, width=int(evt.width))
            except Exception:
                pass

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(evt):
            delta = getattr(evt, "delta", 0)
            if delta:
                step = int(-1 * (delta / 120)) if os.name == "nt" else int(-1 * delta)
                step = step if step != 0 else (-1 if delta > 0 else 1)
                canvas.yview_scroll(step, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda _e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda _e: canvas.yview_scroll(1, "units"))

        top = ttk.Frame(inner)
        top.pack(fill="x", padx=10, pady=(10, 0))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)

        ttk.Button(top, text="Load model (.pt/.pth)", command=self._pick_model).grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Button(top, text="Load structure", command=self._pick_structure).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(top, text="Run", command=self._start).grid(row=0, column=2, sticky="ew", padx=4, pady=2)

        paths = ttk.Frame(inner)
        paths.pack(fill="x", padx=10, pady=(6, 0))
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text="Model:").grid(row=0, column=0, sticky="w", padx=(4, 6), pady=1)
        ttk.Label(paths, textvariable=self.var_model, wraplength=980, justify="left").grid(row=0, column=1, sticky="ew", padx=4, pady=1)
        ttk.Label(paths, text="Structure:").grid(row=1, column=0, sticky="w", padx=(4, 6), pady=1)
        ttk.Label(paths, textvariable=self.var_structure, wraplength=980, justify="left").grid(row=1, column=1, sticky="ew", padx=4, pady=1)

        physics = ttk.LabelFrame(inner, text="Model Physics")
        physics.pack(fill="x", padx=10, pady=(8, 0))
        for column in range(6):
            physics.columnconfigure(column, weight=1 if column in (1, 3, 5) else 0)

        ttk.Label(physics, text="PES mode").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=5)
        ttk.Combobox(
            physics,
            textvariable=self.var_model_mode,
            values=_MODEL_MODES,
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=5)
        ttk.Label(physics, text="Device").grid(row=0, column=2, sticky="w", padx=(0, 4), pady=5)
        ttk.Combobox(
            physics,
            textvariable=self.var_device,
            values=("auto", "cpu", "mps", "cuda"),
            state="readonly",
            width=10,
        ).grid(row=0, column=3, sticky="ew", padx=(0, 10), pady=5)
        ttk.Label(physics, text="Spin state").grid(row=0, column=4, sticky="w", padx=(0, 4), pady=5)
        ttk.Combobox(
            physics,
            textvariable=self.var_spin_policy,
            values=_SPIN_POLICIES,
            state="readonly",
            width=12,
        ).grid(row=0, column=5, sticky="ew", padx=(0, 8), pady=5)

        ttk.Label(physics, text="Unit-cell charge (e)").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=5)
        ttk.Entry(physics, textvariable=self.var_total_charge, width=12).grid(
            row=1, column=1, sticky="ew", padx=(0, 10), pady=5
        )
        ttk.Label(physics, text="Electric field (V/Å)").grid(row=1, column=2, sticky="w", padx=(0, 4), pady=5)
        field_frame = ttk.Frame(physics)
        field_frame.grid(row=1, column=3, columnspan=3, sticky="ew", padx=(0, 8), pady=5)
        for column in range(3):
            field_frame.columnconfigure(column, weight=1)
        for column, (axis, variable) in enumerate(
            (("Ex", self.var_field_x), ("Ey", self.var_field_y), ("Ez", self.var_field_z))
        ):
            item = ttk.Frame(field_frame)
            item.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
            ttk.Label(item, text=axis).pack(side="left", padx=(0, 3))
            ttk.Entry(item, textvariable=variable, width=10).pack(side="left", fill="x", expand=True)

        ttk.Checkbutton(
            physics,
            text="Compile native inference",
            variable=self.var_compile,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 7))
        ttk.Checkbutton(
            physics,
            text="Subtract equilibrium forces",
            variable=self.var_subtract_equilibrium,
        ).grid(row=2, column=2, columnspan=3, sticky="w", padx=(0, 8), pady=(2, 7))

        ttk.Label(inner, textvariable=self._status).pack(fill="x", padx=10, pady=(8, 6))

        self._has_mpl = False
        self._plot_ready = False
        self._txt = tk.Text(inner, height=8)
        self._txt.pack(fill="x", padx=10, pady=(0, 8))

        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            self._has_mpl = True
            plots = ttk.Frame(inner)
            plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            # Figure 1: phonon band + DOS (shared Y: Frequency (THz))
            fig_ph = Figure(figsize=(10, 4), dpi=100)
            self._fig_ph = fig_ph
            gs_ph = fig_ph.add_gridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.35)
            self._ax_band = fig_ph.add_subplot(gs_ph[0, 0])
            self._ax_dos = fig_ph.add_subplot(gs_ph[0, 1], sharey=self._ax_band)
            self._ax_band.set_title("Phonon Band Structure")
            self._ax_band.set_xlabel("k-path distance")
            self._ax_band.set_ylabel("Frequency (THz)")
            self._ax_band.grid(True, alpha=0.3)
            self._ax_dos.set_title("DOS")
            self._ax_dos.set_xlabel("DOS (arb.)")
            self._ax_dos.grid(True, alpha=0.3)
            self._ax_dos.tick_params(labelleft=False)

            self._canvas_ph = FigureCanvasTkAgg(fig_ph, master=plots)
            self._canvas_ph.get_tk_widget().pack(fill="both", expand=True, pady=(0, 8))
            try:
                fig_ph.subplots_adjust(bottom=0.20)
            except Exception:
                pass

            # Figure 2: thermo summary.
            fig_th = Figure(figsize=(10, 4), dpi=100)
            self._fig_th = fig_th
            self._ax_th = fig_th.add_subplot(111)
            self._ax_th.set_title("Thermal Properties Summary")
            self._ax_th.set_xlabel("Temperature (K)")
            self._ax_th.set_ylabel("Energy (kJ/mol)")
            self._ax_th.grid(True, alpha=0.3)

            self._canvas_th = FigureCanvasTkAgg(fig_th, master=plots)
            self._canvas_th.get_tk_widget().pack(fill="both", expand=True)
            self._plot_ready = True
        except Exception:
            self._has_mpl = False

    def _log(self, msg: str) -> None:
        self._log_q.put(str(msg))

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self.var_model.set(path)

    def _pick_structure(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if path:
            self.var_structure.set(path)

    def _start(self) -> None:
        model = self.var_model.get().strip()
        struct = self.var_structure.get().strip()
        if not model or not Path(model).is_file():
            return messagebox.showerror("Error", "Please load a valid .pt/.pth model.")
        if not struct or not Path(struct).is_file():
            return messagebox.showerror("Error", "Please load a valid structure file.")
        try:
            total_charge = float(self.var_total_charge.get().strip())
            electric_field = (
                float(self.var_field_x.get().strip()),
                float(self.var_field_y.get().strip()),
                float(self.var_field_z.get().strip()),
            )
            if not np.isfinite(total_charge) or not np.isfinite(np.asarray(electric_field)).all():
                raise ValueError("values must be finite")
        except ValueError as exc:
            return messagebox.showerror("Error", f"Invalid charge or electric field: {exc}")
        device = self.var_device.get()
        model_mode = self.var_model_mode.get()
        spin_policy = self.var_spin_policy.get()
        compile_inference = bool(self.var_compile.get())
        subtract_equilibrium = bool(self.var_subtract_equilibrium.get())

        self._stop = False
        self._status.set("Running phonopy...")
        self._txt.delete("1.0", "end")

        defaults = _Defaults()
        temps = _parse_float_sequence_spec(defaults.thermo_spec, name="phonopy_thermal_temperatures_K", min_len=2)

        def run():
            try:
                res = compute_phonon_thermo_phonopy(
                    model_ckpt=model,
                    structure_path=struct,
                    device=device,
                    e3mu_compile_infer=compile_inference,
                    model_mode=model_mode,
                    total_charge=total_charge,
                    electric_field=electric_field,
                    spin_policy=spin_policy,
                    supercell_matrix=defaults.supercell,
                    displacement_amplitude_A=defaults.amp_A,
                    subtract_equilibrium_forces=subtract_equilibrium,
                    dos_mesh=defaults.dos_mesh,
                    dos_sigma_THz=defaults.dos_sigma_THz,
                    band_npoints=defaults.band_npoints,
                    thermal_temperatures_K=temps,
                    log=self._log,
                    progress=lambda evt: self._evt_q.put(evt),
                    stop_flag=lambda: self._stop,
                )
                self._evt_q.put({"type": "done", "result": res})
            except Exception as e:
                self._evt_q.put({"type": "error", "error": f"{e}\n{traceback.format_exc()}"})

        threading.Thread(target=run, daemon=True).start()

    def _render(self, res: Dict[str, Any]) -> None:
        self._last = res
        formula = res.get("formula", "")
        band = res.get("band", {}) or {}
        fmin = band.get("fmin_THz", float("nan"))
        fmax = band.get("fmax_THz", float("nan"))

        self._txt.insert("end", f"formula: {formula}\n")
        model_info = res.get("model", {}) or {}
        self._txt.insert(
            "end",
            f"model: {model_info.get('model_class', 'unknown')}  "
            f"mode={model_info.get('mode', 'unknown')}  device={model_info.get('device', 'unknown')}\n",
        )
        self._txt.insert(
            "end",
            f"physics: {model_info.get('configured_physics', [])}  "
            f"spin={res.get('spin_treatment', 'disabled')}  "
            f"force evaluations={model_info.get('force_evaluations', 0)}\n",
        )
        self._txt.insert(
            "end",
            f"unit-cell charge: {res.get('input_unitcell_total_charge_e')} e  "
            f"supercell charge: {res.get('supercell_total_charge_e')} e  "
            f"field: {res.get('electric_field_V_per_A')} V/Å\n",
        )
        self._txt.insert("end", f"band freq range (THz): {fmin} .. {fmax}\n")
        self._txt.insert("end", f"supercell: {res.get('supercell_matrix')}\n")
        self._txt.insert("end", f"dos_mesh: {res.get('dos_mesh')}  dos_sigma_THz: {res.get('dos_sigma_THz')}\n")
        self._txt.insert("end", f"band_npoints: {res.get('band_npoints')}\n")
        self._txt.see("end")

        if not (self._has_mpl and self._plot_ready):
            return

        # Phonon spectrum (band + DOS)
        self._ax_band.cla()
        self._ax_dos.cla()
        self._ax_band.set_title("Phonon Band Structure")
        self._ax_band.set_xlabel("k-path distance")
        self._ax_band.set_ylabel("Frequency (THz)")
        self._ax_band.grid(True, alpha=0.3)
        self._ax_dos.set_title("DOS")
        self._ax_dos.set_xlabel("DOS (arb.)")
        self._ax_dos.grid(True, alpha=0.3)
        self._ax_dos.tick_params(labelleft=False)

        def _as_segments(x: Any) -> List[Any]:
            if x is None:
                return []
            if isinstance(x, list):
                if not x:
                    return []
                if isinstance(x[0], (int, float)):
                    return [x]
                return x
            return [x]

        def _normalize_klabel(lbl: Any) -> str:
            if lbl is None:
                return ""
            s = str(lbl).strip()
            if not s:
                return ""
            up = s.upper()
            if up in ("GAMMA", "Γ", "G"):
                return "Γ"
            if "GAMMA" in up:
                return "Γ"
            return s

        def _cumulate_distances(dsegs_in: List[Any]) -> List[np.ndarray]:
            out: List[np.ndarray] = []
            offset = 0.0
            for ds in dsegs_in:
                d = np.asarray(ds, dtype=float).reshape(-1)
                if d.size == 0:
                    continue
                if abs(float(d[0])) < 1e-12:
                    d = d + offset
                elif float(d[0]) < offset - 1e-8:
                    d = d - float(d[0]) + offset
                out.append(d)
                offset = float(d[-1])
            return out

        def _compute_distances_from_qpoints(qsegs: List[Any]) -> List[np.ndarray]:
            out: List[np.ndarray] = []
            for qseg in qsegs:
                q = np.asarray(qseg, dtype=float)
                if q.ndim != 2 or q.shape[1] != 3:
                    continue
                dq = np.diff(q, axis=0)
                d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dq, axis=1))])
                out.append(d)
            return out

        dsegs_any = band.get("distances", None)
        if dsegs_any is None:
            dsegs_any = band.get("distance", None)
        qsegs_any = band.get("qpoints", None)
        fsegs_any = band.get("frequencies_THz", None)
        if fsegs_any is None:
            fsegs_any = band.get("frequencies", None)

        dsegs = _as_segments(dsegs_any)
        qsegs = _as_segments(qsegs_any)
        fsegs = _as_segments(fsegs_any)
        
        if (not dsegs) and qsegs:
            dsegs = [d.tolist() for d in _compute_distances_from_qpoints(qsegs)]
            
        dsegs_np = _cumulate_distances(dsegs)
        
        try:
            for d, fseg in zip(dsegs_np, fsegs):
                arr = np.asarray(fseg, dtype=float)
                if arr.ndim != 2 or d.size == 0:
                    continue
                
                if d.shape[0] != arr.shape[0]:
                    n = min(int(d.shape[0]), int(arr.shape[0]))
                    d = d[:n]
                    arr = arr[:n, :]
                for m in range(arr.shape[1]):
                    self._ax_band.plot(d, arr[:, m], color="C0", lw=1.0)
        except Exception:
            pass
        
        try:
            labels = band.get("labels", []) or []
            path_conn = band.get("path_connections", []) or []
            x_ticks: List[float] = []
            if dsegs_np:
                x_ticks.append(float(dsegs_np[0][0]))
                for ds in dsegs_np:
                    if ds.size:
                        x_ticks.append(float(ds[-1]))
                        
            tick_labels: List[str] = []
            has_labels = bool(labels)
            if has_labels and len(labels) == len(x_ticks):
                tick_labels = [_normalize_klabel(x) for x in labels]
            elif has_labels and len(labels) == (len(x_ticks) - 1):
                tick_labels = [""] + [_normalize_klabel(x) for x in labels]

            if x_ticks:
                for bi, xv in enumerate(x_ticks[1:-1], start=0):
                    draw = True
                    if isinstance(path_conn, list) and bi < len(path_conn):
                        try:
                            draw = not bool(path_conn[bi])
                        except Exception:
                            draw = True
                    if draw:
                        self._ax_band.axvline(float(xv), color="k", lw=0.8, alpha=0.35)
                if tick_labels:
                    self._ax_band.set_xticks(x_ticks)
                    self._ax_band.set_xticklabels(tick_labels)
                self._ax_band.set_xlim(min(x_ticks), max(x_ticks))
                self._ax_band.tick_params(axis="x", labelsize=10)
        except Exception:
            pass

        dos = res.get("dos", {}) or {}
        fpts = np.asarray(dos.get("frequency_points_THz", []), dtype=float)
        td = np.asarray(dos.get("total_dos", []), dtype=float)
        if fpts.size and td.size:
            n = min(int(fpts.size), int(td.size))
            self._ax_dos.plot(td[:n], fpts[:n], color="C1", lw=1.2)

        try:
            self._canvas_ph.draw_idle()
        except Exception:
            pass
        
        # Thermodynamics
        self._ax_th.cla()
        if formula:
            self._ax_th.set_title(f"{formula} Thermal Properties Summary")
        else:
            self._ax_th.set_title("Thermal Properties Summary")
        self._ax_th.set_xlabel("Temperature (K)")
        self._ax_th.set_ylabel("Energy (kJ/mol)")
        self._ax_th.grid(True, alpha=0.3)

        thermal = res.get("thermal", None)
        if isinstance(thermal, dict):
            T = np.asarray(thermal.get("temperatures_K", []), dtype=float)
            F = np.asarray(thermal.get("free_energy_kJ_per_mol", []), dtype=float)
            H = np.asarray(thermal.get("enthalpy_kJ_per_mol", []), dtype=float)
            S = np.asarray(thermal.get("entropy_J_per_K_mol", []), dtype=float)
            Cv = np.asarray(thermal.get("heat_capacity_J_per_K_mol", []), dtype=float)
            n = min(int(T.size), int(F.size), int(H.size), int(S.size), int(Cv.size))
            if n > 0:
                self._ax_th.plot(T[:n], F[:n], color="b", lw=2.0, label="Free Energy")
                self._ax_th.plot(T[:n], H[:n], color="r", lw=2.0, label="Enthalpy")
                self._ax_th.plot(T[:n], S[:n], color="g", lw=2.0, ls="--", label="Entropy")
                self._ax_th.plot(T[:n], Cv[:n], color="m", lw=2.0, ls=":", label="Heat Capacity")
                self._ax_th.legend(loc="upper left")
                self._ax_th.text(
                    0.5,
                    -0.12,
                    "Note: Enthalpy is generated from thermal properties data (phonopy).",
                    transform=self._ax_th.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )
        else:
            self._ax_th.text(0.02, 0.98, "No thermodynamics data.\n(thermal_temperatures_K empty)", transform=self._ax_th.transAxes, va="top")

        try:
            self._canvas_th.draw_idle()
        except Exception:
            pass

    def _tick(self) -> None:
        while True:
            try:
                msg = self._log_q.get_nowait()
            except queue.Empty:
                break
            self._txt.insert("end", msg + "\n")
            self._txt.see("end")

        try:
            while True:
                evt = self._evt_q.get_nowait()
                if evt.get("type") == "done":
                    self._status.set("Done")
                    res = evt.get("result", {})
                    if isinstance(res, dict) and res.get("ok"):
                        self._render(res)
                    self.after(100, self._tick)
                    return
                if evt.get("type") == "error":
                    self._status.set("Error")
                    err = str(evt.get("error", "Unknown error"))
                    self._txt.insert("end", err + "\n")
                    self._txt.see("end")
                    self.after(100, self._tick)
                    return
                if evt.get("type") == "prop":
                    frac = float(evt.get("overall_frac", 0.0))
                    task = str(evt.get("task", ""))
                    stage = str(evt.get("stage", ""))
                    cur = evt.get("current", None)
                    tot = evt.get("total", None)
                    msg = f"{stage} ({frac*100:.1f}%)"
                    if cur is not None and tot is not None:
                        msg = f"{stage} {int(cur)}/{int(tot)} ({frac*100:.1f}%)"
                    if task:
                        msg = f"{task}: {msg}"
                    self._status.set(msg)
        except queue.Empty:
            pass

        self.after(100, self._tick)


if __name__ == "__main__":
    App().mainloop()
