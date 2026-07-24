#!/usr/bin/env python3
"""Offline dataset acquisition, conversion, validation, and release tooling.

This module is intentionally separate from :mod:`E3_miu_GNN`. Training and the
GUI only need the runtime readers in that module; expensive source converters,
corpus builders, release staging, and VASP data-generation helpers live here.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.data import chemical_symbols as ASE_CHEMICAL_SYMBOLS
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.extxyz import key_val_str_to_dict
from ase.neighborlist import neighbor_list

try:
    import h5py
    HAS_H5PY = True
except Exception:
    h5py = None
    HAS_H5PY = False

try:
    import ijson
    HAS_IJSON = True
except Exception:
    ijson = None
    HAS_IJSON = False

try:
    import pyarrow as _pyarrow
    import pyarrow.parquet as _pyarrow_parquet
    HAS_PYARROW = True
except Exception:
    _pyarrow = None
    _pyarrow_parquet = None
    HAS_PYARROW = False


def _load_runtime_core() -> Any:
    """Reuse an already-loaded core module before importing it by name."""
    core_path = Path(__file__).with_name("E3_miu_GNN.py").resolve()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        try:
            if Path(raw_path).resolve() == core_path:
                return module
        except (OSError, TypeError, ValueError):
            continue
    return importlib.import_module("E3_miu_GNN")


_core = _load_runtime_core()
Configuration = _core.Configuration
DatasetKeys = _core.DatasetKeys
COMPOSITE_HDF5_SCHEMA_VERSION = _core.COMPOSITE_HDF5_SCHEMA_VERSION
HDF5_SCHEMA_VERSION = _core.HDF5_SCHEMA_VERSION
OMAT24_BYTE_SHARD_SCHEMA_VERSION = _core.OMAT24_BYTE_SHARD_SCHEMA_VERSION
OMAT24_MATERIALIZED_SCHEMA_VERSION = _core.OMAT24_MATERIALIZED_SCHEMA_VERSION
OMAT24_PACKED_SCHEMA_VERSION = _core.OMAT24_PACKED_SCHEMA_VERSION
HDF5_CANONICAL_ROOT_GROUPS = _core.HDF5_CANONICAL_ROOT_GROUPS
HDF5_METADATA_FIELDS = _core.HDF5_METADATA_FIELDS
HDF5_STRUCTURE_LABELS = _core.HDF5_STRUCTURE_LABELS
HDF5_ATOM_LABELS = _core.HDF5_ATOM_LABELS
HDF5_UNITS = _core.HDF5_UNITS
NEO_MANIFEST_VERSION = "e3mu-neo-manifest-v1"
HDF5_AUXILIARY_PROPERTY_KEYS = {"bec_acoustic_sum_residual_max"}

_now = _core._now
_checkpoint_safe = _core._checkpoint_safe
_open_text = _core._open_text
_parse_properties_spec = _core._parse_properties_spec
_parse_pbc = _core._parse_pbc
_validate_configuration = _core._validate_configuration
_is_hdf5_path = _core._is_hdf5_path
parse_matrix3x3 = _core.parse_matrix3x3
parse_vector3 = _core.parse_vector3
load_extxyz_configurations = _core.load_extxyz_configurations
iter_hdf5_configurations = _core.iter_hdf5_configurations
load_hdf5_configurations = _core.load_hdf5_configurations
sha256_file = _core.sha256_file
stable_split = _core.stable_split
inspect_composite_dataset = _core.inspect_composite_dataset
_resolve_composite_source = _core._resolve_composite_source

_NEO_TIER_LABEL_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "energy_force": ("energy", "forces"),
    "spin": ("spins", "magnetic_moments"),
    "effective_spin_field": ("effective_field",),
    "charge_response": ("charges", "atomic_dipoles"),
    "electric_response": ("field", "dipole", "total_charge"),
    "polarization_dispersion": ("polarizability", "atomic_polarizability", "c6"),
    "born_effective_charge": ("bec",),
}

NEO_HF_TIER_PATHS: Dict[str, str] = {
    "tiny": "canonical/neo_tiny_l1_l2_l3.h5",
    "small": "canonical/neo_small_l1_l2_l3.h5",
    "standard": "canonical/neo_mixed_l1_l2_l3.h5",
    "large": "canonical/neo_large_l1_l2_l3.h5",
    "plus": "canonical/neo_plus_l1_l2_l3.h5",
    "max": "canonical/neo_max_l1_l2_l3.h5",
}
NEO_HF_REQUIRED_DOCUMENTS: Tuple[str, ...] = (
    "README.md",
    "SOURCES_AND_PROCESSING.md",
    "DATA_SCHEMA.md",
    "LICENSES_AND_ATTRIBUTION.md",
    "HUGGINGFACE_UPLOAD.md",
)
_NEO_HF_ALLOWED_RIGHTS_STATUSES = {
    "allowed",
    "allowed_with_attribution",
    "allowed_with_conditions",
}


def assign_stable_group_splits(
    configurations: Sequence[Configuration],
    *,
    train: int = 80,
    val: int = 10,
) -> Dict[str, Any]:
    """Assign group-safe splits and guarantee that train covers every element."""
    groups: Dict[str, List[Configuration]] = {}
    group_elements: Dict[str, set] = {}
    for index, cfg in enumerate(configurations):
        group_id = str(cfg.properties.get("group_id", f"structure-{index}"))
        groups.setdefault(group_id, []).append(cfg)
        group_elements.setdefault(group_id, set()).update(int(z) for z in cfg.atomic_numbers)
    assignments = {
        group_id: stable_split(group_id, train=train, val=val) for group_id in groups
    }
    all_elements = set().union(*group_elements.values()) if group_elements else set()
    train_elements = set().union(
        *(group_elements[group] for group, split_name in assignments.items() if split_name == "train")
    ) if any(value == "train" for value in assignments.values()) else set()
    promoted: List[str] = []
    for element in sorted(all_elements - train_elements):
        candidates = sorted(
            (group for group in groups if element in group_elements[group]),
            key=lambda group: hashlib.sha256(f"coverage|{group}".encode("utf-8")).hexdigest(),
        )
        if not candidates:
            continue
        selected = candidates[0]
        assignments[selected] = "train"
        train_elements.update(group_elements[selected])
        promoted.append(selected)
    for group_id, family in groups.items():
        for cfg in family:
            cfg.properties["split"] = assignments[group_id]
    counts = {
        split_name: sum(len(groups[group]) for group, value in assignments.items() if value == split_name)
        for split_name in ("train", "val", "test")
    }
    return {
        "strategy": "stable_group_hash_with_train_element_coverage",
        "counts": counts,
        "promoted_groups": promoted,
        "train_elements": sorted(train_elements),
        "all_elements": sorted(all_elements),
    }


def download_with_sha256(
    url: str,
    destination: str,
    *,
    expected_sha256: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> str:
    """Download once, then verify content. Existing verified files are reused."""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha256 is None or sha256_file(str(path)) == expected_sha256):
        log(f"[{_now()}] Reusing verified dataset: {path}")
        return str(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "e3mu-dataset/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    actual = sha256_file(str(temporary))
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    temporary.replace(path)
    log(f"[{_now()}] Downloaded {url} -> {path} sha256={actual}")
    return str(path)


def write_hdf5_dataset(
    configurations: Sequence[Configuration],
    path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    if not HAS_H5PY:
        raise RuntimeError("HDF5 support requires h5py")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if output.exists() and overwrite:
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    n_structures = len(configurations)
    atom_ptr = np.zeros(n_structures + 1, dtype=np.int64)
    for index, cfg in enumerate(configurations):
        atom_ptr[index + 1] = atom_ptr[index] + int(len(cfg.atomic_numbers))
    n_atoms = int(atom_ptr[-1])
    atomic_numbers = np.empty((n_atoms,), dtype=np.int16)
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    cells = np.empty((n_structures, 3, 3), dtype=np.float64)
    pbc = np.empty((n_structures, 3), dtype=np.bool_)

    structure_values = {
        name: np.full((n_structures,) + shape, np.nan, dtype=np.float64)
        for name, shape in HDF5_STRUCTURE_LABELS.items()
    }
    structure_masks = {name: np.zeros((n_structures,), dtype=np.bool_) for name in HDF5_STRUCTURE_LABELS}
    atom_values = {
        name: np.full((n_atoms,) + shape, np.nan, dtype=np.float64)
        for name, shape in HDF5_ATOM_LABELS.items()
    }
    atom_masks = {name: np.zeros((n_structures,), dtype=np.bool_) for name in HDF5_ATOM_LABELS}
    meta_fields = {name: [] for name in HDF5_METADATA_FIELDS}

    for structure_index, cfg in enumerate(configurations):
        start, end = int(atom_ptr[structure_index]), int(atom_ptr[structure_index + 1])
        atomic_numbers[start:end] = np.asarray(cfg.atomic_numbers, dtype=np.int16)
        positions[start:end] = np.asarray(cfg.positions, dtype=np.float64)
        cells[structure_index] = np.asarray(
            cfg.cell if cfg.cell is not None else np.eye(3) * 100.0, dtype=np.float64
        ).reshape(3, 3)
        pbc[structure_index] = np.asarray(cfg.pbc, dtype=np.bool_)
        props = dict(cfg.properties or {})
        for name, shape in HDF5_STRUCTURE_LABELS.items():
            if name not in props:
                continue
            structure_values[name][structure_index] = np.asarray(props[name], dtype=np.float64).reshape(shape)
            structure_masks[name][structure_index] = True
        for name, shape in HDF5_ATOM_LABELS.items():
            if name not in props:
                continue
            atom_values[name][start:end] = np.asarray(props[name], dtype=np.float64).reshape((end - start,) + shape)
            atom_masks[name][structure_index] = True
        group_id = str(props.get("group_id", f"structure-{structure_index}"))
        meta_fields["source"].append(str(props.get("source", "unknown")))
        meta_fields["method_id"].append(str(props.get("method_id", "unknown")))
        meta_fields["system_id"].append(str(props.get("system_id", "unknown")))
        meta_fields["group_id"].append(group_id)
        meta_fields["split"].append(str(props.get("split", stable_split(group_id))))
        meta_fields["sample_id"].append(str(props.get("sample_id", f"structure-{structure_index}")))
        meta_fields["parent_id"].append(str(props.get("parent_id", group_id)))
        meta_fields["domain"].append(
            str(props.get("domain", "periodic" if any(bool(x) for x in cfg.pbc) else "molecular"))
        )
        meta_fields["energy_reference"].append(str(props.get("energy_reference", "unknown")))
        meta_fields["provenance_id"].append(str(props.get("provenance_id", "unknown")))
        meta_fields["dataset_role"].append(str(props.get("dataset_role", "unknown")))
        meta_fields["curriculum_role"].append(str(props.get("curriculum_role", "all")))

    with h5py.File(output, "w") as handle:
        handle.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        handle.attrs["created_at"] = _now()
        handle.attrs["units_json"] = json.dumps(HDF5_UNITS, sort_keys=True)
        handle.attrs["metadata_json"] = json.dumps(_checkpoint_safe(metadata or {}), sort_keys=True)
        structures = handle.create_group("structures")
        structures.create_dataset("atom_ptr", data=atom_ptr, compression="gzip")
        structures.create_dataset("atomic_numbers", data=atomic_numbers, compression="gzip")
        structures.create_dataset("positions", data=positions, compression="gzip")
        structures.create_dataset("cell", data=cells, compression="gzip")
        structures.create_dataset("pbc", data=pbc, compression="gzip")
        labels = handle.create_group("labels")
        masks = handle.create_group("masks")
        for name, values in structure_values.items():
            labels.create_dataset(name, data=values, compression="gzip")
            masks.create_dataset(name, data=structure_masks[name], compression="gzip")
        for name, values in atom_values.items():
            labels.create_dataset(name, data=values, compression="gzip")
            masks.create_dataset(name, data=atom_masks[name], compression="gzip")
        metadata_group = handle.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for name, values in meta_fields.items():
            metadata_group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string_dtype)
    return str(output)


def write_hdf5_dataset_stream(
    configurations: Iterable[Configuration],
    path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    chunk_structures: int = 256,
) -> str:
    """Write canonical HDF5 in bounded batches instead of materializing a corpus."""
    if not HAS_H5PY:
        raise RuntimeError("HDF5 support requires h5py")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    batch_size = max(1, int(chunk_structures))
    structure_count = 0
    atom_count = 0
    elements: set = set()
    split_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    label_counts = {
        name: 0 for name in itertools.chain(HDF5_STRUCTURE_LABELS, HDF5_ATOM_LABELS)
    }

    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        handle.attrs["created_at"] = _now()
        handle.attrs["units_json"] = json.dumps(HDF5_UNITS, sort_keys=True)
        structures = handle.create_group("structures")
        atom_ptr_ds = structures.create_dataset(
            "atom_ptr", data=np.asarray([0], dtype=np.int64), maxshape=(None,), chunks=True
        )
        atomic_numbers_ds = structures.create_dataset(
            "atomic_numbers", shape=(0,), maxshape=(None,), dtype=np.int16,
            chunks=True, compression="gzip",
        )
        positions_ds = structures.create_dataset(
            "positions", shape=(0, 3), maxshape=(None, 3), dtype=np.float64,
            chunks=True, compression="gzip",
        )
        cell_ds = structures.create_dataset(
            "cell", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype=np.float64,
            chunks=True, compression="gzip",
        )
        pbc_ds = structures.create_dataset(
            "pbc", shape=(0, 3), maxshape=(None, 3), dtype=np.bool_,
            chunks=True, compression="gzip",
        )
        labels = handle.create_group("labels")
        masks = handle.create_group("masks")
        structure_label_ds = {
            name: labels.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape,
                dtype=np.float64, chunks=True, compression="gzip", fillvalue=np.nan,
            )
            for name, shape in HDF5_STRUCTURE_LABELS.items()
        }
        atom_label_ds = {
            name: labels.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape,
                dtype=np.float64, chunks=True, compression="gzip", fillvalue=np.nan,
            )
            for name, shape in HDF5_ATOM_LABELS.items()
        }
        mask_ds = {
            name: masks.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=np.bool_,
                chunks=True, compression="gzip", fillvalue=False,
            )
            for name in itertools.chain(HDF5_STRUCTURE_LABELS, HDF5_ATOM_LABELS)
        }
        metadata_group = handle.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata_ds = {
            name: metadata_group.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=True
            )
            for name in HDF5_METADATA_FIELDS
        }

        iterator = iter(configurations)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            for offset, cfg in enumerate(batch):
                _validate_configuration(cfg, context=f"HDF5 stream structure {structure_count + offset}")
            active_structure_labels = {
                name
                for cfg in batch
                for name in HDF5_STRUCTURE_LABELS
                if name in (cfg.properties or {})
                and float(cfg.property_weights.get(name, 1.0)) > 0.0
            }
            active_atom_labels = {
                name
                for cfg in batch
                for name in HDF5_ATOM_LABELS
                if name in (cfg.properties or {})
                and float(cfg.property_weights.get(name, 1.0)) > 0.0
            }
            batch_structures = len(batch)
            batch_atoms = sum(len(np.asarray(cfg.atomic_numbers).reshape(-1)) for cfg in batch)
            structure_start, structure_end = structure_count, structure_count + batch_structures
            atom_start, atom_end = atom_count, atom_count + batch_atoms

            atom_ptr_ds.resize((structure_end + 1,))
            atomic_numbers_ds.resize((atom_end,))
            positions_ds.resize((atom_end, 3))
            cell_ds.resize((structure_end, 3, 3))
            pbc_ds.resize((structure_end, 3))
            for name, shape in HDF5_STRUCTURE_LABELS.items():
                structure_label_ds[name].resize((structure_end,) + shape)
            for name, shape in HDF5_ATOM_LABELS.items():
                atom_label_ds[name].resize((atom_end,) + shape)
            for dataset in mask_ds.values():
                dataset.resize((structure_end,))
            for dataset in metadata_ds.values():
                dataset.resize((structure_end,))

            cursor = atom_start
            pointer_values = np.empty((batch_structures,), dtype=np.int64)
            batch_numbers = np.empty((batch_atoms,), dtype=np.int16)
            batch_positions = np.empty((batch_atoms, 3), dtype=np.float64)
            batch_cells = np.empty((batch_structures, 3, 3), dtype=np.float64)
            batch_pbc = np.empty((batch_structures, 3), dtype=np.bool_)
            batch_structure_values = {
                name: np.full((batch_structures,) + shape, np.nan, dtype=np.float64)
                for name, shape in HDF5_STRUCTURE_LABELS.items()
                if name in active_structure_labels
            }
            batch_atom_values = {
                name: np.full((batch_atoms,) + shape, np.nan, dtype=np.float64)
                for name, shape in HDF5_ATOM_LABELS.items()
                if name in active_atom_labels
            }
            batch_masks = {
                name: np.zeros((batch_structures,), dtype=np.bool_)
                for name in active_structure_labels | active_atom_labels
            }
            metadata_values = {name: [] for name in HDF5_METADATA_FIELDS}
            for local_index, cfg in enumerate(batch):
                numbers = np.asarray(cfg.atomic_numbers, dtype=np.int16).reshape(-1)
                next_cursor = cursor + len(numbers)
                local_atom_start = cursor - atom_start
                local_atom_end = next_cursor - atom_start
                batch_numbers[local_atom_start:local_atom_end] = numbers
                batch_positions[local_atom_start:local_atom_end] = np.asarray(
                    cfg.positions, dtype=np.float64
                )
                batch_cells[local_index] = np.asarray(cfg.cell, dtype=np.float64).reshape(3, 3)
                batch_pbc[local_index] = np.asarray(cfg.pbc, dtype=np.bool_)
                pointer_values[local_index] = next_cursor
                elements.update(int(value) for value in numbers)
                props = dict(cfg.properties or {})
                for name, shape in HDF5_STRUCTURE_LABELS.items():
                    if name not in props or float(cfg.property_weights.get(name, 1.0)) <= 0.0:
                        continue
                    batch_structure_values[name][local_index] = np.asarray(
                        props[name], dtype=np.float64
                    ).reshape(shape)
                    batch_masks[name][local_index] = True
                    label_counts[name] += 1
                for name, shape in HDF5_ATOM_LABELS.items():
                    if name not in props or float(cfg.property_weights.get(name, 1.0)) <= 0.0:
                        continue
                    batch_atom_values[name][local_atom_start:local_atom_end] = np.asarray(
                        props[name], dtype=np.float64
                    ).reshape((len(numbers),) + shape)
                    batch_masks[name][local_index] = True
                    label_counts[name] += 1
                global_index = structure_start + local_index
                group_id = str(props.get("group_id", f"structure-{global_index}"))
                split_name = str(props.get("split", stable_split(group_id)))
                source_name = str(props.get("source", "unknown"))
                defaults = {
                    "source": source_name,
                    "method_id": str(props.get("method_id", "unknown")),
                    "system_id": str(props.get("system_id", "unknown")),
                    "group_id": group_id,
                    "split": split_name,
                    "sample_id": str(props.get("sample_id", f"structure-{global_index}")),
                    "parent_id": str(props.get("parent_id", group_id)),
                    "domain": str(
                        props.get(
                            "domain",
                            "periodic" if any(bool(x) for x in cfg.pbc) else "molecular",
                        )
                    ),
                    "energy_reference": str(props.get("energy_reference", "unknown")),
                    "provenance_id": str(props.get("provenance_id", "unknown")),
                    "dataset_role": str(props.get("dataset_role", "unknown")),
                    "curriculum_role": str(props.get("curriculum_role", "all")),
                }
                for name in HDF5_METADATA_FIELDS:
                    metadata_values[name].append(defaults[name])
                split_counts[split_name] = split_counts.get(split_name, 0) + 1
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                cursor = next_cursor
            atomic_numbers_ds[atom_start:atom_end] = batch_numbers
            positions_ds[atom_start:atom_end] = batch_positions
            cell_ds[structure_start:structure_end] = batch_cells
            pbc_ds[structure_start:structure_end] = batch_pbc
            atom_ptr_ds[structure_start + 1 : structure_end + 1] = pointer_values
            for name, values in batch_structure_values.items():
                structure_label_ds[name][structure_start:structure_end] = values
            for name, values in batch_atom_values.items():
                atom_label_ds[name][atom_start:atom_end] = values
            for name, values in batch_masks.items():
                mask_ds[name][structure_start:structure_end] = values
            for name, values in metadata_values.items():
                metadata_ds[name][structure_start:structure_end] = np.asarray(values, dtype=object)
            structure_count = structure_end
            atom_count = atom_end

        metadata_payload = dict(metadata or {})
        metadata_payload["writer_summary"] = {
            "structures": structure_count,
            "atoms": atom_count,
            "elements": sorted(elements),
            "labels": label_counts,
            "splits": split_counts,
            "sources": source_counts,
        }
        handle.attrs["metadata_json"] = json.dumps(
            _checkpoint_safe(metadata_payload), sort_keys=True
        )
    temporary.replace(output)
    return str(output)


def extxyz_to_hdf5(
    input_path: str,
    output_path: str,
    *,
    keys: Optional[DatasetKeys] = None,
    overwrite: bool = False,
    max_frames: Optional[int] = None,
) -> str:
    configs, _fields = load_extxyz_configurations(
        input_path,
        keys or DatasetKeys(),
        require_energy=False,
        require_forces=False,
        require_field=False,
        max_frames=max_frames,
    )
    split_diagnostics = assign_stable_group_splits(configs)
    return write_hdf5_dataset(
        configs,
        output_path,
        metadata={
            "source_path": str(Path(input_path).resolve()),
            "source_sha256": sha256_file(input_path),
            "split_diagnostics": split_diagnostics,
        },
        overwrite=overwrite,
    )


def rebuild_qm7x_hdf5(
    raw_path: str,
    output_path: str,
    *,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """Rebuild QM7-X without the undocumented polarizability scaling in old extXYZ files."""
    if not HAS_H5PY:
        raise RuntimeError("QM7-X conversion requires h5py")
    bohr_to_ang = 0.529177210903
    hartree_bohr6_to_ev_ang6 = 27.211386245988 * bohr_to_ang**6
    raw_file = Path(raw_path).expanduser().resolve()

    def sample_references(raw: Any) -> List[Tuple[str, str]]:
        molecule_ids = sorted(
            (str(name) for name in raw.keys()),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        if max_frames is None:
            return [
                (molecule_id, str(conformer_id))
                for molecule_id in molecule_ids
                for conformer_id in sorted(raw[molecule_id].keys())
            ]
        conformer_lists = {
            molecule_id: sorted(str(name) for name in raw[molecule_id].keys())
            for molecule_id in molecule_ids
        }
        references: List[Tuple[str, str]] = []
        conformer_index = 0
        while len(references) < max(0, int(max_frames)):
            added = False
            for molecule_id in molecule_ids:
                conformers = conformer_lists[molecule_id]
                if conformer_index >= len(conformers):
                    continue
                references.append((molecule_id, conformers[conformer_index]))
                added = True
                if len(references) >= int(max_frames):
                    break
            if not added:
                break
            conformer_index += 1
        return references

    def configurations() -> Iterable[Configuration]:
        with h5py.File(raw_file, "r") as raw:
            for molecule_id, conformer_id in sample_references(raw):
                sample = raw[molecule_id][conformer_id]
                base_energy = float(np.asarray(sample["ePBE0"]).reshape(-1)[0])
                dispersion_energy = float(np.asarray(sample["eMBD"]).reshape(-1)[0])
                total_energy = float(np.asarray(sample["ePBE0+MBD"]).reshape(-1)[0])
                props: Dict[str, Any] = {
                    "energy": total_energy,
                    "energy_base": base_energy,
                    "energy_dispersion": dispersion_energy,
                    "forces": np.asarray(sample["totFOR"], dtype=float),
                    "forces_base": np.asarray(sample["pbe0FOR"], dtype=float),
                    "forces_dispersion": np.asarray(sample["vdwFOR"], dtype=float),
                    "field": np.zeros(3, dtype=float),
                    "dipole": np.asarray(sample["vDIP"], dtype=float).reshape(3),
                    "polarizability": np.asarray(sample["mTPOL"], dtype=float).reshape(3, 3)
                    * bohr_to_ang**3,
                    "charges": np.asarray(sample["hCHG"], dtype=float).reshape(-1),
                    "atomic_dipoles": np.asarray(sample["hVDIP"], dtype=float).reshape(-1, 3)
                    * bohr_to_ang,
                    "atomic_polarizability": np.eye(3)[None, :, :]
                    * (
                        np.asarray(sample["atPOL"], dtype=float).reshape(-1, 1, 1)
                        * bohr_to_ang**3
                    ),
                    "c6": np.asarray(sample["atC6"], dtype=float).reshape(-1)
                    * hartree_bohr6_to_ev_ang6,
                    "total_charge": 0.0,
                    "source": "QM7-X",
                    "method_id": "PBE0+MBD",
                    "system_id": molecule_id,
                    "group_id": f"QM7-X:{molecule_id}",
                    "sample_id": f"QM7-X:{molecule_id}:{conformer_id}",
                    "parent_id": molecule_id,
                    "domain": "molecular",
                    "energy_reference": "QM7-X:PBE0+MBD:absolute",
                    "provenance_id": f"QM7-X:{molecule_id}/{conformer_id}",
                }
                props["split"] = stable_split(props["group_id"])
                cfg = Configuration(
                    atomic_numbers=np.asarray(sample["atNUM"], dtype=int).reshape(-1),
                    positions=np.asarray(sample["atXYZ"], dtype=float).reshape(-1, 3),
                    properties=props,
                    property_weights={
                        name: 1.0
                        for name in props
                        if name in HDF5_STRUCTURE_LABELS or name in HDF5_ATOM_LABELS
                    },
                    cell=np.eye(3) * 100.0,
                    pbc=(False, False, False),
                    config_type="QM7-X",
                    head="QM7-X:PBE0+MBD",
                )
                _validate_configuration(cfg, context=f"QM7-X/{molecule_id}/{conformer_id}")
                yield cfg

    return write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "QM7-X",
            "raw_path": str(raw_file),
            "raw_sha256": sha256_file(str(raw_file)),
            "polarizability_conversion": "bohr^3 to angstrom^3",
            "atomic_dipole_conversion": "hVDIP e*bohr multiplied by 0.529177210903",
            "c6_conversion": "Hartree*bohr^6 to eV*angstrom^6",
            "energy_semantics": "energy=ePBE0+MBD; energy_base=ePBE0; energy_dispersion=eMBD",
            "force_semantics": "forces=totFOR; forces_base=pbe0FOR; forces_dispersion=vdwFOR",
            "split_strategy": "stable hash of QM7-X molecule id",
        },
        overwrite=overwrite,
    )


def _iter_extxyz_records(
    path: str,
    *,
    start_index: int = 0,
    end_index: Optional[int] = None,
    selected_indices: Optional[set] = None,
) -> Iterable[Tuple[int, Dict[str, Any], np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """Yield selected extXYZ frames without retaining the rest of the file."""
    first = max(0, int(start_index))
    last = None if end_index is None else max(first, int(end_index))
    selected = None if selected_indices is None else {int(value) for value in selected_indices}
    with _open_text(str(Path(path).expanduser())) as handle:
        frame_index = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            atom_count = int(line)
            comment = handle.readline()
            if not comment:
                raise EOFError(f"Truncated extXYZ comment at frame {frame_index}: {path}")
            wanted = frame_index >= first and (last is None or frame_index < last)
            if wanted and selected is not None:
                wanted = frame_index in selected
            if not wanted:
                for _ in range(atom_count):
                    if not handle.readline():
                        raise EOFError(
                            f"Truncated extXYZ atom block at frame {frame_index}: {path}"
                        )
                frame_index += 1
                if last is not None and frame_index >= last:
                    break
                continue
            rows = [handle.readline() for _ in range(atom_count)]
            if any(not row for row in rows):
                raise EOFError(f"Truncated extXYZ atom block at frame {frame_index}: {path}")
            if wanted:
                try:
                    info = key_val_str_to_dict(comment.strip()) if "=" in comment else {}
                except Exception as exc:
                    raise ValueError(f"Cannot parse extXYZ comment at frame {frame_index}") from exc
                properties = _parse_properties_spec(
                    info.get("Properties", "species:S:1:pos:R:3")
                )
                symbols: List[str] = []
                explicit_numbers: List[int] = []
                positions = np.zeros((atom_count, 3), dtype=float)
                arrays: Dict[str, np.ndarray] = {}
                aliases: Dict[str, Tuple[set, Optional[int]]] = {
                    "forces": ({"forces", "force", "F"}, 3),
                    "charges": ({"charges", "charge", "q", "hCHG"}, 1),
                    "atomic_dipoles": ({"atomic_dipoles", "hVDIP"}, 3),
                    "atomic_polarizability": ({"atomic_polarizability", "atPOL"}, None),
                    "c6": ({"c6", "C6", "atC6"}, 1),
                    "bec": ({"bec", "BEC"}, 9),
                    "spins": ({"spins", "spin"}, 3),
                    "magnetic_moments": ({"magmoms", "magmom", "magnetic_moments"}, None),
                    "effective_field": ({"effective_field", "spin_field"}, 3),
                }
                for atom_index, row in enumerate(rows):
                    tokens = row.split()
                    cursor = 0
                    for name, _type, width in properties:
                        values = tokens[cursor : cursor + width]
                        cursor += width
                        if len(values) != width:
                            raise ValueError(
                                f"Frame {frame_index} atom {atom_index}: short {name!r} field"
                            )
                        if name in ("species", "symbol") and width == 1:
                            symbols.append(str(values[0]).capitalize())
                        elif name in ("Z", "atomic_number", "atomic_numbers") and width == 1:
                            explicit_numbers.append(int(float(values[0])))
                        elif name in ("pos", "positions") and width == 3:
                            positions[atom_index] = [float(value) for value in values]
                        else:
                            for canonical, (names, expected_width) in aliases.items():
                                if name not in names or (
                                    expected_width is not None and width != expected_width
                                ):
                                    continue
                                if canonical not in arrays:
                                    arrays[canonical] = np.zeros((atom_count, width), dtype=float)
                                arrays[canonical][atom_index] = [float(value) for value in values]
                                break
                    if cursor != len(tokens):
                        raise ValueError(
                            f"Frame {frame_index} atom {atom_index}: Properties consumes "
                            f"{cursor} of {len(tokens)} columns"
                        )
                if explicit_numbers:
                    numbers = np.asarray(explicit_numbers, dtype=int)
                else:
                    numbers = np.asarray(
                        [ASE_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=int
                    )
                if "bec" in arrays:
                    arrays["bec"] = arrays["bec"].reshape(atom_count, 3, 3)
                if "atomic_polarizability" in arrays:
                    value = arrays["atomic_polarizability"]
                    if value.shape[1] == 9:
                        arrays["atomic_polarizability"] = value.reshape(atom_count, 3, 3)
                    elif value.shape[1] == 1:
                        arrays["atomic_polarizability"] = (
                            np.eye(3)[None, :, :] * value.reshape(atom_count, 1, 1)
                        )
                yield frame_index, info, numbers, positions, arrays
            frame_index += 1
            if last is not None and frame_index >= last:
                break


def _configuration_from_extxyz_record(
    frame_index: int,
    info: Dict[str, Any],
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    arrays: Dict[str, np.ndarray],
) -> Configuration:
    props: Dict[str, Any] = {}
    weights: Dict[str, float] = {}
    if "energy" in info:
        props["energy"] = float(info["energy"])
        weights["energy"] = 1.0
    if "forces" in arrays:
        props["forces"] = np.asarray(arrays["forces"], dtype=float).reshape(-1, 3)
        weights["forces"] = 1.0
    if "field" in info:
        props["field"] = parse_vector3(info["field"], name="field")
        weights["field"] = 1.0
    if "dipole" in info:
        props["dipole"] = parse_vector3(info["dipole"], name="dipole")
        weights["dipole"] = 1.0
    if "polarizability" in info:
        props["polarizability"] = parse_matrix3x3(
            info["polarizability"], name="polarizability"
        )
        weights["polarizability"] = 1.0
    if "total_charge" in info:
        props["total_charge"] = float(info["total_charge"])
        weights["total_charge"] = 1.0
    for name in HDF5_ATOM_LABELS:
        if name in arrays:
            value = np.asarray(arrays[name], dtype=float)
            if name in ("charges", "c6"):
                value = value.reshape(-1)
            elif name == "magnetic_moments" and value.shape[1] == 1:
                value = np.pad(value, ((0, 0), (0, 2)))
            props[name] = value
            weights[name] = 1.0
    if "Lattice" in info:
        cell = np.asarray(info["Lattice"], dtype=float).reshape(3, 3)
        pbc = _parse_pbc(info.get("pbc", (True, True, True)))
    else:
        cell = np.eye(3, dtype=float) * 100.0
        pbc = _parse_pbc(info.get("pbc", (False, False, False)))
    return Configuration(
        atomic_numbers=np.asarray(atomic_numbers, dtype=int).reshape(-1),
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        properties=props,
        property_weights=weights,
        cell=cell,
        pbc=pbc,
        config_type="extXYZ",
        head="Default",
    )


def _geometry_group_hash(cfg: Configuration, *, decimals: int = 7) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cfg.atomic_numbers, dtype=np.int16).reshape(-1).tobytes())
    digest.update(np.round(np.asarray(cfg.positions, dtype=np.float64), decimals).tobytes())
    digest.update(np.round(np.asarray(cfg.cell, dtype=np.float64), decimals).tobytes())
    digest.update(np.asarray(cfg.pbc, dtype=np.bool_).tobytes())
    return digest.hexdigest()


def _so3lr_parent_id(dataset_name: str, sample_id: str) -> str:
    if dataset_name == "DES15K":
        match = re.match(r"^(DES15K-\d+)(?:-\d+)?$", sample_id)
        return match.group(1) if match else sample_id
    if dataset_name == "TorsionNet500":
        match = re.match(r"^(fragment_\d+)_conf_\d+$", sample_id)
        return match.group(1) if match else sample_id
    if dataset_name == "SPICE_dipeptides":
        match = re.match(r"^([A-Z]{3}-[A-Z]{3})\d+$", sample_id)
        return match.group(1) if match else sample_id
    return sample_id


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite float, treating dataset sentinels such as ``na`` as absent."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _iter_json_array(path: str, *, chunk_size: int = 1024 * 1024) -> Iterable[Dict[str, Any]]:
    """Stream a top-level JSON array while accepting non-standard NaN tokens.

    Several official materials archives are large JSON arrays emitted by the
    standard Python encoder with ``allow_nan=True``.  ``ijson`` rejects those
    files, whereas ``json.JSONDecoder`` applies the same NaN semantics as the
    producer and can still be used with bounded memory through ``raw_decode``.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        while True:
            if not eof:
                part = handle.read(max(4096, int(chunk_size)))
                eof = not part
                buffer += part
            cursor = 0
            if not started:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON archive: {path}")
                    continue
                if buffer[cursor] != "[":
                    raise ValueError(f"Expected a top-level JSON array: {path}")
                cursor += 1
                started = True
            decoded = False
            while True:
                while cursor < len(buffer) and (
                    buffer[cursor].isspace() or buffer[cursor] == ","
                ):
                    cursor += 1
                if cursor < len(buffer) and buffer[cursor] == "]":
                    return
                if cursor >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in array: {path}")
                yield value
                cursor = end
                decoded = True
            buffer = buffer[cursor:]
            if eof:
                raise ValueError(f"Truncated JSON archive: {path}")
            if not decoded and len(buffer) > 128 * 1024 * 1024:
                raise ValueError(f"One JSON record is unexpectedly large: {path}")


def extract_verified_zip_member(
    archive_path: str,
    output_directory: str,
    *,
    expected_md5: Optional[str] = None,
    member: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Verify and safely extract one regular file from a downloaded ZIP."""
    archive = Path(archive_path).expanduser().resolve()
    if expected_md5:
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest.lower() != str(expected_md5).lower():
            raise ValueError(
                f"MD5 mismatch for {archive}: expected {expected_md5}, got {digest}"
            )
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zipped:
        regular = [
            info for info in zipped.infolist()
            if not info.is_dir() and not Path(info.filename).is_absolute()
        ]
        if member is None:
            if len(regular) != 1:
                raise ValueError("ZIP contains multiple files; specify the member name")
            info = regular[0]
        else:
            matches = [info for info in regular if info.filename == member]
            if len(matches) != 1:
                raise KeyError(f"ZIP member not found: {member}")
            info = matches[0]
        if ".." in Path(info.filename).parts:
            raise ValueError(f"Unsafe ZIP member path: {info.filename}")
        destination = root / Path(info.filename).name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite extracted file: {destination}")
        temporary = destination.with_name(destination.name + ".extracting")
        with zipped.open(info, "r") as source, temporary.open("wb") as target:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        temporary.replace(destination)
    return str(destination)


def scan_jarvis_multi_element_candidates(
    raw_json: str,
    index_path: str,
    *,
    min_elements: int = 2,
    max_atoms: int = 160,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Index physically valid complex structures in the official JARVIS-DFT JSON."""
    output = Path(index_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    scanned = 0
    eligible = 0
    strata: Dict[str, int] = {}
    elements: Dict[str, int] = {}
    property_counts: Dict[str, int] = {}
    with temporary.open("w", encoding="utf-8") as writer:
        for record in _iter_json_array(raw_json):
            scanned += 1
            atoms = dict(record.get("atoms") or {})
            symbols = [str(value) for value in atoms.get("elements") or []]
            coordinates = np.asarray(atoms.get("coords") or [], dtype=float)
            lattice = np.asarray(atoms.get("lattice_mat") or [], dtype=float)
            unique = sorted(set(symbols), key=lambda value: ASE_ATOMIC_NUMBERS.get(value, 999))
            energy_per_atom = _finite_float(record.get("optb88vdw_total_energy"))
            if (
                len(unique) < int(min_elements)
                or not symbols
                or len(symbols) > int(max_atoms)
                or any(value not in ASE_ATOMIC_NUMBERS for value in unique)
                or coordinates.shape != (len(symbols), 3)
                or lattice.shape != (3, 3)
                or not np.isfinite(coordinates).all()
                or not np.isfinite(lattice).all()
                or abs(float(np.linalg.det(lattice))) <= 1e-8
                or energy_per_atom is None
            ):
                continue
            complexity = (
                "binary" if len(unique) == 2
                else "ternary" if len(unique) == 3
                else "quaternary+"
            )
            size_class = (
                "small" if len(symbols) <= 24
                else "medium" if len(symbols) <= 64
                else "large"
            )
            stratum = f"{complexity}|{size_class}"
            available = ["energy"]
            if _finite_float(record.get("magmom_outcar")) is not None:
                available.append("total_magnetic_moment_metadata")
            if all(_finite_float(record.get(name)) is not None for name in ("epsx", "epsy", "epsz")):
                available.append("dielectric_diagonal_metadata")
            descriptor = {
                "jid": str(record.get("jid")),
                "formula": str(record.get("formula", "unknown")),
                "atom_count": len(symbols),
                "elements": unique,
                "element_count": len(unique),
                "stratum": stratum,
                "available_properties": available,
                "rank": hashlib.sha256(
                    f"neo-jarvis|{record.get('jid')}|{record.get('formula')}".encode()
                ).hexdigest(),
            }
            writer.write(json.dumps(descriptor, sort_keys=True) + "\n")
            eligible += 1
            strata[stratum] = strata.get(stratum, 0) + 1
            for symbol in unique:
                elements[symbol] = elements.get(symbol, 0) + 1
            for name in available:
                property_counts[name] = property_counts.get(name, 0) + 1
            if scanned % 20000 == 0:
                log(f"[{_now()}] JARVIS scan: records={scanned} eligible={eligible}")
    temporary.replace(output)
    return {
        "source": str(Path(raw_json).expanduser().resolve()),
        "records_scanned": scanned,
        "eligible_multi_element_structures": eligible,
        "min_elements": int(min_elements),
        "max_atoms": int(max_atoms),
        "strata": dict(sorted(strata.items())),
        "elements": dict(sorted(elements.items(), key=lambda item: ASE_ATOMIC_NUMBERS[item[0]])),
        "available_property_counts": dict(sorted(property_counts.items())),
        "index_path": str(output),
        "index_sha256": sha256_file(str(output)),
    }


def select_jarvis_multi_element_candidates(
    index_path: str,
    selection_path: str,
    *,
    target_structures: int = 24000,
) -> Dict[str, Any]:
    """Select a deterministic complexity/size-balanced JARVIS subset."""
    candidates = [
        json.loads(line)
        for line in Path(index_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = min(max(0, int(target_structures)), len(candidates))
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    element_frequency: Dict[str, int] = {}
    for item in candidates:
        for symbol in item["elements"]:
            element_frequency[symbol] = element_frequency.get(symbol, 0) + 1
    for symbol in sorted(
        element_frequency,
        key=lambda value: (element_frequency[value], ASE_ATOMIC_NUMBERS[value]),
    ):
        choices = sorted(
            (item for item in candidates if symbol in item["elements"]),
            key=lambda item: str(item["rank"]),
        )
        if choices and choices[0]["jid"] not in selected_ids:
            selected.append(choices[0])
            selected_ids.add(choices[0]["jid"])

    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        if item["jid"] not in selected_ids:
            by_stratum.setdefault(str(item["stratum"]), []).append(item)
    for values in by_stratum.values():
        values.sort(key=lambda item: str(item["rank"]))
    strata = sorted(by_stratum)
    cursors = {name: 0 for name in strata}
    while len(selected) < target:
        progress = False
        for stratum in strata:
            cursor = cursors[stratum]
            values = by_stratum[stratum]
            while cursor < len(values) and values[cursor]["jid"] in selected_ids:
                cursor += 1
            cursors[stratum] = cursor
            if cursor >= len(values):
                continue
            item = values[cursor]
            cursors[stratum] = cursor + 1
            selected.append(item)
            selected_ids.add(item["jid"])
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    selected.sort(key=lambda item: str(item["jid"]))
    output = Path(selection_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in selected),
        encoding="utf-8",
    )
    return {
        "selected_structures": len(selected),
        "selected_materials": len(selected_ids),
        "selection_path": str(output),
        "selection_sha256": sha256_file(str(output)),
        "strata": dict(sorted(collections.Counter(item["stratum"] for item in selected).items())),
        "elements": sorted(
            {symbol for item in selected for symbol in item["elements"]},
            key=lambda value: ASE_ATOMIC_NUMBERS[value],
        ),
    }


def rebuild_jarvis_multi_element_hdf5(
    raw_json: str,
    selection_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Build a JARVIS OptB88vdW energy shard without inventing atom-wise labels."""
    selected = {
        str(item["jid"]): item
        for item in (
            json.loads(line)
            for line in Path(selection_path).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    decoded = 0

    def configurations() -> Iterable[Configuration]:
        nonlocal decoded
        for record in _iter_json_array(raw_json):
            jid = str(record.get("jid"))
            descriptor = selected.get(jid)
            if descriptor is None:
                continue
            atoms = dict(record["atoms"])
            symbols = [str(value) for value in atoms["elements"]]
            energy_per_atom = _finite_float(record.get("optb88vdw_total_energy"))
            if energy_per_atom is None:
                raise ValueError(f"Selected JARVIS record has no finite energy: {jid}")
            props: Dict[str, Any] = {
                # JARVIS stores OptB88vdW energy per atom; the model target is total energy.
                "energy": float(energy_per_atom) * len(symbols),
                "source": "JARVIS-DFT-3D-v2025.09",
                "method_id": "VASP-OptB88vdW",
                "system_id": jid,
                "group_id": f"JARVIS:{jid}",
                "sample_id": f"JARVIS:{jid}",
                "parent_id": jid,
                "domain": "periodic",
                "energy_reference": "JARVIS:OptB88vdW:absolute",
                "provenance_id": f"figshare:10.6084/m9.figshare.6815699.v11#{jid}",
            }
            props["split"] = stable_split(props["group_id"])
            cfg = Configuration(
                atomic_numbers=np.asarray([ASE_ATOMIC_NUMBERS[value] for value in symbols], dtype=int),
                positions=np.asarray(atoms["coords"], dtype=float).reshape(-1, 3),
                properties=props,
                property_weights={"energy": 1.0},
                cell=np.asarray(atoms["lattice_mat"], dtype=float).reshape(3, 3),
                pbc=(True, True, True),
                config_type="JARVIS-DFT-3D",
                head="JARVIS:VASP-OptB88vdW",
            )
            _validate_configuration(cfg, context=f"JARVIS/{jid}")
            decoded += 1
            yield cfg
        if decoded != len(selected):
            raise ValueError(f"Decoded {decoded} JARVIS structures, expected {len(selected)}")

    return write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "JARVIS-DFT 3D complex multi-element subset",
            "raw_path": str(Path(raw_json).expanduser().resolve()),
            "raw_sha256": sha256_file(raw_json),
            "selection_path": str(Path(selection_path).expanduser().resolve()),
            "selection_sha256": sha256_file(selection_path),
            "figshare_article": "10.6084/m9.figshare.6815699.v11",
            "figshare_file_id": 64391379,
            "license": "CC-BY-4.0",
            "energy_conversion": "optb88vdw_total_energy (eV/atom) multiplied by atom count",
            "explicitly_absent_labels": [
                "forces", "charges", "polarizability", "magnetic_moments", "spins"
            ],
            "metadata_not_mapped_to_model_targets": [
                "magmom_outcar (structure total, not atom-wise)",
                "epsx/epsy/epsz (dimensionless dielectric response, not molecular polarizability)",
            ],
        },
        overwrite=overwrite,
    )


def select_jarvis_dfpt_candidates(
    raw_json: str,
    selection_path: str,
    *,
    target_structures: int = 300,
    min_elements: int = 2,
    max_atoms: int = 80,
) -> Dict[str, Any]:
    """Select complex JARVIS records that expose an official DFPT raw archive."""
    candidates: List[Dict[str, Any]] = []
    for record in _iter_json_array(raw_json):
        atoms = dict(record.get("atoms") or {})
        symbols = [str(value) for value in atoms.get("elements") or []]
        unique = sorted(set(symbols), key=lambda value: ASE_ATOMIC_NUMBERS.get(value, 999))
        if (
            len(unique) < int(min_elements)
            or not symbols
            or len(symbols) > int(max_atoms)
            or any(value not in ASE_ATOMIC_NUMBERS for value in unique)
        ):
            continue
        archive_url = None
        archive_name = None
        for raw_file in record.get("raw_files") or []:
            if not isinstance(raw_file, str) or not raw_file.startswith("DFPT,"):
                continue
            parts = raw_file.split(",", 2)
            if len(parts) == 3 and parts[2].startswith("https://"):
                archive_name, archive_url = parts[1], parts[2]
                break
        if archive_url is None:
            continue
        complexity = (
            "binary" if len(unique) == 2
            else "ternary" if len(unique) == 3
            else "quaternary+"
        )
        size_class = "small" if len(symbols) <= 24 else "medium"
        candidates.append({
            "jid": str(record["jid"]),
            "formula": str(record.get("formula", "unknown")),
            "atom_count": len(symbols),
            "elements": unique,
            "element_count": len(unique),
            "stratum": f"{complexity}|{size_class}",
            "archive_name": str(archive_name),
            "archive_url": str(archive_url),
            "rank": hashlib.sha256(f"neo-jarvis-dfpt|{record['jid']}".encode()).hexdigest(),
        })
    target = min(max(0, int(target_structures)), len(candidates))
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        by_stratum.setdefault(item["stratum"], []).append(item)
    for values in by_stratum.values():
        values.sort(key=lambda item: str(item["rank"]))
    selected: List[Dict[str, Any]] = []
    cursor = 0
    strata = sorted(by_stratum)
    while len(selected) < target:
        progress = False
        for stratum in strata:
            values = by_stratum[stratum]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
        cursor += 1
    selected.sort(key=lambda item: str(item["jid"]))
    output = Path(selection_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in selected),
        encoding="utf-8",
    )
    return {
        "candidate_structures": len(candidates),
        "selected_structures": len(selected),
        "selection_path": str(output),
        "selection_sha256": sha256_file(str(output)),
        "strata": dict(sorted(collections.Counter(item["stratum"] for item in selected).items())),
        "elements": sorted(
            {symbol for item in selected for symbol in item["elements"]},
            key=lambda value: ASE_ATOMIC_NUMBERS[value],
        ),
    }


def download_jarvis_dfpt_archives(
    selection_path: str,
    output_directory: str,
    *,
    proxy: Optional[str] = None,
    retries: int = 4,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Resume-download selected JARVIS DFPT archives and record SHA256 provenance."""
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selections = [
        json.loads(line)
        for line in Path(selection_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else urllib.request.ProxyHandler()
    )
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, item in enumerate(selections, start=1):
        destination = root / f"{item['jid']}.zip"
        temporary = destination.with_name(destination.name + ".part")
        if destination.is_file() and zipfile.is_zipfile(destination):
            completed.append({
                "jid": item["jid"], "path": str(destination),
                "bytes": destination.stat().st_size, "sha256": sha256_file(str(destination)),
                "url": item["archive_url"],
            })
            continue
        error = "unknown download failure"
        for attempt in range(max(1, int(retries))):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                headers = {"User-Agent": "E3MU-Neo-Dataset/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with opener.open(urllib.request.Request(item["archive_url"], headers=headers), timeout=120) as response:
                    append = offset > 0 and int(getattr(response, "status", 200)) == 206
                    with temporary.open("ab" if append else "wb") as handle:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                if not zipfile.is_zipfile(temporary):
                    raise ValueError("downloaded file is not a complete ZIP archive")
                temporary.replace(destination)
                completed.append({
                    "jid": item["jid"], "path": str(destination),
                    "bytes": destination.stat().st_size, "sha256": sha256_file(str(destination)),
                    "url": item["archive_url"],
                })
                error = ""
                break
            except Exception as exc:
                error = str(exc)
                time.sleep(min(8, 2 ** attempt))
        if error:
            failures.append({"jid": str(item["jid"]), "error": error})
        if index % 10 == 0 or index == len(selections):
            log(
                f"[{_now()}] JARVIS DFPT download: {index}/{len(selections)} "
                f"complete={len(completed)} failed={len(failures)}"
            )
    return {
        "selection_path": str(Path(selection_path).expanduser().resolve()),
        "output_directory": str(root),
        "completed": completed,
        "failures": failures,
        "total_bytes": sum(int(item["bytes"]) for item in completed),
    }


def _parse_jarvis_dfpt_archive(path: str, item: Dict[str, Any]) -> Configuration:
    """Parse geometry and per-ion Born charges from a JARVIS vasprun.xml ZIP."""
    archive = Path(path).expanduser().resolve()
    with zipfile.ZipFile(archive, "r") as zipped:
        members = {Path(name).name: name for name in zipped.namelist()}
        if "vasprun.xml" not in members:
            raise KeyError(f"JARVIS DFPT archive has no vasprun.xml: {archive}")
        with zipped.open(members["vasprun.xml"], "r") as handle:
            xml_bytes = handle.read()
        poscar_text = (
            zipped.read(members["POSCAR"]).decode("utf-8", errors="replace")
            if "POSCAR" in members else ""
        )
    # One published archive contains a zero-filled filesystem hole between two
    # complete XML elements.  Removing only NUL bytes restores the exact text;
    # all parsed tensor values remain unchanged.
    if b"\x00" in xml_bytes:
        xml_bytes = xml_bytes.replace(b"\x00", b"")
    root = ET.fromstring(xml_bytes)
    atom_array = root.find(".//atominfo/array[@name='atoms']/set")
    if atom_array is None:
        raise ValueError(f"No atom table in {archive}")
    symbols = [str(row.findall("c")[0].text).strip() for row in atom_array.findall("rc")]
    if any(symbol not in ASE_ATOMIC_NUMBERS for symbol in symbols):
        # Two official XML files truncate ``Zr`` to ``r``.  POSCAR preserves
        # the authoritative species/count header, so use it only to repair the
        # malformed element strings while retaining XML geometry and tensors.
        lines = [line.strip() for line in poscar_text.splitlines() if line.strip()]
        if len(lines) < 7:
            raise ValueError(f"Invalid species names and no usable POSCAR in {archive}")
        species = lines[5].split()
        try:
            counts = [int(value) for value in lines[6].split()]
        except ValueError as exc:
            raise ValueError(f"Invalid POSCAR species counts in {archive}") from exc
        repaired = [symbol for symbol, count in zip(species, counts) for _ in range(count)]
        if len(repaired) != len(symbols) or any(value not in ASE_ATOMIC_NUMBERS for value in repaired):
            raise ValueError(f"Cannot repair XML element names from POSCAR in {archive}")
        symbols = repaired
    structures = root.findall(".//structure")
    structure = next(
        (value for value in structures if value.get("name") == "finalpos"),
        structures[-1] if structures else None,
    )
    if structure is None:
        raise ValueError(f"No structure in {archive}")
    basis_node = structure.find(".//crystal/varray[@name='basis']")
    position_node = structure.find(".//varray[@name='positions']")
    born_array = root.find(".//array[@name='born_charges']")
    if basis_node is None or position_node is None or born_array is None:
        raise ValueError(f"Incomplete DFPT tensor/structure data in {archive}")

    def vectors(node: Any) -> np.ndarray:
        return np.asarray(
            [[float(value) for value in str(row.text).split()] for row in node.findall("v")],
            dtype=float,
        )

    cell = vectors(basis_node).reshape(3, 3)
    fractional = vectors(position_node).reshape(-1, 3)
    born_blocks = [vectors(block).reshape(3, 3) for block in born_array.findall("set")]
    bec = np.asarray(born_blocks, dtype=float)
    if len(symbols) != len(fractional) or bec.shape != (len(symbols), 3, 3):
        raise ValueError(f"DFPT atom/BEC count mismatch in {archive}")
    if not np.isfinite(cell).all() or not np.isfinite(fractional).all() or not np.isfinite(bec).all():
        raise ValueError(f"Non-finite DFPT values in {archive}")
    # The acoustic sum rule should be close to zero, but retain the published
    # tensor rather than projecting/correcting the scientific label.
    residual = float(np.max(np.abs(np.sum(bec, axis=0))))
    props: Dict[str, Any] = {
        "field": np.zeros(3, dtype=float),
        "bec": bec,
        "total_charge": 0.0,
        "source": "JARVIS-DFT-DFPT-v2025.09",
        "method_id": "VASP-DFPT-PBE",
        "system_id": str(item["jid"]),
        "group_id": f"JARVIS-DFPT:{item['jid']}",
        "sample_id": f"JARVIS-DFPT:{item['jid']}",
        "parent_id": str(item["jid"]),
        "domain": "periodic-electric-response",
        "energy_reference": "masked:DFPT-response-only",
        "provenance_id": f"figshare:{item['archive_url'].rsplit('/', 1)[-1]}#{item['jid']}",
        "bec_acoustic_sum_residual_max": residual,
    }
    props["split"] = stable_split(props["group_id"])
    cfg = Configuration(
        atomic_numbers=np.asarray([ASE_ATOMIC_NUMBERS[value] for value in symbols], dtype=int),
        positions=fractional @ cell,
        properties=props,
        property_weights={"field": 1.0, "bec": 1.0, "total_charge": 1.0},
        cell=cell,
        pbc=(True, True, True),
        config_type="JARVIS-DFT-DFPT",
        head="JARVIS:VASP-DFPT-PBE",
    )
    _validate_configuration(cfg, context=f"JARVIS-DFPT/{item['jid']}")
    return cfg


def rebuild_jarvis_dfpt_hdf5(
    selection_path: str,
    archive_directory: str,
    output_path: str,
    *,
    overwrite: bool = False,
    max_abs_bec: float = 50.0,
    max_acoustic_sum_residual: float = 0.5,
) -> str:
    """Convert downloaded JARVIS raw DFPT archives into a multi-element BEC shard."""
    selections = [
        json.loads(line)
        for line in Path(selection_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    archive_root = Path(archive_directory).expanduser().resolve()
    missing = [item["jid"] for item in selections if not (archive_root / f"{item['jid']}.zip").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} JARVIS DFPT archives; first: {missing[:5]}")

    rejected: List[Dict[str, Any]] = []

    def configurations() -> Iterable[Configuration]:
        for item in selections:
            cfg = _parse_jarvis_dfpt_archive(
                str(archive_root / f"{item['jid']}.zip"), item
            )
            bec = np.asarray(cfg.properties["bec"], dtype=float)
            max_value = float(np.max(np.abs(bec)))
            residual = float(cfg.properties["bec_acoustic_sum_residual_max"])
            if max_value > float(max_abs_bec) or residual > float(max_acoustic_sum_residual):
                rejected.append({
                    "jid": item["jid"],
                    "max_abs_bec": max_value,
                    "acoustic_sum_residual_max": residual,
                    "reason": "published DFPT tensor exceeds conservative numerical sanity bounds",
                })
                continue
            yield cfg

    output = write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "JARVIS-DFT complex multi-element Born effective charges",
            "selection_path": str(Path(selection_path).expanduser().resolve()),
            "selection_sha256": sha256_file(selection_path),
            "archive_directory": str(archive_root),
            "archive_sha256": {
                item["jid"]: sha256_file(str(archive_root / f"{item['jid']}.zip"))
                for item in selections
            },
            "figshare_article": "10.6084/m9.figshare.6815699.v11",
            "license": "CC-BY-4.0",
            "label_policy": "Published per-ion 3x3 born_charges from vasprun.xml; no ASR correction",
            "energy_policy": "DFPT energy and force records are masked from the shared loss",
            "sanity_filter": {
                "max_abs_bec": float(max_abs_bec),
                "max_acoustic_sum_residual": float(max_acoustic_sum_residual),
                "rejected_records": rejected,
            },
        },
        overwrite=overwrite,
    )
    return output


def _so3lr_leaf(group: Any) -> Any:
    if "atNUM" in group:
        return group
    child_names = list(group.keys())
    if len(child_names) != 1 or "atNUM" not in group[child_names[0]]:
        raise ValueError(f"Unexpected SO3LR HDF5 group layout: {group.name}")
    return group[child_names[0]]


def rebuild_so3lr_hdf5(
    raw_path: str,
    output_path: str,
    *,
    dataset_name: Optional[str] = None,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """Convert an official SO3LR HDF5 source without discarding atomic labels."""
    if not HAS_H5PY:
        raise RuntimeError("SO3LR conversion requires h5py")
    raw_file = Path(raw_path).expanduser().resolve()
    inferred_name = raw_file.stem
    if inferred_name.lower() == "des15k":
        inferred_name = "DES15K"
    elif inferred_name.lower() == "torsionnet500":
        inferred_name = "TorsionNet500"
    elif inferred_name.lower() == "spice_dipeptides":
        inferred_name = "SPICE_dipeptides"
    source_name = str(dataset_name or inferred_name)
    bohr_to_angstrom = 0.529177210903
    configurations: List[Configuration] = []
    with h5py.File(raw_file, "r") as raw:
        sample_ids = sorted(str(name) for name in raw.keys())
        if max_frames is not None:
            limit = max(0, int(max_frames))
            sample_ids = sorted(
                sample_ids,
                key=lambda name: hashlib.sha256(f"neo-so3lr|{source_name}|{name}".encode()).hexdigest(),
            )[:limit]
        for sample_id in sample_ids:
            sample = _so3lr_leaf(raw[sample_id])
            numbers = np.asarray(sample["atNUM"], dtype=int).reshape(-1)
            positions = np.asarray(sample["atXYZ"], dtype=float).reshape(-1, 3)
            total_energy = float(np.asarray(sample["ePBE0+MBD"]).reshape(-1)[0])
            base_energy = float(np.asarray(sample["ePBE0"]).reshape(-1)[0])
            total_forces = np.asarray(sample["totFOR"], dtype=float).reshape(-1, 3)
            base_forces = np.asarray(sample["pbe0FOR"], dtype=float).reshape(-1, 3)
            hirshfeld_charges = np.asarray(sample["hCHG"], dtype=float).reshape(-1)
            integer_total_charge = float(np.rint(np.sum(hirshfeld_charges)))
            parent_id = _so3lr_parent_id(source_name, sample_id)
            props: Dict[str, Any] = {
                "energy": total_energy,
                "energy_base": base_energy,
                "energy_dispersion": total_energy - base_energy,
                "forces": total_forces,
                "forces_base": base_forces,
                "forces_dispersion": total_forces - base_forces,
                "field": np.zeros(3, dtype=float),
                "dipole": np.asarray(sample["vDIP"], dtype=float).reshape(3),
                "charges": hirshfeld_charges,
                "atomic_dipoles": np.asarray(sample["hVDIP"], dtype=float).reshape(-1, 3)
                * bohr_to_angstrom,
                "total_charge": integer_total_charge,
                "source": source_name,
                "method_id": "PBE0+MBD/tight",
                "system_id": sample_id,
                "group_id": f"{source_name}:{parent_id}",
                "sample_id": f"{source_name}:{sample_id}",
                "parent_id": parent_id,
                "domain": "molecular",
                "energy_reference": f"{source_name}:PBE0+MBD/tight:absolute",
                "provenance_id": "zenodo:10.5281/zenodo.14779793",
            }
            cfg = Configuration(
                atomic_numbers=numbers,
                positions=positions,
                properties=props,
                property_weights={
                    name: 1.0
                    for name in props
                    if name in HDF5_STRUCTURE_LABELS or name in HDF5_ATOM_LABELS
                },
                cell=np.eye(3, dtype=float) * 100.0,
                pbc=(False, False, False),
                config_type=source_name,
                head=f"{source_name}:PBE0+MBD/tight",
            )
            _validate_configuration(cfg, context=f"{source_name}/{sample_id}")
            configurations.append(cfg)
    split_diagnostics = assign_stable_group_splits(configurations)
    return write_hdf5_dataset(
        configurations,
        output_path,
        metadata={
            "dataset": source_name,
            "raw_path": str(raw_file),
            "raw_sha256": sha256_file(str(raw_file)),
            "doi": "10.5281/zenodo.14779793",
            "license": "CC-BY-4.0",
            "atomic_dipole_conversion": "hVDIP e*bohr multiplied by 0.529177210903",
            "total_charge_semantics": "nearest integer to the sum of Hirshfeld atomic charges",
            "energy_decomposition": "energy_dispersion = ePBE0+MBD - ePBE0",
            "force_decomposition": "forces_dispersion = totFOR - pbe0FOR",
            "split_diagnostics": split_diagnostics,
        },
        overwrite=overwrite,
    )


def rebuild_deepspin_hdf5(
    raw_directory: str,
    output_path: str,
    *,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """Convert the official DeepSPIN NiO pseudo-atom representation."""
    raw_dir = Path(raw_directory).expanduser().resolve()
    required = {name: raw_dir / name for name in ("box.raw", "coord.raw", "energy.raw", "force.raw", "type.raw")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DeepSPIN raw files are missing: {missing}")
    boxes = np.loadtxt(required["box.raw"], ndmin=2, dtype=float)
    coordinates = np.loadtxt(required["coord.raw"], ndmin=2, dtype=float)
    energies = np.loadtxt(required["energy.raw"], ndmin=1, dtype=float).reshape(-1)
    all_forces = np.loadtxt(required["force.raw"], ndmin=2, dtype=float)
    types = np.loadtxt(required["type.raw"], ndmin=1, dtype=int).reshape(-1)
    if not (len(boxes) == len(coordinates) == len(energies) == len(all_forces)):
        raise ValueError("DeepSPIN raw files have inconsistent frame counts")
    if len(types) != 48 or tuple(np.bincount(types, minlength=3)) != (16, 16, 16):
        raise ValueError("DeepSPIN NiO type.raw must contain 16 Ni, 16 O, and 16 pseudo-atoms")
    frame_count = len(energies) if max_frames is None else min(len(energies), max(0, int(max_frames)))
    virtual_length = 0.4
    spin_magnitude = 1.2737
    configurations: List[Configuration] = []
    spin_norm_errors: List[float] = []
    virtual_length_errors: List[float] = []
    for index in range(frame_count):
        cell = boxes[index].reshape(3, 3)
        coords = coordinates[index].reshape(48, 3)
        force = all_forces[index].reshape(48, 3)
        real_positions = coords[:32]
        pseudo_positions = coords[32:]
        displacement = pseudo_positions - real_positions[:16]
        fractional = np.linalg.solve(cell.T, displacement.T).T
        fractional -= np.rint(fractional)
        displacement = fractional @ cell
        lengths = np.linalg.norm(displacement, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError(f"DeepSPIN frame {index}: zero pseudo-atom displacement")
        spins = np.zeros((32, 3), dtype=float)
        spins[:16] = displacement / lengths[:, None]
        magnetic_moments = spins * spin_magnitude
        effective_field = np.zeros((32, 3), dtype=float)
        effective_field[:16] = force[32:] * virtual_length
        block = index // 20
        group_id = f"DeepSPIN-NiO:block-{block:03d}"
        props: Dict[str, Any] = {
            "energy": float(energies[index]),
            "forces": force[:32],
            "spins": spins,
            "magnetic_moments": magnetic_moments,
            "effective_field": effective_field,
            "source": "DeepSPIN-NiO",
            "method_id": "DeltaSpin/VASP/noncollinear-DFT",
            "system_id": f"DeepSPIN-NiO/frame-{index:03d}",
            "group_id": group_id,
            "split": "train" if block < 4 else "val",
            "sample_id": f"DeepSPIN-NiO:{index:03d}",
            "parent_id": group_id,
            "domain": "periodic-spin",
            "energy_reference": "DeepSPIN-NiO:DeltaSpin:absolute",
            "provenance_id": "github:yangtengleo/DeepSPIN@526ade353906f21cf4d8cb32db3d53ce83e1ed53",
        }
        cfg = Configuration(
            atomic_numbers=np.asarray([28] * 16 + [8] * 16, dtype=int),
            positions=real_positions,
            properties=props,
            property_weights={
                name: 1.0
                for name in props
                if name in HDF5_STRUCTURE_LABELS or name in HDF5_ATOM_LABELS
            },
            cell=cell,
            pbc=(True, True, True),
            config_type="DeepSPIN-NiO",
            head="DeepSPIN-NiO:DeltaSpin",
        )
        _validate_configuration(cfg, context=f"DeepSPIN/frame-{index}")
        spin_norm_errors.append(float(np.max(np.abs(np.linalg.norm(spins[:16], axis=1) - 1.0))))
        virtual_length_errors.append(float(np.max(np.abs(lengths - virtual_length))))
        configurations.append(cfg)
    return write_hdf5_dataset(
        configurations,
        output_path,
        metadata={
            "dataset": "DeepSPIN-NiO",
            "raw_directory": str(raw_dir),
            "raw_sha256": {name: sha256_file(str(path)) for name, path in required.items()},
            "repository": "https://github.com/yangtengleo/DeepSPIN",
            "commit": "526ade353906f21cf4d8cb32db3d53ce83e1ed53",
            "license": "GPL-3.0",
            "virtual_length_angstrom": virtual_length,
            "spin_magnitude_mu_B": spin_magnitude,
            "effective_field_conversion": "0.4 * pseudo_atom_force (eV/spin)",
            "split_strategy": "author five 20-frame blocks; blocks 0-3 train, block 4 val",
            "max_spin_unit_norm_error": max(spin_norm_errors, default=0.0),
            "max_virtual_length_error_angstrom": max(virtual_length_errors, default=0.0),
        },
        overwrite=overwrite,
    )


def _mptrj_site_element(site: Dict[str, Any]) -> str:
    species = list(site.get("species") or [])
    if len(species) != 1 or float(species[0].get("occu", 0.0)) != 1.0:
        raise ValueError("Neo MPtrj does not accept disordered or partially occupied sites")
    element = str(species[0].get("element", ""))
    if element not in ASE_ATOMIC_NUMBERS:
        raise ValueError(f"Unknown MPtrj element: {element!r}")
    return element


def _mptrj_record_descriptor(
    mp_id: str,
    frame_id: str,
    record: Dict[str, Any],
    *,
    min_elements: int,
    min_abs_moment: float,
    max_atoms: int,
) -> Optional[Dict[str, Any]]:
    structure = dict(record.get("structure") or {})
    sites = list(structure.get("sites") or [])
    atom_count = len(sites)
    if atom_count <= 0 or atom_count > int(max_atoms):
        return None
    try:
        symbols = [_mptrj_site_element(site) for site in sites]
        moments = np.asarray(record.get("magmom"), dtype=float).reshape(-1)
    except Exception:
        return None
    if len(moments) != atom_count or not np.isfinite(moments).all():
        return None
    magnetic_mask = np.abs(moments) >= float(min_abs_moment)
    if not np.any(magnetic_mask):
        return None
    unique_symbols = sorted(set(symbols), key=lambda symbol: ASE_ATOMIC_NUMBERS[symbol])
    if len(unique_symbols) < int(min_elements):
        return None
    has_positive = bool(np.any(moments[magnetic_mask] > 0.0))
    has_negative = bool(np.any(moments[magnetic_mask] < 0.0))
    sign_class = "mixed-sign" if has_positive and has_negative else "single-sign"
    complexity = "binary" if len(unique_symbols) == 2 else "ternary" if len(unique_symbols) == 3 else "quaternary+"
    size_class = "small" if atom_count <= 24 else "medium" if atom_count <= 64 else "large"
    magnetic_elements = sorted(
        {
            symbols[index]
            for index in np.flatnonzero(magnetic_mask)
        },
        key=lambda symbol: ASE_ATOMIC_NUMBERS[symbol],
    )
    return {
        "mp_id": str(mp_id),
        "frame_id": str(frame_id),
        "atom_count": atom_count,
        "elements": unique_symbols,
        "magnetic_elements": magnetic_elements,
        "element_count": len(unique_symbols),
        "magnetic_sites": int(np.count_nonzero(magnetic_mask)),
        "max_abs_moment": float(np.max(np.abs(moments))),
        "mean_abs_moment": float(np.mean(np.abs(moments[magnetic_mask]))),
        "stratum": f"{complexity}|{size_class}|{sign_class}",
        "rank": hashlib.sha256(f"neo-mptrj|{mp_id}|{frame_id}".encode()).hexdigest(),
    }


def scan_mptrj_magnetic_candidates(
    raw_json: str,
    index_path: str,
    *,
    min_elements: int = 2,
    min_abs_moment: float = 0.05,
    max_atoms: int = 160,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Stream the official MPtrj JSON and retain one best magnetic frame per mp_id."""
    if not HAS_IJSON:
        raise RuntimeError("MPtrj streaming requires ijson>=3.3,<4")
    source = Path(raw_json).expanduser().resolve()
    output = Path(index_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    materials_scanned = 0
    frames_scanned = 0
    candidates = 0
    stratum_counts: Dict[str, int] = {}
    element_counts: Dict[str, int] = {}
    with source.open("rb") as handle, temporary.open("w", encoding="utf-8") as writer:
        for mp_id, family in ijson.kvitems(handle, ""):
            materials_scanned += 1
            best: Optional[Dict[str, Any]] = None
            for frame_id, record in dict(family).items():
                frames_scanned += 1
                descriptor = _mptrj_record_descriptor(
                    str(mp_id), str(frame_id), record,
                    min_elements=int(min_elements),
                    min_abs_moment=float(min_abs_moment),
                    max_atoms=int(max_atoms),
                )
                if descriptor is None:
                    continue
                if best is None or (
                    descriptor["magnetic_sites"],
                    descriptor["element_count"],
                    descriptor["max_abs_moment"],
                    descriptor["rank"],
                ) > (
                    best["magnetic_sites"],
                    best["element_count"],
                    best["max_abs_moment"],
                    best["rank"],
                ):
                    best = descriptor
            del family
            if best is not None:
                writer.write(json.dumps(best, sort_keys=True) + "\n")
                candidates += 1
                stratum = str(best["stratum"])
                stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
                for element in best["elements"]:
                    element_counts[element] = element_counts.get(element, 0) + 1
            if materials_scanned % 5000 == 0:
                log(
                    f"[{_now()}] MPtrj scan: materials={materials_scanned} "
                    f"frames={frames_scanned} candidates={candidates}"
                )
    temporary.replace(output)
    result = {
        "source": str(source),
        "source_md5_expected": "50ead5f27f9a4f6beb7564c4188f1e9f",
        "materials_scanned": materials_scanned,
        "frames_scanned": frames_scanned,
        "candidate_materials": candidates,
        "min_elements": int(min_elements),
        "min_abs_moment": float(min_abs_moment),
        "max_atoms": int(max_atoms),
        "strata": dict(sorted(stratum_counts.items())),
        "elements": dict(sorted(element_counts.items(), key=lambda item: ASE_ATOMIC_NUMBERS[item[0]])),
        "index_path": str(output),
        "index_sha256": sha256_file(str(output)),
    }
    log(f"[{_now()}] MPtrj candidate scan complete: {json.dumps(result, sort_keys=True)}")
    return result


def select_mptrj_magnetic_candidates(
    index_path: str,
    selection_path: str,
    *,
    target_structures: int = 12000,
) -> Dict[str, Any]:
    """Select a deterministic, stratum-balanced, multi-element MPtrj subset."""
    candidates = [
        json.loads(line)
        for line in Path(index_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = min(max(0, int(target_structures)), len(candidates))
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        by_stratum.setdefault(str(item["stratum"]), []).append(item)
    for values in by_stratum.values():
        values.sort(key=lambda item: str(item["rank"]))
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()
    strata = sorted(by_stratum)
    while len(selected) < target:
        made_progress = False
        for stratum in strata:
            values = by_stratum[stratum]
            while values and values[0]["mp_id"] in selected_ids:
                values.pop(0)
            if not values:
                continue
            item = values.pop(0)
            selected.append(item)
            selected_ids.add(item["mp_id"])
            made_progress = True
            if len(selected) >= target:
                break
        if not made_progress:
            break
    selected.sort(key=lambda item: (item["mp_id"], item["frame_id"]))
    output = Path(selection_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in selected),
        encoding="utf-8",
    )
    stratum_counts: Dict[str, int] = {}
    element_counts: Dict[str, int] = {}
    for item in selected:
        stratum_counts[item["stratum"]] = stratum_counts.get(item["stratum"], 0) + 1
        for element in item["elements"]:
            element_counts[element] = element_counts.get(element, 0) + 1
    return {
        "selected_structures": len(selected),
        "selected_materials": len(selected_ids),
        "selection_path": str(output),
        "selection_sha256": sha256_file(str(output)),
        "strata": dict(sorted(stratum_counts.items())),
        "elements": dict(sorted(element_counts.items(), key=lambda item: ASE_ATOMIC_NUMBERS[item[0]])),
    }


def _configuration_from_mptrj_record(
    mp_id: str,
    frame_id: str,
    record: Dict[str, Any],
    *,
    min_abs_moment: float,
) -> Configuration:
    structure = dict(record["structure"])
    sites = list(structure["sites"])
    symbols = [_mptrj_site_element(site) for site in sites]
    numbers = np.asarray([ASE_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=int)
    positions = np.asarray([site["xyz"] for site in sites], dtype=float).reshape(-1, 3)
    cell = np.asarray(structure["lattice"]["matrix"], dtype=float).reshape(3, 3)
    moments_scalar = np.asarray(record["magmom"], dtype=float).reshape(-1)
    magnetic_mask = np.abs(moments_scalar) >= float(min_abs_moment)
    spins = np.zeros((len(numbers), 3), dtype=float)
    spins[magnetic_mask, 2] = np.sign(moments_scalar[magnetic_mask])
    moments = np.zeros((len(numbers), 3), dtype=float)
    moments[magnetic_mask, 2] = moments_scalar[magnetic_mask]
    corrected_energy = float(record["corrected_total_energy"])
    props: Dict[str, Any] = {
        "energy": corrected_energy,
        "forces": np.asarray(record["force"], dtype=float).reshape(-1, 3),
        "spins": spins,
        "magnetic_moments": moments,
        "source": "MPtrj-v2022.9-magnetic",
        "method_id": "MP-GGA/GGA+U:MP2020-compatible",
        "system_id": str(mp_id),
        "group_id": f"MPtrj:{mp_id}",
        "sample_id": f"MPtrj:{mp_id}:{frame_id}",
        "parent_id": str(mp_id),
        "domain": "periodic-collinear-spin",
        "energy_reference": "MPtrj:corrected_total_energy:MP2020-compatible",
        "provenance_id": f"figshare:10.6084/m9.figshare.23713842.v2#{mp_id}/{frame_id}",
    }
    cfg = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        properties=props,
        property_weights={
            name: 1.0
            for name in props
            if name in HDF5_STRUCTURE_LABELS or name in HDF5_ATOM_LABELS
        },
        cell=cell,
        pbc=(True, True, True),
        config_type="MPtrj-magnetic",
        head="MPtrj:MP2020-compatible",
    )
    _validate_configuration(cfg, context=f"MPtrj/{mp_id}/{frame_id}")
    return cfg


def rebuild_mptrj_magnetic_hdf5(
    raw_json: str,
    selection_path: str,
    output_path: str,
    *,
    min_abs_moment: float = 0.05,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> str:
    """Decode only selected MPtrj material families and write canonical HDF5."""
    if not HAS_IJSON:
        raise RuntimeError("MPtrj streaming requires ijson>=3.3,<4")
    selections = [
        json.loads(line)
        for line in Path(selection_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {str(item["mp_id"]): str(item["frame_id"]) for item in selections}
    decoded = 0

    def configurations() -> Iterable[Configuration]:
        nonlocal decoded
        with Path(raw_json).expanduser().open("rb") as handle:
            for mp_id, family in ijson.kvitems(handle, ""):
                frame_id = selected.get(str(mp_id))
                if frame_id is None:
                    del family
                    continue
                record = dict(family).get(frame_id)
                if record is None:
                    raise KeyError(f"Selected MPtrj frame is missing: {mp_id}/{frame_id}")
                cfg = _configuration_from_mptrj_record(
                    str(mp_id), frame_id, record, min_abs_moment=float(min_abs_moment)
                )
                cfg.properties["split"] = stable_split(cfg.properties["group_id"])
                decoded += 1
                if decoded % 1000 == 0:
                    log(f"[{_now()}] MPtrj decode: {decoded}/{len(selected)} structures")
                yield cfg
                del family

    result = write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "MPtrj-v2022.9 magnetic multi-element subset",
            "raw_path": str(Path(raw_json).expanduser().resolve()),
            "raw_md5": "50ead5f27f9a4f6beb7564c4188f1e9f",
            "selection_path": str(Path(selection_path).expanduser().resolve()),
            "selection_sha256": sha256_file(selection_path),
            "doi": "10.6084/m9.figshare.23713842.v2",
            "license": "MIT",
            "energy_label": "corrected_total_energy after MP2020 compatibility",
            "spin_encoding": "z component is sign of site-wise collinear magmom; zero below threshold",
            "magnetic_moment_unit": "mu_B",
            "min_abs_moment": float(min_abs_moment),
            "explicitly_absent_labels": ["effective_field", "J_effective", "Di", "DMI_effective"],
        },
        overwrite=overwrite,
    )
    if decoded != len(selected):
        raise ValueError(f"Decoded {decoded} MPtrj structures, expected {len(selected)}")
    return result


def _configuration_from_mptrj_large_record(
    mp_id: str,
    frame_id: str,
    record: Dict[str, Any],
    *,
    min_abs_moment: float,
) -> Configuration:
    """Convert a general MPtrj trajectory frame, preserving spin labels when present."""
    structure = dict(record["structure"])
    sites = list(structure["sites"])
    symbols = [_mptrj_site_element(site) for site in sites]
    numbers = np.asarray([ASE_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=int)
    positions = np.asarray([site["xyz"] for site in sites], dtype=float).reshape(-1, 3)
    cell = np.asarray(structure["lattice"]["matrix"], dtype=float).reshape(3, 3)
    props: Dict[str, Any] = {
        "energy": float(record["corrected_total_energy"]),
        "forces": np.asarray(record["force"], dtype=float).reshape(-1, 3),
        "source": "MPtrj-v2022.9-large",
        "method_id": "MP-GGA/GGA+U:MP2020-compatible",
        "system_id": str(mp_id),
        "group_id": f"MPtrj:{mp_id}",
        "sample_id": f"MPtrj:{mp_id}:{frame_id}",
        "parent_id": str(mp_id),
        "domain": "periodic",
        "energy_reference": "MPtrj:corrected_total_energy:MP2020-compatible",
        "provenance_id": f"figshare:10.6084/m9.figshare.23713842.v2#{mp_id}/{frame_id}",
    }
    raw_moments = record.get("magmom")
    if raw_moments is not None:
        try:
            moments_scalar = np.asarray(raw_moments, dtype=float).reshape(-1)
        except Exception:
            moments_scalar = np.empty((0,), dtype=float)
        if len(moments_scalar) == len(numbers) and np.isfinite(moments_scalar).all():
            magnetic_mask = np.abs(moments_scalar) >= float(min_abs_moment)
            if np.any(magnetic_mask):
                spins = np.zeros((len(numbers), 3), dtype=float)
                spins[magnetic_mask, 2] = np.sign(moments_scalar[magnetic_mask])
                moments = np.zeros((len(numbers), 3), dtype=float)
                moments[magnetic_mask, 2] = moments_scalar[magnetic_mask]
                props["spins"] = spins
                props["magnetic_moments"] = moments
                props["domain"] = "periodic-collinear-spin"
    props["split"] = stable_split(props["group_id"])
    cfg = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        properties=props,
        property_weights={
            name: 1.0
            for name in props
            if name in HDF5_STRUCTURE_LABELS or name in HDF5_ATOM_LABELS
        },
        cell=cell,
        pbc=(True, True, True),
        config_type="MPtrj-large",
        head="MPtrj:MP2020-compatible",
    )
    _validate_configuration(cfg, context=f"MPtrj-large/{mp_id}/{frame_id}")
    if cfg.properties["forces"].shape != (len(numbers), 3):
        raise ValueError(f"MPtrj-large/{mp_id}/{frame_id}: force shape does not match atoms")
    return cfg


def rebuild_mptrj_large_hdf5(
    raw_json: str,
    output_path: str,
    *,
    required_hdf5: Sequence[str] = (),
    max_per_material: int = 4,
    min_elements: int = 2,
    max_atoms: int = 160,
    min_abs_moment: float = 0.05,
    overwrite: bool = False,
    report_path: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Build a trajectory-rich MPtrj shard while retaining required canonical samples."""
    if not HAS_IJSON:
        raise RuntimeError("MPtrj streaming requires ijson>=3.3,<4")
    if int(max_per_material) < 1:
        raise ValueError("max_per_material must be at least 1")
    required_samples: set = set()
    required_by_material: Dict[str, set] = {}
    required_sources: List[Dict[str, Any]] = []
    for raw_path in required_hdf5:
        path = str(Path(raw_path).expanduser().resolve())
        if not _is_hdf5_path(path):
            raise ValueError(f"Required sample source is not HDF5: {path}")
        with h5py.File(path, "r") as handle:
            sample_ids = [str(value) for value in handle["metadata/sample_id"].asstr()[:]]
        mptrj_ids = [value for value in sample_ids if value.startswith("MPtrj:")]
        for sample_id in mptrj_ids:
            rest = sample_id[len("MPtrj:"):]
            if ":" not in rest:
                raise ValueError(f"Cannot parse required MPtrj sample ID: {sample_id}")
            mp_id, frame_id = rest.split(":", 1)
            required_samples.add(sample_id)
            required_by_material.setdefault(mp_id, set()).add(frame_id)
        required_sources.append({
            "path": path,
            "sha256": sha256_file(path),
            "mptrj_samples": len(mptrj_ids),
        })

    counters: Dict[str, int] = {
        "materials_scanned": 0,
        "frames_scanned": 0,
        "eligible_frames": 0,
        "selected_structures": 0,
        "selected_magnetic_structures": 0,
    }
    matched_required: set = set()
    element_counts: Dict[str, int] = {}
    source = Path(raw_json).expanduser().resolve()

    def configurations() -> Iterable[Configuration]:
        with source.open("rb") as handle:
            for mp_id_raw, family_raw in ijson.kvitems(handle, ""):
                mp_id = str(mp_id_raw)
                counters["materials_scanned"] += 1
                family = dict(family_raw)
                candidates: List[Tuple[str, str, Dict[str, Any], bool]] = []
                required_frames = required_by_material.get(mp_id, set())
                for frame_id_raw, record_raw in family.items():
                    frame_id = str(frame_id_raw)
                    record = dict(record_raw)
                    counters["frames_scanned"] += 1
                    try:
                        structure = dict(record.get("structure") or {})
                        sites = list(structure.get("sites") or [])
                        if not (0 < len(sites) <= int(max_atoms)):
                            continue
                        symbols = [_mptrj_site_element(site) for site in sites]
                        if len(set(symbols)) < int(min_elements):
                            continue
                        energy = float(record["corrected_total_energy"])
                        forces = np.asarray(record["force"], dtype=float)
                        if not math.isfinite(energy) or forces.shape != (len(sites), 3):
                            continue
                        if not np.isfinite(forces).all():
                            continue
                    except Exception:
                        continue
                    counters["eligible_frames"] += 1
                    sample_id = f"MPtrj:{mp_id}:{frame_id}"
                    required = frame_id in required_frames
                    rank = hashlib.sha256(
                        f"neo-large-mptrj|{mp_id}|{frame_id}".encode()
                    ).hexdigest()
                    candidates.append((rank, frame_id, record, required))

                required_candidates = sorted(
                    (item for item in candidates if item[3]), key=lambda item: item[1]
                )
                optional_candidates = sorted(
                    (item for item in candidates if not item[3]), key=lambda item: item[0]
                )
                material_limit = max(int(max_per_material), len(required_candidates))
                selected = required_candidates + optional_candidates[
                    : max(0, material_limit - len(required_candidates))
                ]
                for _rank, frame_id, record, required in selected:
                    cfg = _configuration_from_mptrj_large_record(
                        mp_id,
                        frame_id,
                        record,
                        min_abs_moment=float(min_abs_moment),
                    )
                    counters["selected_structures"] += 1
                    if "spins" in cfg.properties:
                        counters["selected_magnetic_structures"] += 1
                    if required:
                        matched_required.add(str(cfg.properties["sample_id"]))
                    for number in np.unique(cfg.atomic_numbers):
                        symbol = ASE_CHEMICAL_SYMBOLS[int(number)]
                        element_counts[symbol] = element_counts.get(symbol, 0) + 1
                    yield cfg
                del family, candidates, selected
                if counters["materials_scanned"] % 5000 == 0:
                    log(
                        f"[{_now()}] MPtrj large build: "
                        f"materials={counters['materials_scanned']} "
                        f"frames={counters['frames_scanned']} "
                        f"selected={counters['selected_structures']} "
                        f"required={len(matched_required)}/{len(required_samples)}"
                    )

    output = write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "MPtrj-v2022.9 trajectory-rich large multi-element shard",
            "raw_path": str(source),
            "raw_md5_expected": "50ead5f27f9a4f6beb7564c4188f1e9f",
            "doi": "10.6084/m9.figshare.23713842.v2",
            "license": "MIT",
            "selection": {
                "max_per_material": int(max_per_material),
                "min_elements": int(min_elements),
                "max_atoms": int(max_atoms),
                "rank": "SHA256(neo-large-mptrj|mp_id|frame_id)",
                "required_samples_always_retained": True,
            },
            "required_sources": required_sources,
            "energy_label": "corrected_total_energy after MP2020 compatibility",
            "split_strategy": "stable hash of MPtrj mp_id; all trajectory frames remain in one split",
            "missing_magmom_policy": "Spin and magnetic-moment masks are absent, never filled with zero targets",
        },
        overwrite=overwrite,
    )
    missing_required = sorted(required_samples - matched_required)
    if missing_required:
        raise ValueError(
            f"Large MPtrj build did not retain {len(missing_required)} required samples; "
            f"first missing IDs: {missing_required[:10]}"
        )
    summary = hdf5_dataset_summary(output)
    report: Dict[str, Any] = {
        "output": output,
        "summary": summary,
        "counters": dict(counters),
        "required_samples": len(required_samples),
        "matched_required_samples": len(matched_required),
        "required_sources": required_sources,
        "elements": dict(sorted(element_counts.items(), key=lambda item: ASE_ATOMIC_NUMBERS[item[0]])),
    }
    if report_path:
        destination = Path(report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(_checkpoint_safe(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def select_static_mptrj_parquet_rows(
    parquet_directory: str,
    selection_path: str,
    *,
    source_rows: int = 200000,
    target_materials: int = 40000,
    min_elements: int = 2,
    max_atoms: int = 160,
) -> Dict[str, Any]:
    """Recover and select provenance IDs for the MPtrj prefix in static.extxyz."""
    if not HAS_PYARROW:
        raise RuntimeError("MPtrj Parquet ID recovery requires pyarrow>=16")
    root = Path(parquet_directory).expanduser().resolve()
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {root}")
    columns = [
        "numbers", "energy", "corrected_total_energy", "mp_id", "task_id",
        "calc_id", "ionic_step", "num_atoms",
    ]
    best_by_material: Dict[str, Dict[str, Any]] = {}
    global_index = 0
    for shard in shards:
        parquet = _pyarrow_parquet.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for row in batch.to_pylist():
                if global_index >= int(source_rows):
                    break
                numbers = np.asarray(row["numbers"], dtype=int).reshape(-1)
                elements_z = sorted(int(value) for value in np.unique(numbers))
                atom_count = int(row.get("num_atoms") or len(numbers))
                raw_energy = float(row["energy"])
                corrected_energy = float(row["corrected_total_energy"])
                if (
                    len(elements_z) >= int(min_elements)
                    and 0 < atom_count <= int(max_atoms)
                    and len(numbers) == atom_count
                    and math.isfinite(raw_energy)
                    and math.isfinite(corrected_energy)
                ):
                    mp_id = str(row["mp_id"])
                    task_id = str(row["task_id"])
                    calc_id = int(row["calc_id"])
                    ionic_step = int(row["ionic_step"])
                    frame_id = f"{task_id}-{calc_id}-{ionic_step}"
                    rank = hashlib.sha256(
                        f"neo-static|{mp_id}|{frame_id}|{global_index}".encode()
                    ).hexdigest()
                    element_count = len(elements_z)
                    complexity = (
                        "binary" if element_count == 2
                        else "ternary" if element_count == 3
                        else "quaternary+"
                    )
                    size_class = (
                        "small" if atom_count <= 24
                        else "medium" if atom_count <= 64
                        else "large"
                    )
                    candidate = {
                        "extxyz_index": global_index,
                        "mp_id": mp_id,
                        "task_id": task_id,
                        "calc_id": calc_id,
                        "ionic_step": ionic_step,
                        "frame_id": frame_id,
                        "atom_count": atom_count,
                        "elements": [ASE_CHEMICAL_SYMBOLS[value] for value in elements_z],
                        "raw_energy": raw_energy,
                        "corrected_total_energy": corrected_energy,
                        "stratum": f"{complexity}|{size_class}",
                        "rank": rank,
                    }
                    previous = best_by_material.get(mp_id)
                    if previous is None or rank < str(previous["rank"]):
                        best_by_material[mp_id] = candidate
                global_index += 1
            if global_index >= int(source_rows):
                break
        if global_index >= int(source_rows):
            break
    if global_index != int(source_rows):
        raise ValueError(
            f"Parquet shards contain only {global_index} rows; expected {source_rows}"
        )
    candidates = list(best_by_material.values())
    target = min(max(0, int(target_materials)), len(candidates))
    selected: List[Dict[str, Any]] = []
    selected_materials: set = set()

    element_frequency: Dict[str, int] = {}
    for item in candidates:
        for element in item["elements"]:
            element_frequency[element] = element_frequency.get(element, 0) + 1
    for element in sorted(
        element_frequency,
        key=lambda symbol: (element_frequency[symbol], ASE_ATOMIC_NUMBERS[symbol]),
    ):
        choices = [item for item in candidates if element in item["elements"]]
        choices.sort(key=lambda item: str(item["rank"]))
        if choices and choices[0]["mp_id"] not in selected_materials:
            selected.append(choices[0])
            selected_materials.add(choices[0]["mp_id"])
            if len(selected) >= target:
                break

    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        if item["mp_id"] in selected_materials:
            continue
        by_stratum.setdefault(str(item["stratum"]), []).append(item)
    for values in by_stratum.values():
        values.sort(key=lambda item: str(item["rank"]))
    strata = sorted(by_stratum)
    cursors = {name: 0 for name in strata}
    while len(selected) < target:
        progress = False
        for stratum in strata:
            cursor = cursors[stratum]
            values = by_stratum[stratum]
            while cursor < len(values) and values[cursor]["mp_id"] in selected_materials:
                cursor += 1
            cursors[stratum] = cursor
            if cursor >= len(values):
                continue
            item = values[cursor]
            cursors[stratum] = cursor + 1
            selected.append(item)
            selected_materials.add(item["mp_id"])
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    selected.sort(key=lambda item: int(item["extxyz_index"]))
    output = Path(selection_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in selected),
        encoding="utf-8",
    )
    selected_elements = sorted(
        {element for item in selected for element in item["elements"]},
        key=lambda symbol: ASE_ATOMIC_NUMBERS[symbol],
    )
    return {
        "source_rows": global_index,
        "candidate_materials": len(candidates),
        "selected_materials": len(selected),
        "elements": selected_elements,
        "selection_path": str(output),
        "selection_sha256": sha256_file(str(output)),
        "parquet_shards": [str(path) for path in shards],
    }


def rebuild_static_mptrj_hdf5(
    static_extxyz: str,
    parquet_directory: str,
    selection_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Rebuild selected static MPtrj frames with exact IDs and corrected energies."""
    selections = [
        json.loads(line)
        for line in Path(selection_path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_index = {int(item["extxyz_index"]): item for item in selections}
    selected_indices = set(by_index)

    def configurations() -> Iterable[Configuration]:
        decoded = 0
        for record in _iter_extxyz_records(
            static_extxyz,
            start_index=0,
            end_index=200000,
            selected_indices=selected_indices,
        ):
            frame_index = int(record[0])
            item = by_index[frame_index]
            cfg = _configuration_from_extxyz_record(*record)
            if len(cfg.atomic_numbers) != int(item["atom_count"]):
                raise ValueError(f"Static/Parquet atom-count mismatch at frame {frame_index}")
            extxyz_energy = float(cfg.properties.get("energy", math.nan))
            if not math.isclose(
                extxyz_energy, float(item["raw_energy"]), rel_tol=0.0, abs_tol=5e-6
            ):
                raise ValueError(
                    f"Static/Parquet raw-energy mismatch at frame {frame_index}: "
                    f"{extxyz_energy} vs {item['raw_energy']}"
                )
            cfg.properties["energy"] = float(item["corrected_total_energy"])
            cfg.properties.update({
                "source": "MPtrj-static-recovered",
                "method_id": "MP-GGA/GGA+U:MP2020-compatible",
                "system_id": str(item["mp_id"]),
                "group_id": f"MPtrj:{item['mp_id']}",
                "sample_id": f"MPtrj:{item['mp_id']}:{item['frame_id']}",
                "parent_id": str(item["mp_id"]),
                "domain": "periodic",
                "energy_reference": "MPtrj:corrected_total_energy:MP2020-compatible",
                "provenance_id": (
                    f"local-static+parquet:{item['mp_id']}/{item['frame_id']}"
                ),
            })
            cfg.properties["split"] = stable_split(cfg.properties["group_id"])
            cfg.head = "MPtrj:MP2020-compatible"
            _validate_configuration(cfg, context=f"static/MPtrj/{frame_index}")
            decoded += 1
            yield cfg
        if decoded != len(selections):
            raise ValueError(f"Decoded {decoded} static frames, expected {len(selections)}")

    parquet_root = Path(parquet_directory).expanduser().resolve()
    return write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "MPtrj static multi-element recovered subset",
            "static_path": str(Path(static_extxyz).expanduser().resolve()),
            "static_sha256": sha256_file(static_extxyz),
            "static_mptrj_prefix_frames": 200000,
            "parquet_directory": str(parquet_root),
            "parquet_shards": [
                {"path": str(path), "sha256": sha256_file(str(path))}
                for path in sorted(parquet_root.glob("*.parquet"))
            ],
            "selection_path": str(Path(selection_path).expanduser().resolve()),
            "selection_sha256": sha256_file(selection_path),
            "energy_replacement": "static raw energy replaced by Parquet corrected_total_energy",
            "split_strategy": "stable hash of recovered mp_id",
        },
        overwrite=overwrite,
    )


def rebuild_scfnn_from_combined_extxyz(
    static_extxyz: str,
    response_extxyz: str,
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Recover the unique zero- and finite-field SCFNN water-box tail segments."""
    def configurations() -> Iterable[Configuration]:
        for role, path, start, end in (
            ("zero-field", static_extxyz, 200000, 201590),
            ("finite-field", response_extxyz, 101993, 105179),
        ):
            for record in _iter_extxyz_records(path, start_index=start, end_index=end):
                frame_index = int(record[0])
                cfg = _configuration_from_extxyz_record(*record)
                geometry_hash = _geometry_group_hash(cfg)
                cfg.properties.update({
                    "source": f"SCFNN/{role}",
                    "method_id": "SCFNN-water/CP2K-DFT",
                    "system_id": f"SCFNN:{geometry_hash}",
                    "group_id": f"SCFNN:{geometry_hash}",
                    "sample_id": f"SCFNN:{role}:{frame_index}",
                    "parent_id": geometry_hash,
                    "domain": "periodic-electric-field",
                    "energy_reference": "SCFNN:CP2K-DFT:absolute",
                    "provenance_id": f"local-{role}:{Path(path).name}#{frame_index}",
                    "total_charge": 0.0,
                })
                cfg.property_weights["total_charge"] = 1.0
                cfg.properties["split"] = stable_split(cfg.properties["group_id"])
                cfg.head = "SCFNN:CP2K-DFT"
                _validate_configuration(cfg, context=f"SCFNN/{role}/{frame_index}")
                yield cfg

    return write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "SCFNN water zero- and finite-field response",
            "static_path": str(Path(static_extxyz).expanduser().resolve()),
            "static_sha256": sha256_file(static_extxyz),
            "static_frame_range": [200000, 201590],
            "response_path": str(Path(response_extxyz).expanduser().resolve()),
            "response_sha256": sha256_file(response_extxyz),
            "response_frame_range": [101993, 105179],
            "grouping": "SHA256 of Z, positions, cell, and PBC rounded to 1e-7",
            "split_strategy": "field variants share geometry group and split",
        },
        overwrite=overwrite,
    )


def rebuild_bec_from_combined_response(
    response_extxyz: str,
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Recover the 550 real DFPT Born-effective-charge structures."""
    source_ordinals: Dict[str, int] = {}

    def configurations() -> Iterable[Configuration]:
        for record in _iter_extxyz_records(
            response_extxyz, start_index=105179, end_index=105729
        ):
            frame_index, info = int(record[0]), record[1]
            cfg = _configuration_from_extxyz_record(*record)
            if "bec" not in cfg.properties:
                raise ValueError(f"BEC frame {frame_index} has no BEC tensor")
            source = str(info.get("source", "BEC/unknown"))
            ordinal = source_ordinals.get(source, 0)
            source_ordinals[source] = ordinal + 1
            system = source.split("/", 1)[-1]
            parent_id = f"{system}:{ordinal:06d}"
            cfg.properties.update({
                "source": source,
                "method_id": "VASP-DFPT-BEC",
                "system_id": parent_id,
                "group_id": f"BEC:{parent_id}",
                "sample_id": f"BEC:{parent_id}",
                "parent_id": parent_id,
                "domain": "periodic-electric-response",
                "energy_reference": f"BEC:{system}:dataset-provided",
                "provenance_id": f"local-response:{Path(response_extxyz).name}#{frame_index}",
                "total_charge": 0.0,
            })
            cfg.property_weights["total_charge"] = 1.0
            cfg.properties["split"] = stable_split(cfg.properties["group_id"])
            cfg.head = f"BEC:{system}:VASP-DFPT"
            _validate_configuration(cfg, context=f"BEC/{frame_index}")
            yield cfg

    output = write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata={
            "dataset": "MLFF_and_BEC DFPT subset",
            "response_path": str(Path(response_extxyz).expanduser().resolve()),
            "response_sha256": sha256_file(response_extxyz),
            "frame_range": [105179, 105729],
            "source_expected": {"BEC/H2O": 100, "BEC/MAPbI3": 300, "BEC/dimer": 150},
            "split_strategy": "stable hash of source system and source ordinal",
        },
        overwrite=overwrite,
    )
    if source_ordinals != {"BEC/H2O": 100, "BEC/MAPbI3": 300, "BEC/dimer": 150}:
        raise ValueError(f"Unexpected BEC source counts: {source_ordinals}")
    return output


def _ranked_hdf5_indices(
    path: str,
    *,
    target_structures: Optional[int],
    target_groups: Optional[int],
    max_per_group: Optional[int],
    seed: int,
) -> List[int]:
    if not HAS_H5PY:
        raise RuntimeError("Neo mixed-dataset selection requires h5py")
    with h5py.File(path, "r") as handle:
        groups = [str(value) for value in handle["metadata/group_id"].asstr()[:]]
        samples = [str(value) for value in handle["metadata/sample_id"].asstr()[:]]
    by_group: Dict[str, List[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    selected: List[int] = []
    group_order = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"neo-mixed-group|{seed}|{group}".encode()).hexdigest(),
    )
    if target_groups is not None:
        group_order = group_order[: max(0, int(target_groups))]
    for group in group_order:
        values = sorted(
            by_group[group],
            key=lambda index: hashlib.sha256(
                f"neo-mixed-frame|{seed}|{samples[index]}".encode()
            ).hexdigest(),
        )
        if max_per_group is not None:
            values = values[: max(0, int(max_per_group))]
        selected.extend(values)
    if target_structures is not None and len(selected) > int(target_structures):
        target = max(0, int(target_structures))
        selected = sorted(
            selected,
            key=lambda index: hashlib.sha256(
                f"neo-mixed-target|{seed}|{samples[index]}".encode()
            ).hexdigest(),
        )[:target]
    return sorted(selected)


def _allocate_bounded_temperature_quotas(
    capacities: Dict[Any, int],
    total: int,
    *,
    temperature: float,
    minimum: int = 0,
) -> Dict[Any, int]:
    """Allocate an exact capped quota with deterministic largest remainders."""
    usable = {key: max(0, int(value)) for key, value in capacities.items() if int(value) > 0}
    requested = max(0, int(total))
    if requested > sum(usable.values()):
        raise ValueError(
            f"Cannot allocate {requested} samples from capacity {sum(usable.values())}"
        )
    keys = sorted(usable, key=lambda value: str(value))
    floor_value = max(0, int(minimum))
    quotas = {key: min(usable[key], floor_value) for key in keys}
    if sum(quotas.values()) > requested:
        quotas = {key: 0 for key in keys}
        for key in sorted(
            keys,
            key=lambda value: (
                -float(usable[value]) ** max(0.0, float(temperature)),
                str(value),
            ),
        )[:requested]:
            quotas[key] = 1
        return quotas

    remaining = requested - sum(quotas.values())
    exponent = max(0.0, float(temperature))
    while remaining > 0:
        active = [key for key in keys if quotas[key] < usable[key]]
        if not active:
            break
        weights = {
            key: float(usable[key]) ** exponent if exponent > 0.0 else 1.0
            for key in active
        }
        denominator = sum(weights.values())
        ideal = {key: remaining * weights[key] / denominator for key in active}
        grants = {
            key: min(usable[key] - quotas[key], int(math.floor(ideal[key])))
            for key in active
        }
        granted = sum(grants.values())
        if granted:
            for key, value in grants.items():
                quotas[key] += value
            remaining -= granted
            continue
        order = sorted(
            active,
            key=lambda key: (-(ideal[key] - math.floor(ideal[key])), str(key)),
        )
        for key in order:
            if remaining <= 0:
                break
            if quotas[key] < usable[key]:
                quotas[key] += 1
                remaining -= 1
    if remaining:
        raise RuntimeError(f"Quota allocation left {remaining} samples unassigned")
    return quotas


def _distribution_js_divergence(
    left: Dict[Any, int], right: Dict[Any, int]
) -> float:
    """Return Jensen-Shannon divergence in bits for two count distributions."""
    keys = sorted(set(left) | set(right), key=lambda value: str(value))
    left_total = float(sum(max(0, int(left.get(key, 0))) for key in keys))
    right_total = float(sum(max(0, int(right.get(key, 0))) for key in keys))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    p = np.asarray([max(0, int(left.get(key, 0))) / left_total for key in keys], dtype=float)
    q = np.asarray([max(0, int(right.get(key, 0))) / right_total for key in keys], dtype=float)
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        active = values > 0.0
        return float(np.sum(values[active] * np.log2(values[active] / midpoint[active])))

    return 0.5 * (kl(p) + kl(q))


def select_neo_stratified_hdf5_indices(
    path: str,
    *,
    target_structures: int,
    seed: int = 20260720,
    source_temperature: float = 0.5,
    min_per_source: int = 3,
    max_per_group: int = 2,
    preserve_sources_up_to: int = 128,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select a reproducible, source-balanced and coverage-constrained Neo tier."""
    if not HAS_H5PY:
        raise RuntimeError("Neo tier selection requires h5py")
    source_path = str(Path(path).expanduser().resolve())
    with h5py.File(source_path, "r") as handle:
        atom_ptr = np.asarray(handle["structures/atom_ptr"], dtype=np.int64)
        atomic_numbers = np.asarray(handle["structures/atomic_numbers"], dtype=np.int16)
        sources = [str(value) for value in handle["metadata/source"].asstr()[:]]
        splits = [str(value) for value in handle["metadata/split"].asstr()[:]]
        groups = [str(value) for value in handle["metadata/group_id"].asstr()[:]]
        samples = [str(value) for value in handle["metadata/sample_id"].asstr()[:]]
        masks = {
            name: np.asarray(handle[f"masks/{name}"], dtype=bool)
            for labels in _NEO_TIER_LABEL_FAMILIES.values()
            for name in labels
            if f"masks/{name}" in handle
        }
    structure_count = len(sources)
    requested = int(target_structures)
    if not (0 < requested <= structure_count):
        raise ValueError(
            f"target_structures must be in [1, {structure_count}], got {requested}"
        )
    group_limit = max(1, int(max_per_group))
    rank_values = [
        hashlib.sha256(f"neo-tier|{int(seed)}|{sample}".encode()).hexdigest()
        for sample in samples
    ]
    atom_counts = np.diff(atom_ptr)
    unique_elements: List[Tuple[int, ...]] = []
    complexity: List[str] = []
    size_class: List[str] = []
    label_signatures: List[str] = []
    family_active: Dict[str, np.ndarray] = {}
    for family, labels in _NEO_TIER_LABEL_FAMILIES.items():
        active = np.zeros((structure_count,), dtype=bool)
        for label in labels:
            if label in masks:
                active |= masks[label]
        family_active[family] = active
    for index, atom_count in enumerate(atom_counts):
        values = tuple(
            int(value)
            for value in np.unique(
                atomic_numbers[int(atom_ptr[index]) : int(atom_ptr[index + 1])]
            )
        )
        unique_elements.append(values)
        element_count = len(values)
        complexity.append(
            "unary" if element_count == 1
            else "binary" if element_count == 2
            else "ternary" if element_count == 3
            else "quaternary+"
        )
        size_class.append(
            "small_1_24" if int(atom_count) <= 24
            else "medium_25_64" if int(atom_count) <= 64
            else "large_65_plus"
        )
        active_names = [
            family for family, active in family_active.items() if bool(active[index])
        ]
        label_signatures.append("+".join(active_names) if active_names else "geometry")

    raw_source_counts = dict(collections.Counter(sources))
    rare_source_limit = max(0, int(preserve_sources_up_to))
    source_group_counts: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.Counter()
    )
    for source, group in zip(sources, groups):
        source_group_counts[source][group] += 1
    source_capacities = {
        source: (
            count
            if rare_source_limit and count <= rare_source_limit
            else sum(min(group_limit, value) for value in source_group_counts[source].values())
        )
        for source, count in raw_source_counts.items()
    }
    desired_source_quotas = _allocate_bounded_temperature_quotas(
        source_capacities,
        requested,
        temperature=float(source_temperature),
        minimum=max(0, int(min_per_source)),
    )
    protected_source_floor = {
        source: (
            raw_source_counts[source]
            if rare_source_limit and raw_source_counts[source] <= rare_source_limit
            else min(count, max(0, int(min_per_source)))
        )
        for source, count in source_capacities.items()
    }
    for source, floor_count in protected_source_floor.items():
        desired_source_quotas[source] = max(
            desired_source_quotas.get(source, 0), floor_count
        )
    quota_excess = sum(desired_source_quotas.values()) - requested
    while quota_excess > 0:
        reducible = [
            source for source in desired_source_quotas
            if desired_source_quotas[source] > protected_source_floor[source]
        ]
        if not reducible:
            raise ValueError(
                "target_structures is too small for complete rare-source preservation"
            )
        source = max(
            reducible,
            key=lambda value: (
                desired_source_quotas[value] - protected_source_floor[value],
                desired_source_quotas[value],
                str(value),
            ),
        )
        desired_source_quotas[source] -= 1
        quota_excess -= 1
    token_candidates: Dict[Tuple[Any, ...], List[int]] = collections.defaultdict(list)
    source_strata: Dict[str, Dict[Tuple[str, str, str, str], List[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for index in range(structure_count):
        source = sources[index]
        split_name = splits[index]
        token_candidates[("source_split", source, split_name)].append(index)
        token_candidates[("shape_split", complexity[index], size_class[index], split_name)].append(index)
        for family, active in family_active.items():
            if bool(active[index]):
                token_candidates[("label_split", family, split_name)].append(index)
        for atomic_number in unique_elements[index]:
            token_candidates[("element_split", atomic_number, split_name)].append(index)
        source_strata[source][
            (split_name, complexity[index], size_class[index], label_signatures[index])
        ].append(index)

    selected: set = set()
    selected_by_group: Dict[str, int] = collections.Counter()
    selected_by_source: Dict[str, int] = collections.Counter()

    def add(index: int) -> bool:
        if index in selected:
            return True
        group = groups[index]
        source = sources[index]
        effective_group_limit = (
            raw_source_counts[source]
            if rare_source_limit and raw_source_counts[source] <= rare_source_limit
            else group_limit
        )
        if selected_by_group[group] >= effective_group_limit:
            return False
        selected.add(index)
        selected_by_group[group] += 1
        selected_by_source[source] += 1
        return True

    token_order = {"source_split": 0, "label_split": 1, "element_split": 2, "shape_split": 3}
    for token in sorted(
        token_candidates,
        key=lambda value: (token_order.get(str(value[0]), 99), str(value)),
    ):
        for index in sorted(token_candidates[token], key=lambda value: rank_values[value]):
            if add(index):
                break
    if len(selected) > requested:
        raise ValueError(
            f"Coverage constraints require {len(selected)} structures, above target {requested}"
        )

    source_quotas = dict(desired_source_quotas)
    for source, count in selected_by_source.items():
        source_quotas[source] = max(source_quotas.get(source, 0), int(count))
    excess = sum(source_quotas.values()) - requested
    while excess > 0:
        reducible = [
            source for source in source_quotas
            if source_quotas[source] > selected_by_source.get(source, 0)
        ]
        if not reducible:
            raise ValueError("Coverage constraints cannot fit the requested source quotas")
        source = max(
            reducible,
            key=lambda value: (
                source_quotas[value] - selected_by_source.get(value, 0),
                source_quotas[value],
                str(value),
            ),
        )
        source_quotas[source] -= 1
        excess -= 1

    for source in sorted(source_strata):
        needed = source_quotas.get(source, 0) - selected_by_source.get(source, 0)
        if needed <= 0:
            continue
        buckets = source_strata[source]
        capacities = {
            key: sum(1 for index in values if index not in selected)
            for key, values in buckets.items()
        }
        capacities = {key: value for key, value in capacities.items() if value > 0}
        bucket_quotas = _allocate_bounded_temperature_quotas(
            capacities,
            needed,
            temperature=1.0,
            minimum=1 if needed >= len(capacities) else 0,
        )
        for key in sorted(bucket_quotas, key=lambda value: str(value)):
            remaining = bucket_quotas[key]
            for index in sorted(buckets[key], key=lambda value: rank_values[value]):
                if remaining <= 0:
                    break
                was_selected = index in selected
                if add(index) and not was_selected:
                    remaining -= 1
        remaining = source_quotas.get(source, 0) - selected_by_source.get(source, 0)
        if remaining > 0:
            all_source = sorted(
                (index for values in buckets.values() for index in values),
                key=lambda value: rank_values[value],
            )
            for index in all_source:
                if remaining <= 0:
                    break
                was_selected = index in selected
                if add(index) and not was_selected:
                    remaining -= 1

    if len(selected) < requested:
        for index in sorted(range(structure_count), key=lambda value: rank_values[value]):
            if len(selected) >= requested:
                break
            add(index)
    if len(selected) != requested:
        raise ValueError(
            f"Group cap selected {len(selected)} structures, expected {requested}; "
            "increase max_per_group"
        )

    selected_indices = sorted(int(value) for value in selected)
    selected_nonrare_groups: Dict[str, int] = collections.Counter()
    selected_rare_groups: Dict[str, int] = collections.Counter()
    for index in selected_indices:
        target = (
            selected_rare_groups
            if rare_source_limit and raw_source_counts[sources[index]] <= rare_source_limit
            else selected_nonrare_groups
        )
        target[groups[index]] += 1
    selected_sources = dict(collections.Counter(sources[index] for index in selected_indices))
    selected_splits = dict(collections.Counter(splits[index] for index in selected_indices))
    selected_complexity = dict(collections.Counter(complexity[index] for index in selected_indices))
    selected_sizes = dict(collections.Counter(size_class[index] for index in selected_indices))
    selected_families = {
        family: int(sum(bool(active[index]) for index in selected_indices))
        for family, active in family_active.items()
    }
    parent_splits = dict(collections.Counter(splits))
    parent_complexity = dict(collections.Counter(complexity))
    parent_sizes = dict(collections.Counter(size_class))
    elements_by_split: Dict[str, set] = collections.defaultdict(set)
    for index in selected_indices:
        elements_by_split[splits[index]].update(unique_elements[index])
    train_elements = set(elements_by_split.get("train", set()))
    evaluation_elements = set(elements_by_split.get("val", set())) | set(
        elements_by_split.get("test", set())
    )
    all_parent_elements = set(int(value) for value in np.unique(atomic_numbers))
    all_selected_elements = set().union(*elements_by_split.values()) if elements_by_split else set()
    report = {
        "strategy": "deterministic_temperature_source_and_coverage_stratification",
        "seed": int(seed),
        "target_structures": requested,
        "selected_structures": len(selected_indices),
        "source_temperature": float(source_temperature),
        "min_per_source": int(min_per_source),
        "max_per_group": group_limit,
        "preserve_sources_up_to": rare_source_limit,
        "fully_preserved_sources": sorted(
            source for source, count in raw_source_counts.items()
            if rare_source_limit and count <= rare_source_limit
        ),
        "groups": len(selected_by_group),
        "max_selected_per_group": max(selected_by_group.values()) if selected_by_group else 0,
        "max_selected_per_nonrare_group": (
            max(selected_nonrare_groups.values()) if selected_nonrare_groups else 0
        ),
        "max_selected_per_fully_preserved_group": (
            max(selected_rare_groups.values()) if selected_rare_groups else 0
        ),
        "selected_sample_ids_unique": len({samples[index] for index in selected_indices})
        == len(selected_indices),
        "parent_source_counts": dict(sorted(raw_source_counts.items())),
        "source_selection_capacities": dict(sorted(source_capacities.items())),
        "target_source_quotas": dict(sorted(source_quotas.items())),
        "selected_source_counts": dict(sorted(selected_sources.items())),
        "parent_splits": dict(sorted(parent_splits.items())),
        "selected_splits": dict(sorted(selected_splits.items())),
        "selected_label_families": dict(sorted(selected_families.items())),
        "selected_complexity": dict(sorted(selected_complexity.items())),
        "selected_size_classes": dict(sorted(selected_sizes.items())),
        "source_js_to_temperature_target_bits": _distribution_js_divergence(
            selected_sources, source_quotas
        ),
        "split_js_to_parent_bits": _distribution_js_divergence(selected_splits, parent_splits),
        "complexity_js_to_parent_bits": _distribution_js_divergence(
            selected_complexity, parent_complexity
        ),
        "size_js_to_parent_bits": _distribution_js_divergence(selected_sizes, parent_sizes),
        "parent_elements": sorted(all_parent_elements),
        "selected_elements": sorted(all_selected_elements),
        "element_coverage_fraction": (
            len(all_selected_elements) / len(all_parent_elements) if all_parent_elements else 1.0
        ),
        "elements_by_split": {
            split_name: sorted(values) for split_name, values in sorted(elements_by_split.items())
        },
        "evaluation_elements_missing_from_train": sorted(evaluation_elements - train_elements),
    }
    return selected_indices, report


def build_neo_stratified_tier(
    input_path: str,
    output_path: str,
    *,
    target_mib: float,
    seed: int = 20260720,
    source_temperature: float = 0.5,
    min_per_source: int = 3,
    max_per_group: int = 2,
    preserve_sources_up_to: int = 128,
    tolerance_fraction: float = 0.05,
    max_calibration_rounds: int = 4,
    overwrite: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a size-calibrated Neo tier without changing any selected labels."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("Tier output must not overwrite its parent dataset")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    target_bytes = int(round(float(target_mib) * 1024.0 * 1024.0))
    if target_bytes <= 0:
        raise ValueError("target_mib must be positive")
    with h5py.File(source, "r") as handle:
        parent_structures = int(len(handle["structures/atom_ptr"]) - 1)
    parent_bytes = int(source.stat().st_size)
    target_structures = max(
        1,
        min(parent_structures, int(round(parent_structures * target_bytes / parent_bytes))),
    )
    attempts: List[Dict[str, Any]] = []
    final_selection: Dict[str, Any] = {}
    rounds = max(1, int(max_calibration_rounds))
    for attempt in range(1, rounds + 1):
        indices, selection = select_neo_stratified_hdf5_indices(
            str(source),
            target_structures=target_structures,
            seed=int(seed),
            source_temperature=float(source_temperature),
            min_per_source=int(min_per_source),
            max_per_group=int(max_per_group),
            preserve_sources_up_to=int(preserve_sources_up_to),
        )
        metadata = {
            "dataset": f"Neo size-calibrated stratified tier ({float(target_mib):g} MiB target)",
            "corpus_role": "portable-stratified-tier",
            "parent_path": str(source),
            "parent_sha256": sha256_file(str(source)),
            "target_mib": float(target_mib),
            "selection": selection,
            "missing_label_policy": "Preserve parent masks and values; never synthesize labels",
        }
        write_hdf5_dataset_stream(
            iter_hdf5_configurations(str(source), selected_indices=indices),
            str(destination),
            metadata=metadata,
            overwrite=bool(overwrite or attempt > 1),
        )
        actual_bytes = int(destination.stat().st_size)
        relative_error = (actual_bytes - target_bytes) / target_bytes
        attempts.append({
            "round": attempt,
            "structures": len(indices),
            "bytes": actual_bytes,
            "mib": actual_bytes / (1024.0 * 1024.0),
            "relative_size_error": relative_error,
        })
        final_selection = selection
        if abs(relative_error) <= max(0.0, float(tolerance_fraction)):
            break
        proposed = int(round(len(indices) * target_bytes / max(1, actual_bytes)))
        proposed = max(1, min(parent_structures, proposed))
        if proposed == target_structures:
            proposed += -1 if actual_bytes > target_bytes else 1
            proposed = max(1, min(parent_structures, proposed))
        if proposed == target_structures:
            break
        target_structures = proposed
    validation = validate_neo_hdf5(str(destination))
    result = {
        "output": str(destination),
        "parent": str(source),
        "target_mib": float(target_mib),
        "actual_bytes": int(destination.stat().st_size),
        "actual_mib": destination.stat().st_size / (1024.0 * 1024.0),
        "attempts": attempts,
        "selection": final_selection,
        "validation": validation,
    }
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def audit_neo_tier_hierarchy(
    tiers: Sequence[Tuple[str, str]],
    *,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit ordered nested tiers from smallest through the reference corpus."""
    if not tiers:
        raise ValueError("At least one dataset tier is required")
    tier_reports: List[Dict[str, Any]] = []
    sample_sets: List[set] = []
    for name, raw_path in tiers:
        path = Path(raw_path).expanduser().resolve()
        validation = validate_neo_hdf5(str(path))
        with h5py.File(path, "r") as handle:
            sample_ids = [str(value) for value in handle["metadata/sample_id"].asstr()[:]]
            splits = [str(value) for value in handle["metadata/split"].asstr()[:]]
            atom_ptr = np.asarray(handle["structures/atom_ptr"], dtype=np.int64)
            atomic_numbers = np.asarray(
                handle["structures/atomic_numbers"], dtype=np.int16
            )
        ids = set(sample_ids)
        sample_sets.append(ids)
        elements_by_split: Dict[str, set] = collections.defaultdict(set)
        for index, split_name in enumerate(splits):
            elements_by_split[split_name].update(
                int(value)
                for value in np.unique(
                    atomic_numbers[int(atom_ptr[index]) : int(atom_ptr[index + 1])]
                )
            )
        train_elements = set(elements_by_split.get("train", set()))
        evaluation_elements = set(elements_by_split.get("val", set())) | set(
            elements_by_split.get("test", set())
        )
        tier_reports.append({
            "name": str(name),
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "decimal_mb": path.stat().st_size / 1_000_000.0,
            "mib": path.stat().st_size / (1024.0 * 1024.0),
            "structures": int(validation["structures"]),
            "atoms": int(validation["atoms"]),
            "groups": int(validation["groups"]),
            "sha256": str(validation["sha256"]),
            "valid": bool(validation["valid"]),
            "errors": list(validation["errors"]),
            "warnings": list(validation["warnings"]),
            "sample_ids_unique": len(ids) == len(sample_ids),
            "sources": dict(validation["sources"]),
            "active_labels": sorted(
                name for name, count in validation["label_counts"].items() if int(count) > 0
            ),
            "elements": list(validation["elements"]),
            "elements_by_split": {
                split_name: sorted(values)
                for split_name, values in sorted(elements_by_split.items())
            },
            "evaluation_elements_missing_from_train": sorted(
                evaluation_elements - train_elements
            ),
            "splits": dict(validation["splits"]),
            "composition": dict(validation["composition"]),
        })
    nesting: List[Dict[str, Any]] = []
    for index in range(len(tiers) - 1):
        missing = sorted(sample_sets[index] - sample_sets[index + 1])
        nesting.append({
            "subset": str(tiers[index][0]),
            "superset": str(tiers[index + 1][0]),
            "subset_samples": len(sample_sets[index]),
            "matched_samples": len(sample_sets[index]) - len(missing),
            "missing_samples": len(missing),
            "first_missing_sample_ids": missing[:10],
            "is_nested": not missing,
        })
    reference_sources = set(tier_reports[-1]["sources"])
    reference_labels = set(tier_reports[-1]["active_labels"])
    for report in tier_reports:
        report["source_coverage_fraction"] = (
            len(set(report["sources"]) & reference_sources) / len(reference_sources)
            if reference_sources else 1.0
        )
        report["active_label_coverage_fraction"] = (
            len(set(report["active_labels"]) & reference_labels) / len(reference_labels)
            if reference_labels else 1.0
        )
    result = {
        "strategy": "ordered_nested_tier_audit",
        "tiers": tier_reports,
        "nesting": nesting,
        "all_valid": all(report["valid"] for report in tier_reports),
        "all_sample_ids_unique": all(
            report["sample_ids_unique"] for report in tier_reports
        ),
        "all_nested": all(item["is_nested"] for item in nesting),
        "all_reference_sources_covered": all(
            math.isclose(float(report["source_coverage_fraction"]), 1.0)
            for report in tier_reports
        ),
        "all_reference_active_labels_covered": all(
            math.isclose(float(report["active_label_coverage_fraction"]), 1.0)
            for report in tier_reports
        ),
        "all_evaluation_elements_seen_in_train": all(
            not report["evaluation_elements_missing_from_train"]
            for report in tier_reports
        ),
        "generalization_scope": (
            "Coverage and leakage audit only; predictive generalization requires "
            "converged held-out model evaluation."
        ),
    }
    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def _copy_configuration_with_label_policy(
    cfg: Configuration,
    *,
    keep_labels: set,
    source_prefix: str,
) -> Configuration:
    props = dict(cfg.properties)
    weights = dict(cfg.property_weights)
    for name in itertools.chain(HDF5_STRUCTURE_LABELS, HDF5_ATOM_LABELS):
        if name not in keep_labels:
            props.pop(name, None)
            weights.pop(name, None)
    original_group = str(props.get("group_id", "unknown"))
    original_sample = str(props.get("sample_id", "unknown"))
    # Canonical source shards already namespace their IDs. Preserving MPtrj IDs
    # across the static and magnetic shards prevents material-family leakage.
    props["group_id"] = original_group
    props["sample_id"] = original_sample
    if str(props.get("split", "")).lower() not in ("train", "val", "test"):
        props["split"] = stable_split(props["group_id"])
    return Configuration(
        atomic_numbers=np.asarray(cfg.atomic_numbers, dtype=int),
        positions=np.asarray(cfg.positions, dtype=float),
        properties=props,
        property_weights=weights,
        cell=np.asarray(cfg.cell, dtype=float),
        pbc=tuple(bool(value) for value in cfg.pbc),
        weight=cfg.weight,
        config_type=cfg.config_type,
        head=cfg.head,
    )


def build_neo_mixed_dataset(
    source_specs: Sequence[Dict[str, Any]],
    output_path: str,
    *,
    seed: int = 20260719,
    overwrite: bool = False,
    dataset_name: str = "Neo balanced L1-L3 mixed-granularity training corpus",
    corpus_role: str = "balanced-training",
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a directly trainable mixed corpus with explicit source label policies."""
    source_report: List[Dict[str, Any]] = []
    prepared_sources: List[Tuple[str, str, List[int], set]] = []
    for source_index, raw_spec in enumerate(source_specs):
        spec = dict(raw_spec)
        path = str(Path(spec["path"]).expanduser().resolve())
        source_prefix = str(spec.get("name", f"source-{source_index}"))
        indices = _ranked_hdf5_indices(
            path,
            target_structures=spec.get("target_structures"),
            target_groups=spec.get("target_groups"),
            max_per_group=spec.get("max_per_group"),
            seed=int(seed) + source_index,
        )
        keep_labels = set(str(name) for name in spec.get("keep_labels", []))
        prepared_sources.append((source_prefix, path, indices, keep_labels))
        source_report.append({
            "name": source_prefix,
            "path": path,
            "sha256": sha256_file(path),
            "selected_indices": len(indices),
            "structures_kept": 0,
            "duplicate_sample_ids_skipped": 0,
            "max_per_group": spec.get("max_per_group"),
            "target_groups": spec.get("target_groups"),
            "target_structures": spec.get("target_structures"),
            "keep_labels": sorted(keep_labels),
            "deduplicate_by": str(spec.get("deduplicate_by", "sample_id")),
            "label_counts": {},
        })
    if not any(item["selected_indices"] for item in source_report):
        raise ValueError("Neo mixed build selected no configurations")

    def configurations() -> Iterable[Configuration]:
        emitted_keys: set = set()
        for source_index, (source_prefix, path, indices, keep_labels) in enumerate(prepared_sources):
            spec = dict(source_specs[source_index])
            report = source_report[source_index]
            duplicate_key_name = str(spec.get("deduplicate_by", "sample_id"))
            for cfg in iter_hdf5_configurations(path, selected_indices=indices):
                filtered = _copy_configuration_with_label_policy(
                    cfg, keep_labels=keep_labels, source_prefix=source_prefix
                )
                duplicate_key = str(filtered.properties.get(duplicate_key_name, ""))
                if duplicate_key in emitted_keys:
                    report["duplicate_sample_ids_skipped"] += 1
                    continue
                emitted_keys.add(duplicate_key)
                sample_id = str(filtered.properties["sample_id"])
                _validate_configuration(filtered, context=f"Neo mixed/{sample_id}")
                report["structures_kept"] += 1
                label_counts = report["label_counts"]
                for name in itertools.chain(HDF5_STRUCTURE_LABELS, HDF5_ATOM_LABELS):
                    if name in filtered.properties and float(
                        filtered.property_weights.get(name, 1.0)
                    ) > 0.0:
                        label_counts[name] = int(label_counts.get(name, 0)) + 1
                yield filtered

    build_metadata: Dict[str, Any] = {
        "dataset": str(dataset_name),
        "corpus_role": str(corpus_role),
        "manifest_version": NEO_MANIFEST_VERSION,
        "seed": int(seed),
        "source_policies": source_report,
        "energy_policy": (
            "Only sources explicitly retaining energy supervise the shared energy loss; "
            "incompatible absolute references are masked."
        ),
        "domain_policy": (
            "Default mixed preset uses QEq/Spin/FiLM; PME, DEQ, and molecular D4 "
            "remain disabled and are trained with domain-specific source shards."
        ),
    }
    if metadata_extra:
        build_metadata.update(dict(metadata_extra))
    output = write_hdf5_dataset_stream(
        configurations(),
        output_path,
        metadata=build_metadata,
        overwrite=overwrite,
    )
    summary = hdf5_dataset_summary(output)
    result = {
        "output": output,
        "summary": summary,
        "sources": source_report,
    }
    return result


def build_neo_smoke_dataset(
    source_specs: Sequence[Dict[str, Any]],
    output_path: str,
    *,
    per_source_split: int = 2,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build a tiny deterministic corpus with every source represented per split."""
    selected: List[Configuration] = []
    source_report: List[Dict[str, Any]] = []
    count = max(1, int(per_source_split))
    for source_index, raw_spec in enumerate(source_specs):
        spec = dict(raw_spec)
        path = str(Path(spec["path"]).expanduser().resolve())
        keep_labels = set(str(name) for name in spec.get("keep_labels", []))
        by_split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        with h5py.File(path, "r") as handle:
            split_values = handle["metadata/split"].asstr()[:]
            groups = handle["metadata/group_id"].asstr()[:]
            samples = handle["metadata/sample_id"].asstr()[:]
            masks = handle["masks"]
            useful = np.zeros((len(split_values),), dtype=bool)
            for label in keep_labels:
                if label in masks:
                    useful |= np.asarray(masks[label], dtype=bool)
            indices = list(range(len(split_values)))
            indices.sort(
                key=lambda index: hashlib.sha256(
                    f"neo-smoke|{source_index}|{groups[index]}|{samples[index]}".encode()
                ).hexdigest()
            )
            for index in indices:
                split_name = str(split_values[index])
                if split_name not in by_split or not useful[index]:
                    continue
                if len(by_split[split_name]) < count:
                    by_split[split_name].append(index)
        source_counts: Dict[str, int] = {}
        for split_name, indices in by_split.items():
            for cfg in iter_hdf5_configurations(path, selected_indices=indices):
                filtered = _copy_configuration_with_label_policy(
                    cfg,
                    keep_labels=keep_labels,
                    source_prefix=str(spec.get("name", f"source-{source_index}")),
                )
                filtered.properties["split"] = split_name
                selected.append(filtered)
                source_counts[split_name] = source_counts.get(split_name, 0) + 1
        source_report.append({
            "name": str(spec.get("name", f"source-{source_index}")),
            "path": path,
            "keep_labels": sorted(keep_labels),
            "splits": source_counts,
        })
    output = write_hdf5_dataset_stream(
        selected,
        output_path,
        metadata={
            "dataset": "Neo balanced rare-target functional smoke corpus",
            "source_policies": source_report,
            "purpose": "forward/backward, short-training, and memory-regression checks",
        },
        overwrite=overwrite,
    )
    validation = validate_neo_hdf5(output)
    return {"output": output, "validation": validation, "sources": source_report}


def validate_neo_hdf5(path: str) -> Dict[str, Any]:
    """Validate finite labels, masks, group splits, charge sums, and spin geometry."""
    if not HAS_H5PY:
        raise RuntimeError("Neo validation requires h5py")
    with h5py.File(path, "r") as schema_handle:
        if str(schema_handle.attrs.get("schema_version", "")) == COMPOSITE_HDF5_SCHEMA_VERSION:
            inspected = inspect_composite_dataset(str(path), verify_sources=False)
            return {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": sha256_file(path),
                "structures": int(inspected.get("structures", 0)),
                "atoms": int(inspected.get("atoms", 0)),
                "groups": 0,
                "splits": dict(inspected.get("splits", {})),
                "sources": dict(inspected.get("sources", {})),
                "labels": dict(inspected.get("labels", {})),
                "valid": bool(inspected.get("valid", False)),
                "errors": list(inspected.get("embedded_errors", []))
                + list(inspected.get("missing_sources", []))
                + list(inspected.get("checksum_mismatches", [])),
                "warnings": [
                    "Composite validation checks embedded Large and OMat24 selector/shard structure; "
                    "full numerical shard hashing is available through dataset-composite-validate."
                ],
                "embedded_large": bool(inspected.get("embedded_large", False)),
                "embedded_omat24": bool(inspected.get("embedded_omat24", False)),
            }
    report: Dict[str, Any] = {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": sha256_file(path),
        "errors": [],
        "warnings": [],
    }
    with h5py.File(path, "r") as handle:
        ptr = np.asarray(handle["structures/atom_ptr"], dtype=np.int64)
        numbers = np.asarray(handle["structures/atomic_numbers"], dtype=int)
        positions = np.asarray(handle["structures/positions"], dtype=float)
        pbc = np.asarray(handle["structures/pbc"], dtype=bool)
        groups = [str(value) for value in handle["metadata/group_id"].asstr()[:]]
        splits = [str(value) for value in handle["metadata/split"].asstr()[:]]
        sources = [str(value) for value in handle["metadata/source"].asstr()[:]]
        structure_count = len(ptr) - 1
        if ptr.shape != (structure_count + 1,) or ptr[0] != 0 or ptr[-1] != len(numbers):
            report["errors"].append("atom_ptr is inconsistent with atomic_numbers")
        if np.any(np.diff(ptr) <= 0):
            report["errors"].append("one or more structures have no atoms")
        if positions.shape != (len(numbers), 3) or not np.isfinite(positions).all():
            report["errors"].append("positions are non-finite or have the wrong shape")
        if np.any(numbers <= 0) or np.any(numbers >= len(ASE_CHEMICAL_SYMBOLS)):
            report["errors"].append("atomic numbers are outside the ASE element table")
        group_split: Dict[str, str] = {}
        split_conflicts: List[str] = []
        for group, split_name in zip(groups, splits):
            previous = group_split.setdefault(group, split_name)
            if previous != split_name:
                split_conflicts.append(group)
        if split_conflicts:
            report["errors"].append(
                f"{len(set(split_conflicts))} group IDs cross split boundaries"
            )
        label_counts: Dict[str, int] = {}
        for name in handle["masks"].keys():
            mask = np.asarray(handle[f"masks/{name}"], dtype=bool)
            label_counts[name] = int(mask.sum())
            if mask.shape != (structure_count,):
                report["errors"].append(f"mask {name} has the wrong shape")
                continue
            values = handle[f"labels/{name}"]
            if name in HDF5_STRUCTURE_LABELS:
                labeled = np.asarray(values[mask], dtype=float)
            else:
                atom_mask = np.zeros((len(numbers),), dtype=bool)
                for index in np.flatnonzero(mask):
                    atom_mask[int(ptr[index]) : int(ptr[index + 1])] = True
                labeled = np.asarray(values[atom_mask], dtype=float)
            if labeled.size and not np.isfinite(labeled).all():
                report["errors"].append(f"label {name} contains non-finite values under its mask")
        charge_residual_max = 0.0
        if label_counts.get("charges", 0) and label_counts.get("total_charge", 0):
            charge_mask = np.asarray(handle["masks/charges"], dtype=bool)
            total_mask = np.asarray(handle["masks/total_charge"], dtype=bool)
            charge_values = handle["labels/charges"]
            total_values = handle["labels/total_charge"]
            for index in np.flatnonzero(charge_mask & total_mask):
                residual = abs(
                    float(np.asarray(charge_values[int(ptr[index]) : int(ptr[index + 1])]).sum())
                    - float(total_values[index])
                )
                charge_residual_max = max(charge_residual_max, residual)
            if charge_residual_max > 0.01:
                report["warnings"].append(
                    f"Hirshfeld charge sums differ from integer total charge by up to {charge_residual_max:.6g} e"
                )
        spin_norm_error = 0.0
        spin_mask_count = label_counts.get("spins", 0)
        if spin_mask_count:
            spin_mask = np.asarray(handle["masks/spins"], dtype=bool)
            spins = handle["labels/spins"]
            for index in np.flatnonzero(spin_mask):
                values = np.asarray(spins[int(ptr[index]) : int(ptr[index + 1])], dtype=float)
                norms = np.linalg.norm(values, axis=1)
                active = norms > 1e-12
                if np.any(active):
                    spin_norm_error = max(
                        spin_norm_error, float(np.max(np.abs(norms[active] - 1.0)))
                    )
        bec_acoustic_sum_residual = 0.0
        if label_counts.get("bec", 0):
            bec_mask = np.asarray(handle["masks/bec"], dtype=bool)
            bec_values = handle["labels/bec"]
            for index in np.flatnonzero(bec_mask):
                tensor_sum = np.asarray(
                    bec_values[int(ptr[index]) : int(ptr[index + 1])], dtype=float
                ).sum(axis=0)
                bec_acoustic_sum_residual = max(
                    bec_acoustic_sum_residual,
                    float(np.max(np.abs(tensor_sum))),
                )
            if bec_acoustic_sum_residual > 1.0:
                report["warnings"].append(
                    "Published BEC tensors violate the acoustic sum rule by up to "
                    f"{bec_acoustic_sum_residual:.6g} e; labels were not projected or altered"
                )
        report.update({
            "valid": not report["errors"],
            "structures": structure_count,
            "atoms": int(len(numbers)),
            "elements": sorted(int(value) for value in np.unique(numbers)),
            "periodic_structures": int(np.count_nonzero(np.any(pbc, axis=1))),
            "sources": dict(sorted(collections.Counter(sources).items())),
            "splits": dict(sorted(collections.Counter(splits).items())),
            "groups": len(group_split),
            "label_counts": label_counts,
            "charge_sum_residual_max_e": charge_residual_max,
            "active_spin_norm_max_error": spin_norm_error,
            "bec_acoustic_sum_residual_max_e": bec_acoustic_sum_residual,
            "composition": _composition_statistics(ptr, numbers),
        })
    return report


def _composition_statistics(atom_ptr: np.ndarray, numbers: np.ndarray) -> Dict[str, Any]:
    """Summarize per-structure chemical and size complexity."""
    complexity = {"unary": 0, "binary": 0, "ternary": 0, "quaternary+": 0}
    sizes = {"small_1_24": 0, "medium_25_64": 0, "large_65_plus": 0}
    atom_counts = np.diff(np.asarray(atom_ptr, dtype=np.int64))
    element_frequency: Dict[int, int] = {}
    for index, atom_count in enumerate(atom_counts):
        start, end = int(atom_ptr[index]), int(atom_ptr[index + 1])
        unique = np.unique(numbers[start:end])
        element_count = len(unique)
        key = (
            "unary" if element_count == 1
            else "binary" if element_count == 2
            else "ternary" if element_count == 3
            else "quaternary+"
        )
        complexity[key] += 1
        size_key = (
            "small_1_24" if atom_count <= 24
            else "medium_25_64" if atom_count <= 64
            else "large_65_plus"
        )
        sizes[size_key] += 1
        for atomic_number in unique:
            value = int(atomic_number)
            element_frequency[value] = element_frequency.get(value, 0) + 1
    return {
        "complexity": complexity,
        "size_classes": sizes,
        "multi_element_structures": int(len(atom_counts) - complexity["unary"]),
        "atom_count_min": int(atom_counts.min()) if len(atom_counts) else 0,
        "atom_count_median": float(np.median(atom_counts)) if len(atom_counts) else 0.0,
        "atom_count_max": int(atom_counts.max()) if len(atom_counts) else 0,
        "element_structure_frequency": {
            ASE_CHEMICAL_SYMBOLS[value]: int(count)
            for value, count in sorted(element_frequency.items())
        },
    }


def hdf5_dataset_summary(path: str) -> Dict[str, Any]:
    if not HAS_H5PY:
        raise RuntimeError("HDF5 support requires h5py")
    with h5py.File(path, "r") as handle:
        ptr = np.asarray(handle["structures/atom_ptr"], dtype=np.int64)
        z = np.asarray(handle["structures/atomic_numbers"], dtype=int)
        pbc = np.asarray(handle["structures/pbc"], dtype=bool)
        masks = handle["masks"]
        groups = handle["metadata/group_id"].asstr()[:]
        sources = handle["metadata/source"].asstr()[:]
        return {
            "schema_version": str(handle.attrs.get("schema_version", "")),
            "structures": int(len(ptr) - 1),
            "atoms": int(ptr[-1]),
            "elements": sorted(int(x) for x in np.unique(z)),
            "labels": {name: int(np.asarray(masks[name], dtype=bool).sum()) for name in masks.keys()},
            "splits": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(handle["metadata/split"].asstr()[:], return_counts=True)
                )
            },
            "groups": len(set(str(value) for value in groups)),
            "periodic_structures": int(np.count_nonzero(np.any(pbc, axis=1))),
            "sources": dict(sorted(collections.Counter(str(value) for value in sources).items())),
            "composition": _composition_statistics(ptr, z),
            "sha256": sha256_file(path),
        }


def _select_balanced_half_composite_indices(
    source_orders: np.ndarray,
    row_indices: np.ndarray,
    split_codes: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Select exactly one row per adjacent pair in every source/split stratum."""
    sources = np.asarray(source_orders, dtype=np.uint16).reshape(-1)
    rows = np.asarray(row_indices, dtype=np.uint32).reshape(-1)
    splits = np.asarray(split_codes, dtype=np.uint8).reshape(-1)
    if not (sources.size == rows.size == splits.size):
        raise ValueError("Composite selector arrays have inconsistent lengths")
    if sources.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(sources[1:] < sources[:-1]):
        raise ValueError("Composite selectors must be ordered by source_order")

    selected = np.zeros((sources.size,), dtype=np.bool_)
    boundaries = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(sources[1:] != sources[:-1]).astype(np.int64) + 1,
        np.asarray([sources.size], dtype=np.int64),
    ))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        source = int(sources[int(start)])
        local_splits = splits[int(start):int(end)]
        for split_code in np.unique(local_splits):
            candidates = int(start) + np.flatnonzero(local_splits == split_code)
            pair_count = int(candidates.size // 2)
            if pair_count:
                pairs = candidates[: 2 * pair_count].reshape(pair_count, 2)
                keys = (
                    rows[pairs[:, 0]].astype(np.uint64)
                    ^ np.uint64(source << 32)
                    ^ np.uint64(int(split_code) << 56)
                    ^ np.uint64(int(seed) & ((1 << 64) - 1))
                )
                with np.errstate(over="ignore"):
                    keys += np.uint64(0x9E3779B97F4A7C15)
                    keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(
                        0xBF58476D1CE4E5B9
                    )
                    keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(
                        0x94D049BB133111EB
                    )
                    keys ^= keys >> np.uint64(31)
                choices = np.asarray(keys & np.uint64(1), dtype=np.int64)
                selected[pairs[np.arange(pair_count), choices]] = True
            if candidates.size % 2:
                selected[int(candidates[-1])] = True
    return np.flatnonzero(selected).astype(np.int64, copy=False)


def _allocate_composite_fraction_quotas(
    stratum_counts: np.ndarray,
    *,
    denominator: int,
    seed: int,
) -> np.ndarray:
    """Allocate an exact proportional sample across source/split strata."""
    counts = np.asarray(stratum_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 3 or np.any(counts < 0):
        raise ValueError("Composite stratum counts must have shape (sources, 3)")
    divisor = int(denominator)
    if divisor < 2:
        raise ValueError("Composite sampling denominator must be at least 2")
    total = int(np.sum(counts, dtype=np.int64))
    target = int(round(total / float(divisor)))
    quotas = counts // divisor
    remaining = target - int(np.sum(quotas, dtype=np.int64))
    if remaining < 0:
        raise RuntimeError("Composite quota allocation exceeded its target")
    ranked: List[Tuple[int, str, int, int]] = []
    remainders = counts % divisor
    for source_order in range(counts.shape[0]):
        for split_code in range(3):
            if quotas[source_order, split_code] >= counts[source_order, split_code]:
                continue
            tie_break = hashlib.sha256(
                f"neo-plus-half-quota-v1|{int(seed)}|{source_order}|{split_code}".encode()
            ).hexdigest()
            ranked.append((
                -int(remainders[source_order, split_code]),
                tie_break,
                source_order,
                split_code,
            ))
    ranked.sort()
    if remaining > len(ranked):
        raise RuntimeError("Composite quota allocation cannot reach its target")
    for _remainder, _tie, source_order, split_code in ranked[:remaining]:
        quotas[source_order, split_code] += 1
    if int(np.sum(quotas, dtype=np.int64)) != target:
        raise RuntimeError("Composite quota allocation is not exact")
    return quotas


def _stratified_interval_sample(
    candidate_count: int,
    target_count: int,
    *,
    seed: int,
    source_order: int,
    split_code: int,
) -> np.ndarray:
    """Choose one deterministic pseudo-random row from each equal-width interval."""
    count = int(candidate_count)
    target = int(target_count)
    if target < 0 or target > count:
        raise ValueError("Invalid interval-sample target")
    if target == 0:
        return np.zeros((0,), dtype=np.int64)
    if target == count:
        return np.arange(count, dtype=np.int64)
    index = np.arange(target, dtype=np.int64)
    starts = (index * count) // target
    ends = ((index + 1) * count) // target
    widths = ends - starts
    base = int(hashlib.sha256(
        f"neo-plus-half-row-v1|{int(seed)}|{int(source_order)}|{int(split_code)}".encode()
    ).hexdigest()[:16], 16)
    keys = index.astype(np.uint64) ^ np.uint64(base)
    with np.errstate(over="ignore"):
        keys += np.uint64(0x9E3779B97F4A7C15)
        keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        keys ^= keys >> np.uint64(31)
    offsets = np.asarray(keys % widths.astype(np.uint64), dtype=np.int64)
    return starts + offsets


def _write_packed_omat24_selection(
    parent: Path,
    output: Path,
    *,
    selected_local_rows: np.ndarray,
    selected_sources: np.ndarray,
    selected_atom_counts: np.ndarray,
) -> None:
    """Replace linked OMat24 shards with compact lossless packed arrays."""
    total = int(len(selected_local_rows))
    total_atoms = int(np.sum(selected_atom_counts, dtype=np.int64))
    with h5py.File(parent, "r") as source, h5py.File(output, "r+") as target:
        source_payload = source["sources/omat24/embedded_parquet"]
        source_names = [str(value) for value in source_payload["shard_names"].asstr()[:]]
        omat = target["sources/omat24"]
        if "embedded_parquet" in omat:
            del omat["embedded_parquet"]
        omat.attrs["embedded"] = True
        omat.attrs["materialized"] = True
        omat.attrs["storage"] = OMAT24_PACKED_SCHEMA_VERSION
        omat.attrs["method_id"] = "OMat24/PBE+U"
        packed = omat.create_group("packed")
        packed.attrs["schema"] = OMAT24_PACKED_SCHEMA_VERSION
        packed.attrs["structures"] = total
        packed.attrs["atoms"] = total_atoms
        packed.attrs["precision_policy"] = (
            "Source float64 geometry and labels retained without quantization"
        )
        atom_ptr = np.zeros((total + 1,), dtype=np.int64)
        np.cumsum(selected_atom_counts.astype(np.int64), out=atom_ptr[1:])
        packed.create_dataset("atom_ptr", data=atom_ptr, compression="gzip")
        packed.create_dataset(
            "atomic_numbers", shape=(total_atoms,), dtype=np.int16,
            chunks=(min(1_000_000, max(1, total_atoms)),), compression="gzip",
            compression_opts=6,
        )
        for name in ("positions", "forces"):
            packed.create_dataset(
                name, shape=(total_atoms, 3), dtype=np.float64,
                chunks=(min(100_000, max(1, total_atoms)), 3),
                compression="gzip", compression_opts=6, shuffle=True,
            )
        packed.create_dataset(
            "cell", shape=(total, 3, 3), dtype=np.float64,
            chunks=(min(4096, max(1, total)), 3, 3),
            compression="gzip", compression_opts=6, shuffle=True,
        )
        packed.create_dataset(
            "pbc", shape=(total, 3), dtype=np.bool_,
            chunks=(min(32768, max(1, total)), 3), compression="gzip",
        )
        packed.create_dataset(
            "energy", shape=(total,), dtype=np.float64,
            chunks=(min(32768, max(1, total)),),
            compression="gzip", compression_opts=6, shuffle=True,
        )
        packed.create_dataset(
            "stress", shape=(total, 3, 3), dtype=np.float64,
            chunks=(min(4096, max(1, total)), 3, 3),
            compression="gzip", compression_opts=6, shuffle=True,
        )
        packed.create_dataset(
            "stress_volume_normalized", shape=(total,), dtype=np.bool_,
            chunks=(min(32768, max(1, total)),), compression="gzip",
        )
        packed.create_dataset(
            "source_row_index",
            data=np.asarray(target["selection/source_row_index"][:], dtype=np.uint32),
            compression="gzip",
        )
        string_dtype = h5py.string_dtype("utf-8")
        packed.create_dataset("configuration_id", shape=(total,), dtype=string_dtype)
        packed.create_dataset("material_id", shape=(total,), dtype=string_dtype)

        cursor = 0
        atom_cursor = 0
        while cursor < total:
            source_order = int(selected_sources[cursor])
            end = cursor + 1
            while end < total and int(selected_sources[end]) == source_order:
                end += 1
            stream = _core._HDF5ByteStream(
                source_payload["shards"][source_names[source_order]]
            )
            parquet = _pyarrow_parquet.ParquetFile(stream)
            selected = np.asarray(selected_local_rows[cursor:end], dtype=np.int64)
            selected_cursor = 0
            source_start = 0
            rows: List[Dict[str, Any]] = []
            for batch in parquet.iter_batches(batch_size=32_768):
                source_end = source_start + len(batch)
                right = int(np.searchsorted(selected, source_end, side="left"))
                if right > selected_cursor:
                    local = selected[selected_cursor:right] - source_start
                    rows.extend(
                        batch.take(_pyarrow.array(local, type=_pyarrow.int64())).to_pylist()
                    )
                    selected_cursor = right
                source_start = source_end
                if selected_cursor == len(selected):
                    break
            if len(rows) != end - cursor:
                raise ValueError(
                    f"Packed OMat24 shard {source_order} produced {len(rows)} rows, "
                    f"expected {end - cursor}"
                )
            shard_counts = np.asarray(
                [len(row["atomic_numbers"]) for row in rows], dtype=np.int64
            )
            if not np.array_equal(
                shard_counts, selected_atom_counts[cursor:end].astype(np.int64)
            ):
                raise ValueError("Packed OMat24 atom counts changed during decoding")
            shard_atom_count = int(np.sum(shard_counts, dtype=np.int64))
            shard_atom_end = atom_cursor + shard_atom_count
            numbers_buffer = np.empty((shard_atom_count,), dtype=np.int16)
            positions_buffer = np.empty((shard_atom_count, 3), dtype=np.float64)
            forces_buffer = np.empty((shard_atom_count, 3), dtype=np.float64)
            cell_buffer = np.empty((len(rows), 3, 3), dtype=np.float64)
            pbc_buffer = np.empty((len(rows), 3), dtype=np.bool_)
            energy_buffer = np.empty((len(rows),), dtype=np.float64)
            stress_buffer = np.empty((len(rows), 3, 3), dtype=np.float64)
            normalized_buffer = np.empty((len(rows),), dtype=np.bool_)
            configuration_ids: List[str] = []
            material_ids: List[str] = []
            local_atom_cursor = 0
            for offset, row in enumerate(rows):
                numbers = np.asarray(row["atomic_numbers"], dtype=np.int16)
                count = int(len(numbers))
                local_atom_end = local_atom_cursor + count
                numbers_buffer[local_atom_cursor:local_atom_end] = numbers
                positions_buffer[local_atom_cursor:local_atom_end] = np.asarray(
                    row["positions"], dtype=np.float64
                ).reshape(count, 3)
                forces_buffer[local_atom_cursor:local_atom_end] = np.asarray(
                    row["atomic_forces"], dtype=np.float64
                ).reshape(count, 3)
                cell_buffer[offset] = np.asarray(row["cell"], dtype=np.float64).reshape(3, 3)
                pbc_buffer[offset] = np.asarray(row["pbc"], dtype=np.bool_).reshape(3)
                energy_buffer[offset] = float(row["energy"])
                stress = np.asarray(row["cauchy_stress"], dtype=np.float64).reshape(3, 3)
                stress_buffer[offset] = 0.5 * (stress + stress.T)
                normalized_buffer[offset] = bool(
                    row.get("cauchy_stress_volume_normalized", False)
                )
                configuration_id = str(row.get("configuration_id", "unknown"))
                configuration_ids.append(configuration_id)
                material_ids.append(_core._omat24_material_id(row))
                local_atom_cursor = local_atom_end
            packed["atomic_numbers"][atom_cursor:shard_atom_end] = numbers_buffer
            packed["positions"][atom_cursor:shard_atom_end] = positions_buffer
            packed["forces"][atom_cursor:shard_atom_end] = forces_buffer
            packed["cell"][cursor:end] = cell_buffer
            packed["pbc"][cursor:end] = pbc_buffer
            packed["energy"][cursor:end] = energy_buffer
            packed["stress"][cursor:end] = stress_buffer
            packed["stress_volume_normalized"][cursor:end] = normalized_buffer
            packed["configuration_id"][cursor:end] = configuration_ids
            packed["material_id"][cursor:end] = material_ids
            atom_cursor = shard_atom_end
            cursor = end
            if cursor == total or source_order % 25 == 0:
                print(
                    f"[{_now()}] Packed OMat24 {cursor}/{total} structures, "
                    f"{atom_cursor}/{total_atoms} atoms",
                    flush=True,
                )
        if atom_cursor != total_atoms:
            raise RuntimeError("Packed OMat24 atom pointer is inconsistent")
        target.flush()


def pack_omat24_selection_in_composite(
    path: str,
    *,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a self-contained lossless packed copy of a Plus/Max dataset.

    The source composite remains untouched.  OMat24 geometry, forces, labels,
    and identifiers are copied as float64/source logical values; only the
    storage layout changes from embedded Parquet rows to packed HDF5 arrays.
    """
    if not HAS_H5PY or not HAS_PYARROW:
        raise RuntimeError("Packed OMat24 conversion requires h5py and pyarrow")
    source = Path(path).expanduser().resolve()
    output = Path(output_path or source.with_name(
        f"{source.stem}_packed{source.suffix}"
    )).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if output == source:
        raise ValueError("Packed conversion requires a distinct output path")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    with h5py.File(source, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != COMPOSITE_HDF5_SCHEMA_VERSION:
            raise ValueError("Input is not an E3MU composite HDF5 dataset")
        omat = handle["sources/omat24"]
        storage = str(omat.attrs.get("storage", ""))
        if storage == OMAT24_PACKED_SCHEMA_VERSION or "packed" in omat:
            result = inspect_composite_dataset(str(source), verify_sources=False)
            result.update({
                "output": str(source), "already_packed": True,
                "storage": OMAT24_PACKED_SCHEMA_VERSION,
            })
            return result
        if not bool(omat.attrs.get("embedded", False)) or not bool(
            omat.attrs.get("materialized", False)
        ):
            raise ValueError(
                "Packed conversion requires an embedded materialized OMat24 payload"
            )
        payload = omat.get("embedded_parquet")
        if payload is None or "shards" not in payload:
            raise ValueError("Embedded materialized Parquet payload is missing")
        selection = handle["selection"]
        selected_sources = np.asarray(selection["source_order"][:], dtype=np.uint16)
        selected_rows = np.asarray(selection["row_index"][:], dtype=np.uint32)
        selected_atoms = np.asarray(selection["atom_count"][:], dtype=np.uint16)
        if not (
            selected_sources.size == selected_rows.size == selected_atoms.size
        ):
            raise ValueError("Composite selector arrays have inconsistent lengths")
        if selected_sources.size and np.any(np.diff(selected_sources.astype(np.int64)) < 0):
            raise ValueError(
                "Composite selector source_order must be grouped before packing"
            )
        if selected_sources.size:
            boundaries = np.flatnonzero(
                np.diff(selected_sources.astype(np.int64)) != 0
            ) + 1
            for start, end in zip(
                np.concatenate(([0], boundaries)),
                np.concatenate((boundaries, [selected_sources.size])),
            ):
                if np.any(np.diff(selected_rows[int(start):int(end)].astype(np.int64)) <= 0):
                    raise ValueError(
                        "Composite selector row_index must be strictly increasing "
                        "within each OMat24 shard"
                    )
        source_names = [str(value) for value in payload["shard_names"].asstr()[:]]
        if len(source_names) != len(omat["file_paths"]):
            raise ValueError("Embedded shard names do not match OMat24 file paths")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.building-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        _write_packed_omat24_selection(
            source,
            temporary,
            selected_local_rows=selected_rows,
            selected_sources=selected_sources,
            selected_atom_counts=selected_atoms,
        )
        validation = inspect_composite_dataset(str(temporary), verify_sources=False)
        if not bool(validation.get("valid", False)):
            raise ValueError(
                "Packed composite validation failed: "
                + json.dumps(validation.get("embedded_errors", []))
            )
        if output.exists():
            output.unlink()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    result = inspect_composite_dataset(str(output), verify_sources=False)
    result.update({
        "output": str(output),
        "source": str(source),
        "storage": OMAT24_PACKED_SCHEMA_VERSION,
        "packed": True,
        "self_contained": True,
        "sha256": sha256_file(str(output)),
        "bytes": int(output.stat().st_size),
        "precision_policy": "Source float64 geometry and labels retained without quantization",
    })
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def build_neo_composite_half(
    parent_composite: str,
    standard_hdf5: str,
    output_path: str,
    *,
    seed: int = 20260723,
    omat_denominator: int = 180,
    max_bytes: int = 800_000_000,
    overwrite: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build local Plus Half from complete Standard plus 1/N materialized OMat24."""
    if not HAS_H5PY or not HAS_PYARROW:
        raise RuntimeError("Portable Plus Half construction requires h5py and pyarrow")
    parent = Path(parent_composite).expanduser().resolve()
    standard = Path(standard_hdf5).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    denominator = int(omat_denominator)
    if parent == output or standard == output:
        raise ValueError("The Plus Half output must differ from its sources")
    if int(max_bytes) <= 0:
        raise ValueError("Plus Half max_bytes must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if not parent.is_file() or not standard.is_file():
        raise FileNotFoundError(f"Plus Half sources are unavailable: {parent}, {standard}")

    with h5py.File(parent, "r") as source, h5py.File(standard, "r") as standard_source:
        if str(source.attrs.get("schema_version", "")) != COMPOSITE_HDF5_SCHEMA_VERSION:
            raise ValueError("OMat24 parent is not an E3MU composite HDF5 dataset")
        omat_source = source["sources/omat24"]
        if not bool(omat_source.attrs.get("embedded", False)):
            raise ValueError("OMat24 parent is not self-contained")
        omat_payload = omat_source.get("embedded_parquet")
        if omat_payload is None or "shards" not in omat_payload:
            raise ValueError("OMat24 parent has no embedded Parquet payload")
        if str(standard_source.attrs.get("schema_version", "")) != HDF5_SCHEMA_VERSION:
            raise ValueError("Standard source is not canonical Neo HDF5")

        source_orders = np.asarray(source["selection/source_order"][:], dtype=np.uint16)
        local_rows = np.asarray(source["selection/row_index"][:], dtype=np.uint32)
        source_rows = np.asarray(
            source["selection/source_row_index"][:]
            if "source_row_index" in source["selection"]
            else local_rows,
            dtype=np.uint32,
        )
        split_codes = np.asarray(source["selection/split_code"][:], dtype=np.uint8)
        atom_counts = np.asarray(source["selection/atom_count"][:], dtype=np.uint16)
        parent_metadata = json.loads(str(source.attrs.get("metadata_json", "{}")))
        source_count = int(len(omat_source["file_paths"]))
        if not (
            source_orders.size == local_rows.size == source_rows.size
            == split_codes.size == atom_counts.size
        ):
            raise ValueError("OMat24 parent selector arrays disagree")
        if np.any(source_orders[1:] < source_orders[:-1]):
            raise ValueError("OMat24 parent selectors are not source ordered")

        stratum = np.zeros((source_count, 3), dtype=np.int64)
        np.add.at(stratum, (source_orders.astype(np.int64), split_codes.astype(np.int64)), 1)
        quotas = _allocate_composite_fraction_quotas(
            stratum, denominator=denominator, seed=int(seed)
        )
        selected_parts: List[np.ndarray] = []
        for source_order in range(source_count):
            source_positions = np.flatnonzero(source_orders == source_order)
            for split_code in range(3):
                candidates = source_positions[split_codes[source_positions] == split_code]
                chosen_local = _stratified_interval_sample(
                    int(candidates.size), int(quotas[source_order, split_code]),
                    seed=int(seed), source_order=source_order, split_code=split_code,
                )
                if chosen_local.size:
                    selected_parts.append(candidates[chosen_local])
        omat_selected = np.sort(np.concatenate(selected_parts)).astype(np.int64)
        expected_omat = int(round(source_orders.size / float(denominator)))
        if int(omat_selected.size) != expected_omat:
            raise RuntimeError(
                f"Plus Half selected {omat_selected.size} OMat24 rows; expected {expected_omat}"
            )
        selected_sources = source_orders[omat_selected]
        selected_local_rows = local_rows[omat_selected]
        selected_source_rows = source_rows[omat_selected]
        selected_splits = split_codes[omat_selected]
        selected_atom_counts = atom_counts[omat_selected]
        selected_atoms = int(np.sum(selected_atom_counts, dtype=np.int64))

        standard_summary = hdf5_dataset_summary(str(standard))
        standard_validation = validate_neo_hdf5(str(standard))
        if not bool(standard_validation.get("valid")):
            raise ValueError(
                "Standard validation failed: "
                + json.dumps(standard_validation.get("errors", []))
            )
        standard_metadata = json.loads(
            str(standard_source.attrs.get("metadata_json", "{}"))
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        linked = output.with_name(output.name + ".linked.building")
        materialized = output.with_name(output.name + ".materialized.building")
        try:
            if not linked.exists():
                linked_target = h5py.File(linked, "w")
            else:
                linked_target = None
            if linked_target is not None:
                target = linked_target
                target.attrs["schema_version"] = COMPOSITE_HDF5_SCHEMA_VERSION
                target.attrs["created_at"] = _now()
                target.attrs["units_json"] = json.dumps(HDF5_UNITS, sort_keys=True)
                target.attrs["tier"] = "plus-half-local"
                target.attrs["seed"] = int(seed)
                target.attrs["selection_policy"] = (
                    f"complete Neo Standard plus exact stratified 1/{denominator} "
                    "of deduplicated OMat24 Max, allocated by source shard and split"
                )
                target.attrs["materialization_input_signature"] = (
                    "neo-plus-half-resume-20260723-v2"
                )
                target.attrs["large_storage"] = "embedded-canonical-root"

                sources = target.create_group("sources")
                target_omat = sources.create_group("omat24")
                for name, value in omat_source.attrs.items():
                    target_omat.attrs[name] = value
                for name in omat_source:
                    if name != "embedded_parquet":
                        source.copy(omat_source[name], target_omat, name=name)
                target_omat.attrs["embedded"] = True
                target_omat.attrs["materialized"] = False
                target_omat.attrs["storage"] = OMAT24_BYTE_SHARD_SCHEMA_VERSION
                if "root" in target_omat.attrs:
                    del target_omat.attrs["root"]
                linked_payload = target_omat.create_group("embedded_parquet")
                for name, value in omat_payload.attrs.items():
                    linked_payload.attrs[name] = value
                linked_payload.attrs["schema"] = OMAT24_BYTE_SHARD_SCHEMA_VERSION
                source.copy(omat_payload["shard_names"], linked_payload, name="shard_names")
                linked_shards = linked_payload.create_group("shards")
                shard_names = [str(value) for value in omat_payload["shard_names"].asstr()[:]]
                for shard_name in shard_names:
                    linked_shards[shard_name] = h5py.ExternalLink(
                        str(parent),
                        f"/sources/omat24/embedded_parquet/shards/{shard_name}",
                    )

                standard_group = sources.create_group("neo_large")
                standard_group.attrs["embedded"] = True
                standard_group.attrs["original_path"] = standard.name
                standard_group.attrs["source_sha256"] = str(standard_summary["sha256"])
                standard_group.attrs["source_bytes"] = int(standard.stat().st_size)
                standard_group.attrs["structures"] = int(standard_summary["structures"])
                standard_group.attrs["atoms"] = int(standard_summary["atoms"])
                standard_group.attrs["payload_groups_json"] = json.dumps(
                    HDF5_CANONICAL_ROOT_GROUPS
                )

                selector = target.create_group("selection")
                selector_chunk = max(1, min(250_000, int(omat_selected.size)))
                for name, values, dtype in (
                    ("source_order", selected_sources, np.uint16),
                    ("row_index", selected_local_rows, np.uint32),
                    ("source_row_index", selected_source_rows, np.uint32),
                    ("split_code", selected_splits, np.uint8),
                    ("atom_count", selected_atom_counts, np.uint16),
                ):
                    selector.create_dataset(
                        name, data=np.asarray(values, dtype=dtype),
                        chunks=(selector_chunk,), compression="gzip",
                    )
                for name in HDF5_CANONICAL_ROOT_GROUPS:
                    standard_source.copy(name, target, name=name)
                source.copy(source["atomic_reference"], target, name="atomic_reference")
                target["atomic_reference"].attrs["subset_policy"] = (
                    f"Inherited OMat24 Max normal equations; runtime fit uses the 1/{denominator} sample"
                )

                per_file = np.bincount(selected_sources.astype(np.int64), minlength=source_count)
                file_paths = [str(value) for value in omat_source["file_paths"].asstr()[:]]
                omat_sources: Dict[str, int] = collections.Counter()
                for source_order, count in enumerate(per_file):
                    if int(count):
                        omat_sources[f"OMat24/{Path(file_paths[source_order]).parts[0]}"] += int(count)
                derived_corpora: List[Dict[str, Any]] = []
                for raw_corpus in parent_metadata.get("omat24", {}).get("corpora", []):
                    corpus = dict(raw_corpus)
                    count = int(omat_sources.get(f"OMat24/{corpus.get('name', 'unknown')}", 0))
                    corpus.update({
                        "selected_before_dedup": count,
                        "selected_unique": count,
                        "duplicate_configuration_ids": 0,
                        "derived_from_max_selector": True,
                    })
                    derived_corpora.append(corpus)
                omat_metadata = dict(parent_metadata.get("omat24", {}))
                omat_metadata.update({
                    "selected_unique_structures": int(omat_selected.size),
                    "atoms": selected_atoms,
                    "parent_selected_structures": int(source_orders.size),
                    "sampling_denominator": denominator,
                    "selection_fraction_of_parent": float(omat_selected.size) / float(source_orders.size),
                    "selection_policy": (
                        "exact proportional source-shard/split quotas with deterministic "
                        "interval coverage inside every non-empty stratum"
                    ),
                    "corpora": derived_corpora,
                })
                metadata = {
                    "dataset": "Neo Plus Half local self-contained curriculum",
                    "corpus_role": "local-foundation-response-curriculum",
                    "seed": int(seed),
                    "omat24": omat_metadata,
                    "neo_large": {
                        "structures": int(standard_summary["structures"]),
                        "atoms": int(standard_summary["atoms"]),
                        "elements": list(standard_summary["elements"]),
                        "periodic_structures": int(standard_summary["periodic_structures"]),
                        "labels": dict(standard_summary["labels"]),
                        "sources": dict(standard_summary["sources"]),
                        "splits": dict(standard_summary["splits"]),
                        "source_tier": "standard",
                        "source_metadata": standard_metadata,
                    },
                    "curriculum": {
                        "base": "materialized OMat24 L1 foundation sample",
                        "response": "complete Neo Standard response and spin records",
                        "joint": "role-balanced OMat24 plus complete Standard coupling corpus",
                    },
                    "public_release": False,
                    "precision_policy": (
                        "Source Arrow logical types and canonical Standard float64 arrays retained; "
                        "no numerical quantization"
                    ),
                }
                target.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
                target.attrs["structures"] = int(omat_selected.size + standard_summary["structures"])
                target.attrs["omat24_structures"] = int(omat_selected.size)
                target.attrs["large_structures"] = int(standard_summary["structures"])
                target.attrs["atoms"] = int(selected_atoms + standard_summary["atoms"])
                target.attrs["elements_json"] = json.dumps(sorted(
                    set(int(value) for value in omat_metadata.get("elements", []))
                    | set(int(value) for value in standard_summary["elements"])
                ))
                target.close()

            shutil.copy2(linked, materialized)
            _write_packed_omat24_selection(
                parent, materialized,
                selected_local_rows=selected_local_rows,
                selected_sources=selected_sources,
                selected_atom_counts=selected_atom_counts,
            )
            actual_bytes = int(materialized.stat().st_size)
            if actual_bytes > int(max_bytes):
                raise ValueError(
                    f"Plus Half is {actual_bytes} bytes, above the {int(max_bytes)}-byte limit"
                )
            validation = inspect_composite_dataset(str(materialized), verify_sources=False)
            if not bool(validation.get("valid", False)):
                raise ValueError(
                    "Portable Plus Half validation failed: "
                    + json.dumps(validation.get("embedded_errors", []))
                )
            materialized.replace(output)
        finally:
            if output.exists() and not materialized.exists():
                linked.unlink(missing_ok=True)

    result = inspect_composite_dataset(str(output), verify_sources=False)
    result.update({
        "output": str(output),
        "sha256": sha256_file(str(output)),
        "tier": "plus-half-local",
        "parent": str(parent),
        "standard": str(standard),
        "max_bytes": int(max_bytes),
        "omat24_parent_structures": int(source_orders.size),
        "omat24_sampling_denominator": denominator,
        "omat24_selection_fraction": float(omat_selected.size) / float(source_orders.size),
        "standard_structures": int(standard_summary["structures"]),
        "standard_sha256": str(standard_summary["sha256"]),
        "bytes": int(output.stat().st_size),
        "self_contained": True,
        "public_release": False,
    })
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def embed_omat24_parquet_in_composite(
    path: str,
    *,
    output_path: Optional[str] = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    overwrite: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Embed every declared OMat24 Parquet shard into one composite HDF5."""
    if not HAS_H5PY:
        raise RuntimeError("Embedding OMat24 shards requires h5py")
    composite = Path(path).expanduser().resolve()
    output = Path(output_path or path).expanduser().resolve()
    if not composite.is_file():
        raise FileNotFoundError(str(composite))
    if output != composite and output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    with h5py.File(composite, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != COMPOSITE_HDF5_SCHEMA_VERSION:
            raise ValueError("Target is not an E3MU composite HDF5 dataset")
        omat = handle["sources/omat24"]
        if bool(omat.attrs.get("embedded", False)):
            result = inspect_composite_dataset(str(composite), verify_sources=False)
            result.update({"output": str(composite), "embedded_omat24": True})
            return result
        root = _resolve_composite_source(composite, str(omat.attrs["root"]))
        file_paths = [str(value) for value in omat["file_paths"].asstr()[:]]
        file_hashes = [str(value) for value in omat["file_sha256"].asstr()[:]]
    if len(file_paths) != len(file_hashes):
        raise ValueError("OMat24 file path and checksum arrays disagree")
    sources = [root / relative for relative in file_paths]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("OMat24 shards are missing: " + ", ".join(missing[:5]))
    chunk = max(1, int(chunk_bytes))
    temporary = output.with_name(output.name + ".omat-embedded.building")
    if temporary.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace temporary file: {temporary}")
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(composite, temporary)
        with h5py.File(temporary, "r+") as handle:
            omat = handle["sources/omat24"]
            payload = omat.create_group("embedded_parquet")
            payload.attrs["schema"] = "hdf5-byte-shards-v1"
            payload.attrs["chunk_bytes"] = chunk
            payload.attrs["source_count"] = len(sources)
            shard_names = [f"shard_{index:06d}" for index in range(len(sources))]
            string_dtype = h5py.string_dtype("utf-8")
            payload.create_dataset(
                "shard_names", data=np.asarray(shard_names, dtype=object), dtype=string_dtype
            )
            shards = payload.create_group("shards")
            for index, (source, expected) in enumerate(zip(sources, file_hashes)):
                digest = hashlib.sha256()
                with source.open("rb") as source_handle:
                    while True:
                        block = source_handle.read(chunk)
                        if not block:
                            break
                        digest.update(block)
                actual = digest.hexdigest()
                if actual != expected:
                    raise ValueError(
                        f"OMat24 checksum mismatch for {source}: "
                        f"expected {expected}, got {actual}"
                    )
                size = int(source.stat().st_size)
                dataset = shards.create_dataset(
                    shard_names[index],
                    shape=(size,),
                    dtype=np.uint8,
                    chunks=(min(chunk, max(1, size)),),
                )
                dataset.attrs["relative_path"] = file_paths[index]
                dataset.attrs["sha256"] = expected
                dataset.attrs["bytes"] = size
                cursor = 0
                with source.open("rb") as source_handle:
                    while cursor < size:
                        block = source_handle.read(min(chunk, size - cursor))
                        if not block:
                            raise EOFError(f"Truncated OMat24 shard while embedding: {source}")
                        dataset[cursor:cursor + len(block)] = np.frombuffer(
                            block, dtype=np.uint8
                        )
                        cursor += len(block)
            omat.attrs["embedded"] = True
            omat.attrs["storage"] = "hdf5-byte-shards-v1"
            omat.attrs["embedded_source_count"] = len(sources)
            # Keep relative names in file_paths for provenance, but remove the
            # machine-specific root so runtime cannot silently fall back to it.
            if "root" in omat.attrs:
                del omat.attrs["root"]
            omat.attrs["source_root_provenance"] = "colabfit/OMat24 manifests"
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    result = inspect_composite_dataset(str(output), verify_sources=False)
    result.update({
        "output": str(output),
        "embedded_omat24": True,
        "embedded_shards": len(file_paths),
        "bytes": int(output.stat().st_size),
        "sha256": sha256_file(str(output)),
    })
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _copy_parquet_bytes_into_hdf5(
    source: Any,
    target_group: Any,
    name: str,
    *,
    chunk_bytes: int,
) -> Tuple[int, str]:
    """Copy a file or HDF5 uint8 dataset while hashing the exact bytes."""
    is_hdf5 = HAS_H5PY and isinstance(source, h5py.Dataset)
    size = int(source.shape[0]) if is_hdf5 else int(Path(source).stat().st_size)
    chunk = max(1, int(chunk_bytes))
    target = target_group.create_dataset(
        name,
        shape=(size,),
        dtype=np.uint8,
        chunks=(min(chunk, max(1, size)),),
    )
    digest = hashlib.sha256()
    if is_hdf5:
        for start in range(0, size, chunk):
            values = np.asarray(source[start:start + chunk], dtype=np.uint8)
            target[start:start + len(values)] = values
            digest.update(values.tobytes())
    else:
        cursor = 0
        with Path(source).open("rb") as handle:
            while cursor < size:
                block = handle.read(min(chunk, size - cursor))
                if not block:
                    raise EOFError(f"Truncated Parquet file while embedding: {source}")
                target[cursor:cursor + len(block)] = np.frombuffer(block, dtype=np.uint8)
                digest.update(block)
                cursor += len(block)
    return size, digest.hexdigest()


def _write_selected_parquet_rows(
    parquet: Any,
    selected_rows: np.ndarray,
    output: Path,
    *,
    batch_rows: int,
    row_group_rows: int,
    compression_level: int = 3,
) -> None:
    """Write selected source rows without converting Arrow logical values."""
    selected = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if selected.size and (
        int(selected[0]) < 0
        or np.any(np.diff(selected) <= 0)
        or int(selected[-1]) >= int(parquet.metadata.num_rows)
    ):
        raise ValueError("Selected Parquet row indices must be unique and increasing")
    output.unlink(missing_ok=True)
    writer = _pyarrow_parquet.ParquetWriter(
        str(output),
        parquet.schema_arrow,
        compression="zstd",
        compression_level=int(compression_level),
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    source_start = 0
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=max(1, int(batch_rows))):
            source_end = source_start + len(batch)
            left = int(np.searchsorted(selected, source_start, side="left"))
            right = int(np.searchsorted(selected, source_end, side="left"))
            if right > left:
                local = selected[left:right] - source_start
                filtered = batch.take(
                    _pyarrow.array(local, type=_pyarrow.int64())
                )
                writer.write_batch(
                    filtered, row_group_size=max(1, int(row_group_rows))
                )
                written += len(filtered)
            source_start = source_end
    finally:
        writer.close()
    if written != int(selected.size):
        raise ValueError(
            f"Materialized {written} rows, expected {int(selected.size)}"
        )
    check = _pyarrow_parquet.ParquetFile(str(output))
    if int(check.metadata.num_rows) != int(selected.size):
        raise ValueError("Materialized Parquet row count is inconsistent")
    if not check.schema_arrow.equals(parquet.schema_arrow, check_metadata=True):
        raise ValueError("Materialized Parquet schema differs from its source")


def _composite_source_counts(
    source_order: Any,
    source_count: int,
    *,
    chunk_rows: int = 4_000_000,
) -> np.ndarray:
    """Count monotonic source selectors without loading the full index."""
    counts = np.zeros((int(source_count),), dtype=np.int64)
    previous = -1
    total = int(source_order.shape[0])
    for start in range(0, total, max(1, int(chunk_rows))):
        values = np.asarray(
            source_order[start:start + max(1, int(chunk_rows))], dtype=np.int64
        )
        if values.size == 0:
            continue
        if (
            int(values[0]) < previous
            or np.any(np.diff(values) < 0)
            or np.any(values < 0)
            or np.any(values >= int(source_count))
        ):
            raise ValueError(
                "Composite source_order must be grouped and monotonic for materialization"
            )
        counts += np.bincount(values, minlength=int(source_count))
        previous = int(values[-1])
    return counts


def materialize_omat24_selection_in_composite(
    path: str,
    *,
    output_path: Optional[str] = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    batch_rows: int = 32_768,
    row_group_rows: int = 4_096,
    compression_level: int = 3,
    resume: bool = True,
    restart: bool = False,
    overwrite: bool = False,
    compute_sha256: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Embed only selected OMat24 rows as lossless, random-access Parquet shards."""
    if not HAS_H5PY or not HAS_PYARROW:
        raise RuntimeError("OMat24 materialization requires h5py and pyarrow")
    if not 1 <= int(compression_level) <= 22:
        raise ValueError("Zstandard compression_level must be in [1, 22]")
    composite = Path(path).expanduser().resolve()
    output = Path(output_path or path).expanduser().resolve()
    if not composite.is_file():
        raise FileNotFoundError(str(composite))
    if output != composite and output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    source_stat = composite.stat()
    with h5py.File(composite, "r") as source:
        if str(source.attrs.get("schema_version", "")) != COMPOSITE_HDF5_SCHEMA_VERSION:
            raise ValueError("Target is not an E3MU composite HDF5 dataset")
        source_omat = source["sources/omat24"]
        source_storage = str(source_omat.attrs.get("storage", ""))
        already_materialized = bool(source_omat.attrs.get("materialized", False)) or (
            source_storage == OMAT24_MATERIALIZED_SCHEMA_VERSION
        )
    if already_materialized:
        if output != composite:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(composite, output)
        result = inspect_composite_dataset(str(output), verify_sources=False)
        result.update({"output": str(output), "already_materialized": True})
        return result

    temporary = output.with_name(output.name + ".omat-materialized.building")
    shard_temporary = output.with_name(output.name + ".selected-shard.parquet.building")
    if restart:
        temporary.unlink(missing_ok=True)
        shard_temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(composite, "r") as source:
        source_omat = source["sources/omat24"]
        file_paths = [str(value) for value in source_omat["file_paths"].asstr()[:]]
        file_hashes = [str(value) for value in source_omat["file_sha256"].asstr()[:]]
        if len(file_paths) != len(file_hashes):
            raise ValueError("OMat24 file path and checksum arrays disagree")
        source_count = len(file_paths)
        source_is_embedded = bool(source_omat.attrs.get("embedded", False))
        source_payload = source_omat.get("embedded_parquet")
        if source_is_embedded:
            if source_payload is None or "shard_names" not in source_payload:
                raise ValueError("Embedded OMat24 payload is incomplete")
            source_shard_names = [
                str(value) for value in source_payload["shard_names"].asstr()[:]
            ]
            if len(source_shard_names) != source_count:
                raise ValueError("Embedded shard names do not match file_paths")
            source_root = None
        else:
            source_root = _resolve_composite_source(
                composite, str(source_omat.attrs["root"])
            )
            source_shard_names = [f"shard_{index:06d}" for index in range(source_count)]
        if "file_rows" in source_omat:
            source_rows = np.asarray(source_omat["file_rows"][:], dtype=np.int64)
        else:
            source_rows = np.full((source_count,), -1, dtype=np.int64)
        source_counts = _composite_source_counts(
            source["selection/source_order"], source_count
        )
        source_offsets = np.concatenate((
            np.zeros((1,), dtype=np.int64),
            np.cumsum(source_counts, dtype=np.int64),
        ))

        if temporary.exists():
            if not resume:
                raise FileExistsError(
                    f"Resumable materialization exists: {temporary}; use restart=True"
                )
            with h5py.File(temporary, "r") as probe:
                source_signature = str(
                    source.attrs.get("materialization_input_signature", "")
                )
                stored_signature = str(
                    probe.attrs.get("materialization_input_signature", "")
                )
                stable_match = bool(source_signature) and (
                    stored_signature == source_signature
                )
                compatible = (
                    str(probe.attrs.get("materialization_state", ""))
                    in {"building", "complete"}
                    and (
                        stable_match
                        or (
                            int(probe.attrs.get("materialization_source_bytes", -1))
                            == int(source_stat.st_size)
                            and int(probe.attrs.get("materialization_source_mtime_ns", -1))
                            == int(source_stat.st_mtime_ns)
                        )
                    )
                    and int(probe.attrs.get("omat24_structures", -1))
                    == int(source_counts.sum())
                )
            if not compatible:
                raise ValueError(
                    f"Existing resumable file does not match its source: {temporary}"
                )
        else:
            with h5py.File(temporary, "w") as target:
                for name, value in source.attrs.items():
                    target.attrs[name] = value
                target.attrs["created_at"] = _now()
                target.attrs["materialization_state"] = "building"
                target.attrs["materialization_source_bytes"] = int(source_stat.st_size)
                target.attrs["materialization_source_mtime_ns"] = int(
                    source_stat.st_mtime_ns
                )
                target.attrs["materialization_source_name"] = composite.name
                target.attrs["materialization_input_signature"] = str(
                    source.attrs.get("materialization_input_signature", "")
                )

                for name in (*HDF5_CANONICAL_ROOT_GROUPS, "atomic_reference"):
                    source.copy(source[name], target, name=name)

                target_sources = target.create_group("sources")
                source.copy(source["sources/neo_large"], target_sources, name="neo_large")
                target_omat = target_sources.create_group("omat24")
                for name, value in source_omat.attrs.items():
                    if name not in {
                        "root", "embedded", "storage", "materialized",
                        "embedded_source_count", "materialized_source_count",
                    }:
                        target_omat.attrs[name] = value
                for name in source_omat:
                    if name != "embedded_parquet":
                        source.copy(source_omat[name], target_omat, name=name)
                target_omat.attrs["embedded"] = True
                target_omat.attrs["materialized"] = True
                target_omat.attrs["storage"] = OMAT24_MATERIALIZED_SCHEMA_VERSION
                target_omat.attrs["source_root_provenance"] = (
                    "colabfit/OMat24 manifests; selected rows embedded below"
                )

                target_selection = target.create_group("selection")
                for name in ("source_order", "split_code", "atom_count"):
                    source.copy(source[f"selection/{name}"], target_selection, name=name)
                provenance_rows = (
                    source["selection/source_row_index"]
                    if "source_row_index" in source["selection"]
                    else source["selection/row_index"]
                )
                source.copy(provenance_rows, target_selection, name="source_row_index")
                source_row_dataset = source["selection/row_index"]
                target_selection.create_dataset(
                    "row_index",
                    shape=source_row_dataset.shape,
                    dtype=np.uint32,
                    chunks=source_row_dataset.chunks,
                    compression=source_row_dataset.compression,
                    compression_opts=source_row_dataset.compression_opts,
                )

                payload = target_omat.create_group("embedded_parquet")
                payload.attrs["schema"] = OMAT24_MATERIALIZED_SCHEMA_VERSION
                payload.attrs["source_count"] = source_count
                payload.attrs["selected_rows"] = int(source_counts.sum())
                payload.attrs["row_group_rows"] = int(row_group_rows)
                payload.attrs["compression_level"] = int(compression_level)
                payload.attrs["chunk_bytes"] = int(chunk_bytes)
                payload.attrs["completed_source_count"] = 0
                payload.attrs["precision_policy"] = (
                    "Arrow logical values preserved; no floating-point dtype conversion"
                )
                string_dtype = h5py.string_dtype("utf-8")
                payload.create_dataset(
                    "shard_names",
                    data=np.asarray(source_shard_names, dtype=object),
                    dtype=string_dtype,
                )
                payload.create_dataset(
                    "materialized_sha256",
                    data=np.asarray([""] * source_count, dtype=object),
                    dtype=string_dtype,
                )
                payload.create_dataset(
                    "materialized_rows", shape=(source_count,), dtype=np.int64
                )
                payload.create_dataset(
                    "source_rows", data=np.asarray(source_rows, dtype=np.int64)
                )
                payload.create_dataset(
                    "materialized_bytes", shape=(source_count,), dtype=np.int64
                )
                payload.create_group("shards")
                target.flush()

    started = time.monotonic()
    try:
        with h5py.File(composite, "r") as source, h5py.File(temporary, "r+") as target:
            source_omat = source["sources/omat24"]
            source_payload = source_omat.get("embedded_parquet")
            payload = target["sources/omat24/embedded_parquet"]
            target_shards = payload["shards"]
            completed = 0
            for source_order in range(source_count):
                shard_name = source_shard_names[source_order]
                start = int(source_offsets[source_order])
                end = int(source_offsets[source_order + 1])
                selected = np.asarray(
                    source["selection/row_index"][start:end], dtype=np.int64
                )
                if selected.size and np.any(np.diff(selected) <= 0):
                    raise ValueError(
                        f"Source {source_order} selector rows are not strictly increasing"
                    )
                existing = target_shards.get(shard_name)
                if existing is not None and bool(existing.attrs.get("complete", False)):
                    if int(existing.attrs.get("selected_rows", -1)) != int(selected.size):
                        raise ValueError(
                            f"Resumable shard {source_order} has the wrong row count"
                        )
                    completed += 1
                    continue
                if existing is not None:
                    del target_shards[shard_name]

                if source_is_embedded:
                    byte_source = source_payload["shards"][shard_name]
                    parquet_source = _core._HDF5ByteStream(byte_source)
                else:
                    byte_source = source_root / file_paths[source_order]
                    parquet_source = str(byte_source)
                declared_rows = int(source_rows[source_order])
                actual_rows = int(
                    _pyarrow_parquet.ParquetFile(parquet_source).metadata.num_rows
                )
                if source_is_embedded:
                    parquet_source.seek(0)
                if declared_rows < 0:
                    declared_rows = actual_rows
                elif actual_rows != declared_rows:
                    # A duplicate removed by the Max selector can make its
                    # retained materialized shard one row shorter than the
                    # upstream file_rows provenance. The embedded shard is the
                    # authoritative source for further nested materialization.
                    if source_is_embedded and actual_rows < declared_rows:
                        declared_rows = actual_rows
                    else:
                        raise ValueError(
                            f"Source {source_order} row metadata is inconsistent"
                        )
                if selected.size and int(selected[-1]) >= declared_rows:
                    raise ValueError(
                        f"Source {source_order} selector exceeds {declared_rows} rows"
                    )
                identity = (
                    int(selected.size) == declared_rows
                    and (
                        declared_rows == 0
                        or (
                            int(selected[0]) == 0
                            and int(selected[-1]) == declared_rows - 1
                            and np.all(np.diff(selected) == 1)
                        )
                    )
                )
                if identity:
                    embedded_bytes, materialized_hash = _copy_parquet_bytes_into_hdf5(
                        byte_source,
                        target_shards,
                        shard_name,
                        chunk_bytes=chunk_bytes,
                    )
                    if materialized_hash != file_hashes[source_order]:
                        raise ValueError(
                            f"Source checksum mismatch for {file_paths[source_order]}"
                        )
                else:
                    if source_is_embedded:
                        parquet_source = _core._HDF5ByteStream(byte_source)
                    parquet = _pyarrow_parquet.ParquetFile(parquet_source)
                    if int(parquet.metadata.num_rows) != declared_rows:
                        raise ValueError(
                            f"Source {source_order} row metadata is inconsistent"
                        )
                    _write_selected_parquet_rows(
                        parquet,
                        selected,
                        shard_temporary,
                        batch_rows=batch_rows,
                        row_group_rows=row_group_rows,
                        compression_level=compression_level,
                    )
                    embedded_bytes, materialized_hash = _copy_parquet_bytes_into_hdf5(
                        shard_temporary,
                        target_shards,
                        shard_name,
                        chunk_bytes=chunk_bytes,
                    )
                    shard_temporary.unlink(missing_ok=True)

                embedded = target_shards[shard_name]
                embedded.attrs["relative_path"] = file_paths[source_order]
                embedded.attrs["source_sha256"] = file_hashes[source_order]
                embedded.attrs["materialized_sha256"] = materialized_hash
                embedded.attrs["source_rows"] = declared_rows
                embedded.attrs["selected_rows"] = int(selected.size)
                embedded.attrs["bytes"] = int(embedded_bytes)
                embedded.attrs["precision"] = (
                    "source Arrow schema and floating-point values preserved"
                )
                payload["materialized_sha256"][source_order] = materialized_hash
                payload["materialized_rows"][source_order] = int(selected.size)
                payload["source_rows"][source_order] = declared_rows
                payload["materialized_bytes"][source_order] = int(embedded_bytes)
                target["selection/row_index"][start:end] = np.arange(
                    int(selected.size), dtype=np.uint32
                )
                # Mark completion last so an interrupted shard is never skipped
                # before its selector mapping and audit metadata are durable.
                embedded.attrs["complete"] = True
                completed += 1
                payload.attrs["completed_source_count"] = completed
                target.flush()
                elapsed = max(1e-9, time.monotonic() - started)
                print(
                    f"[{_now()}] OMat24 materialization {source_order + 1}/{source_count}: "
                    f"selected={int(selected.size)}/{declared_rows} "
                    f"bytes={int(embedded_bytes)} completed={completed} "
                    f"elapsed={elapsed / 60.0:.1f} min",
                    flush=True,
                )

            if completed != source_count:
                raise ValueError(
                    f"Materialization completed {completed}/{source_count} shards"
                )
            metadata = json.loads(str(target.attrs.get("metadata_json", "{}")))
            omat_metadata = dict(metadata.get("omat24", {}))
            omat_metadata.update({
                "storage": OMAT24_MATERIALIZED_SCHEMA_VERSION,
                "embedded_source_shards": source_count,
                "materialized_selected_rows": int(source_counts.sum()),
                "source_rows": int(np.sum(payload["source_rows"][:], dtype=np.int64)),
                "materialized_bytes": int(
                    np.sum(payload["materialized_bytes"][:], dtype=np.int64)
                ),
                "row_index_policy": (
                    "selection/row_index is local to materialized shards; "
                    "selection/source_row_index preserves original OMat24 rows"
                ),
            })
            metadata["omat24"] = omat_metadata
            metadata["precision_policy"] = (
                "Selected Arrow values retain source logical types, including float64; "
                "no coordinate, energy, force, stress, or response-target quantization"
            )
            target.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            target.attrs["materialization_state"] = "complete"
            target.attrs["materialized_at"] = _now()
            target["sources/omat24"].attrs["materialized_source_count"] = source_count
            target.flush()
    except Exception:
        print(
            f"[{_now()}] Materialization interrupted; resumable state retained at {temporary}",
            flush=True,
        )
        raise
    finally:
        shard_temporary.unlink(missing_ok=True)

    validation = inspect_composite_dataset(str(temporary), verify_sources=False)
    if not bool(validation.get("valid", False)):
        raise ValueError(
            "Materialized composite validation failed: "
            + json.dumps(validation.get("embedded_errors", []))
        )
    temporary.replace(output)
    result = inspect_composite_dataset(str(output), verify_sources=False)
    result.update({
        "output": str(output),
        "already_materialized": False,
        "materialized_omat24": True,
        "source_bytes": int(source_stat.st_size),
        "bytes": int(output.stat().st_size),
        "source_shards": source_count,
        "selected_rows": int(source_counts.sum()),
        "sha256": sha256_file(str(output)) if compute_sha256 else None,
    })
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def build_neo_omat24_composite(
    omat_root: str,
    large_hdf5: str,
    output_path: str,
    *,
    tier: str,
    seed: int = 20260722,
    overwrite: bool = False,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create Plus/Max with embedded Large data and streamed OMat24 selectors.

    This first-stage composite references source shards. Run
    ``dataset-composite-materialize-omat`` afterward to produce the portable
    selected-row release artifact.
    """
    if not HAS_H5PY or not HAS_PYARROW:
        raise RuntimeError("OMat24 composite building requires h5py and pyarrow")
    selected_tier = str(tier).strip().lower()
    if selected_tier not in {"plus", "max"}:
        raise ValueError("tier must be 'plus' or 'max'")
    root = Path(omat_root).expanduser().resolve()
    large = Path(large_hdf5).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if output.exists() and overwrite:
        output.unlink()
    corpora = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("OMat24_")
        and (path / "ds.parquet").is_file()
        and any((path / "co").glob("*.parquet"))
    )
    if not corpora:
        raise FileNotFoundError(f"No complete OMat24 corpora found under {root}")
    parquet_files = [
        path for corpus in corpora for path in sorted((corpus / "co").glob("*.parquet"))
    ]
    incomplete_omat = [
        path for corpus in corpora for path in corpus.rglob("*.part")
    ]
    if incomplete_omat:
        raise ValueError(
            f"OMat24 corpus contains incomplete .part files: {incomplete_omat[:5]}"
        )

    try:
        import duckdb
    except Exception as exc:
        raise RuntimeError("OMat24 identity indexing requires duckdb>=1.2,<2") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    temporary.unlink(missing_ok=True)
    work_db = output.with_name(output.stem + ".identity.building.duckdb")
    resume_db = False
    resume_max_source_order = -1
    if selected_tier == "max" and work_db.is_file():
        try:
            probe = duckdb.connect(str(work_db), read_only=True)
            tables = {str(row[0]) for row in probe.execute("show tables").fetchall()}
            if "selected_rows" in tables:
                stored_max = probe.execute(
                    "select max(source_order) from selected_rows"
                ).fetchone()[0]
                stored_max = -1 if stored_max is None else int(stored_max)
                if 0 <= stored_max < len(parquet_files):
                    checks = probe.execute(
                        "select source_order, min(source_file) from selected_rows "
                        "where source_order in (0, ?) group by source_order "
                        "order by source_order",
                        [stored_max],
                    ).fetchall()
                    expected = {
                        0: str(parquet_files[0].resolve()),
                        stored_max: str(parquet_files[stored_max].resolve()),
                    }
                    resume_db = all(
                        int(order) in expected and str(source) == expected[int(order)]
                        for order, source in checks
                    ) and len(checks) == (1 if stored_max == 0 else 2)
                    if resume_db:
                        resume_max_source_order = stored_max
            probe.close()
        except Exception:
            resume_db = False
    if not resume_db:
        work_db.unlink(missing_ok=True)
    connection = duckdb.connect(str(work_db))
    connection.execute(f"PRAGMA threads={max(1, min(8, os.cpu_count() or 1))}")
    connection.execute("PRAGMA memory_limit='12GB'")
    if not resume_db:
        connection.execute(
            "CREATE TABLE selected_rows ("
            "source_order INTEGER, source_file VARCHAR, row_index BIGINT, "
            "configuration_id VARCHAR, material_id VARCHAR, split VARCHAR, nsites INTEGER, "
            "PRIMARY KEY(configuration_id))"
        )
    corpus_reports: List[Dict[str, Any]] = []
    source_order = 0
    duplicate_rows = 0
    selected_omat_rows = int(
        connection.execute("select count(*) from selected_rows").fetchone()[0]
    ) if resume_db else 0
    selected_omat_atoms = int(
        connection.execute("select coalesce(sum(nsites), 0) from selected_rows").fetchone()[0]
    ) if resume_db else 0
    omat_elements: set[int] = set()
    try:
        for corpus in corpora:
            ds_table = _pyarrow_parquet.read_table(corpus / "ds.parquet", columns=["elements"])
            for symbols in ds_table.column("elements").to_pylist():
                omat_elements.update(
                    int(ASE_ATOMIC_NUMBERS[str(symbol)]) for symbol in symbols
                )
            report = {
                "name": corpus.name,
                "rows": 0,
                "selected_before_dedup": 0,
                "selected_unique": 0,
                "duplicate_configuration_ids": 0,
                "parquet_files": [],
            }
            for parquet in sorted((corpus / "co").glob("*.parquet")):
                parquet_meta = _pyarrow_parquet.ParquetFile(parquet).metadata
                rows = int(parquet_meta.num_rows)
                report["rows"] += rows
                relative = parquet.relative_to(root).as_posix()
                report["parquet_files"].append({
                    "path": relative,
                    "bytes": int(parquet.stat().st_size),
                    "sha256": sha256_file(str(parquet)),
                    "rows": rows,
                })
                if resume_db and source_order <= resume_max_source_order:
                    kept, kept_atoms = connection.execute(
                        "select count(*), coalesce(sum(nsites), 0) from selected_rows "
                        "where source_order = ?",
                        [source_order],
                    ).fetchone()
                    report["selected_before_dedup"] += int(rows)
                    report["selected_unique"] += int(kept)
                    report["duplicate_configuration_ids"] += int(rows) - int(kept)
                    source_order += 1
                    continue
                selection_clause = "TRUE"
                if selected_tier == "plus":
                    selection_clause = (
                        "cast('0x'||substr(sha256('neo-plus-v1|'||material_id),1,2) "
                        "AS integer) < 64"
                    )
                query = f"""
                    WITH decoded AS (
                        SELECT file_row_number AS row_index, configuration_id,
                               coalesce(nullif(regexp_extract(names[1],
                                   'OMat24__(agm[0-9]+)_', 1), ''),
                                   configuration_id) AS material_id,
                               nsites
                        FROM read_parquet(?, file_row_number=true)
                    ), chosen AS (
                        SELECT *, CASE
                            WHEN cast('0x'||substr(sha256('neo-split-v1|'||material_id),1,2)
                                      AS integer) < 205 THEN 'train'
                            WHEN cast('0x'||substr(sha256('neo-split-v1|'||material_id),1,2)
                                      AS integer) < 230 THEN 'val'
                            ELSE 'test' END AS split
                        FROM decoded WHERE {selection_clause}
                    )
                    SELECT count(*), coalesce(sum(nsites), 0) FROM chosen
                """
                chosen_rows, chosen_atoms = connection.execute(
                    query, [str(parquet)]
                ).fetchone()
                report["selected_before_dedup"] += int(chosen_rows)
                before = int(
                    connection.execute("SELECT count(*) FROM selected_rows").fetchone()[0]
                )
                insert_query = f"""
                    INSERT OR IGNORE INTO selected_rows
                    WITH decoded AS (
                        SELECT file_row_number AS row_index, configuration_id,
                               coalesce(nullif(regexp_extract(names[1],
                                   'OMat24__(agm[0-9]+)_', 1), ''),
                                   configuration_id) AS material_id,
                               nsites
                        FROM read_parquet(?, file_row_number=true)
                    ), chosen AS (
                        SELECT *, CASE
                            WHEN cast('0x'||substr(sha256('neo-split-v1|'||material_id),1,2)
                                      AS integer) < 205 THEN 'train'
                            WHEN cast('0x'||substr(sha256('neo-split-v1|'||material_id),1,2)
                                      AS integer) < 230 THEN 'val'
                            ELSE 'test' END AS split
                        FROM decoded WHERE {selection_clause}
                    )
                    SELECT ?, ?, row_index, configuration_id, material_id, split, nsites
                    FROM chosen
                """
                connection.execute(insert_query, [str(parquet), source_order, str(parquet.resolve())])
                after = int(
                    connection.execute("SELECT count(*) FROM selected_rows").fetchone()[0]
                )
                kept = after - before
                report["selected_unique"] += kept
                report["duplicate_configuration_ids"] += int(chosen_rows) - kept
                duplicate_rows += int(chosen_rows) - kept
                selected_omat_rows += kept
                # Exact total atoms after duplicate removal is recomputed below.
                source_order += 1
            corpus_reports.append(report)

        selected_omat_atoms = int(
            connection.execute(
                "SELECT coalesce(sum(nsites), 0) FROM selected_rows"
            ).fetchone()[0]
        )
        if selected_tier == "max":
            duplicate_rows = int(
                sum(int(report["selected_before_dedup"]) for report in corpus_reports)
                - selected_omat_rows
            )
        reference_target = min(1_000_000, int(selected_omat_rows))
        connection.execute(f"""
            CREATE TEMP TABLE atomic_reference_sample AS
            SELECT row_number() OVER () AS sample_index, p.energy, p.atomic_numbers
            FROM read_parquet(?, filename=true, file_row_number=true, union_by_name=true) p
            JOIN selected_rows s
              ON p.filename = s.source_file AND p.file_row_number = s.row_index
            USING SAMPLE reservoir ({reference_target} ROWS) REPEATABLE ({int(seed)})
        """, [[str(path.resolve()) for path in parquet_files]])
        reference_rows = int(
            connection.execute("SELECT count(*) FROM atomic_reference_sample").fetchone()[0]
        )
        connection.execute("""
            CREATE TEMP TABLE atomic_reference_counts AS
            SELECT sample_index, energy, entry.key::INTEGER AS atomic_number,
                   entry.value::DOUBLE AS count
            FROM atomic_reference_sample,
                 UNNEST(map_entries(list_histogram(atomic_numbers))) AS values(entry)
        """)
        rhs_rows = connection.execute("""
            SELECT atomic_number, sum(count * energy)
            FROM atomic_reference_counts GROUP BY atomic_number
        """).fetchall()
        normal_rows = connection.execute("""
            SELECT left_count.atomic_number, right_count.atomic_number,
                   sum(left_count.count * right_count.count)
            FROM atomic_reference_counts left_count
            JOIN atomic_reference_counts right_count USING (sample_index)
            GROUP BY left_count.atomic_number, right_count.atomic_number
        """).fetchall()
        reference_elements = sorted(omat_elements)
        reference_index = {value: index for index, value in enumerate(reference_elements)}
        reference_normal = np.zeros(
            (len(reference_elements), len(reference_elements)), dtype=np.float64
        )
        reference_rhs = np.zeros((len(reference_elements),), dtype=np.float64)
        for atomic_number, value in rhs_rows:
            if int(atomic_number) in reference_index:
                reference_rhs[reference_index[int(atomic_number)]] = float(value)
        for left, right, value in normal_rows:
            if int(left) in reference_index and int(right) in reference_index:
                reference_normal[
                    reference_index[int(left)], reference_index[int(right)]
                ] = float(value)

        with h5py.File(large, "r") as large_handle:
            if str(large_handle.attrs.get("schema_version", "")) != HDF5_SCHEMA_VERSION:
                raise ValueError("Neo Large has an unsupported HDF5 schema")
            large_ptr = np.asarray(large_handle["structures/atom_ptr"], dtype=np.int64)
            large_structures = int(len(large_ptr) - 1)
            large_atoms = int(large_ptr[-1])
            large_elements = sorted(
                int(value) for value in np.unique(
                    large_handle["structures/atomic_numbers"][:]
                )
            )
            large_periodic = int(np.count_nonzero(np.any(
                np.asarray(large_handle["structures/pbc"][:], dtype=bool), axis=1
            )))
            large_labels = {
                name: int(np.count_nonzero(large_handle["masks"][name][:]))
                for name in large_handle["masks"]
            }
            large_sources = dict(collections.Counter(
                str(value) for value in large_handle["metadata/source"].asstr()[:]
            ))
            large_splits = dict(collections.Counter(
                str(value) for value in large_handle["metadata/split"].asstr()[:]
            ))

        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = COMPOSITE_HDF5_SCHEMA_VERSION
            handle.attrs["created_at"] = _now()
            handle.attrs["units_json"] = json.dumps(HDF5_UNITS, sort_keys=True)
            handle.attrs["tier"] = selected_tier
            handle.attrs["seed"] = int(seed)
            handle.attrs["large_storage"] = "embedded-canonical-root"
            handle.attrs["selection_policy"] = (
                "all unique OMat24 configuration_id values"
                if selected_tier == "max"
                else "stable SHA-256 material-family selection: first byte < 64 (25%)"
            )
            sources = handle.create_group("sources")
            omat = sources.create_group("omat24")
            omat.attrs["root"] = os.path.relpath(root, output.parent)
            omat.attrs["license"] = "CC-BY-4.0"
            omat.attrs["selection_hash_namespace"] = "neo-plus-v1"
            selected_group = handle.create_group("selection")
            string_dtype = h5py.string_dtype("utf-8")
            cursor = connection.execute(
                "SELECT source_order, row_index, split, nsites FROM selected_rows "
                "ORDER BY source_order, row_index"
            )
            chunk = 250_000
            datasets = {
                "source_order": selected_group.create_dataset(
                    "source_order", shape=(0,), maxshape=(None,), dtype=np.uint16,
                    chunks=(chunk,), compression="gzip"
                ),
                "row_index": selected_group.create_dataset(
                    "row_index", shape=(0,), maxshape=(None,), dtype=np.uint32,
                    chunks=(chunk,), compression="gzip"
                ),
                "split_code": selected_group.create_dataset(
                    "split_code", shape=(0,), maxshape=(None,), dtype=np.uint8,
                    chunks=(chunk,), compression="gzip"
                ),
                "atom_count": selected_group.create_dataset(
                    "atom_count", shape=(0,), maxshape=(None,), dtype=np.uint16,
                    chunks=(chunk,), compression="gzip"
                ),
            }
            source_files = [
                item for report in corpus_reports for item in report["parquet_files"]
            ]
            omat.create_dataset(
                "file_paths",
                data=np.asarray([item["path"] for item in source_files], dtype=object),
                dtype=string_dtype,
            )
            omat.create_dataset(
                "file_sha256",
                data=np.asarray([item["sha256"] for item in source_files], dtype=object),
                dtype=string_dtype,
            )
            omat.create_dataset(
                "file_rows", data=np.asarray([item["rows"] for item in source_files], dtype=np.int64)
            )
            written = 0
            while True:
                values = cursor.fetchmany(chunk)
                if not values:
                    break
                end = written + len(values)
                for dataset in datasets.values():
                    dataset.resize((end,))
                datasets["source_order"][written:end] = np.asarray(
                    [value[0] for value in values], dtype=np.uint16
                )
                datasets["row_index"][written:end] = np.asarray(
                    [value[1] for value in values], dtype=np.uint32
                )
                split_codes = {"train": 0, "val": 1, "test": 2}
                datasets["split_code"][written:end] = np.asarray(
                    [split_codes[str(value[2])] for value in values], dtype=np.uint8
                )
                datasets["atom_count"][written:end] = np.asarray(
                    [value[3] for value in values], dtype=np.uint16
                )
                written = end
            large_group = sources.create_group("neo_large")
            large_group.attrs["embedded"] = True
            large_group.attrs["original_path"] = os.path.relpath(
                large, output.parent
            )
            large_group.attrs["source_sha256"] = sha256_file(str(large))
            large_group.attrs["source_bytes"] = int(large.stat().st_size)
            large_group.attrs["structures"] = large_structures
            large_group.attrs["atoms"] = large_atoms
            large_group.attrs["payload_groups_json"] = json.dumps(
                HDF5_CANONICAL_ROOT_GROUPS
            )
            with h5py.File(large, "r") as large_handle:
                for name in HDF5_CANONICAL_ROOT_GROUPS:
                    large_handle.copy(name, handle, name=name)
            reference_group = handle.create_group("atomic_reference")
            reference_group.attrs["policy"] = (
                "deterministic reservoir sample of selected OMat24 only; "
                "Neo Large absolute energies retain a distinct MP2020 reference"
            )
            reference_group.attrs["samples"] = int(reference_rows)
            reference_group.create_dataset(
                "elements", data=np.asarray(reference_elements, dtype=np.int16)
            )
            reference_group.create_dataset("normal_matrix", data=reference_normal)
            reference_group.create_dataset("rhs", data=reference_rhs)
            metadata = {
                "dataset": f"Neo {selected_tier.title()} OMat24 + Large composite",
                "corpus_role": "foundation-response-curriculum",
                "manifest_version": NEO_MANIFEST_VERSION,
                "seed": int(seed),
                "omat24": {
                    "selected_unique_structures": int(selected_omat_rows),
                    "atoms": int(selected_omat_atoms),
                    "elements": reference_elements,
                    "duplicate_configuration_ids_skipped": int(duplicate_rows),
                    "corpora": corpus_reports,
                },
                "neo_large": {
                    "structures": large_structures,
                    "atoms": large_atoms,
                    "elements": large_elements,
                    "periodic_structures": large_periodic,
                    "labels": large_labels,
                    "sources": large_sources,
                    "splits": large_splits,
                },
                "curriculum": {
                    "base": "OMat24 L1 foundation only; its PBE+U energy reference remains isolated",
                    "response": "Large records carrying L2/L3 response or spin labels",
                    "joint": (
                        "role-balanced OMat24 + complete Large coupling corpus; "
                        "incompatible Large absolute energies are masked"
                    ),
                },
                "precision_policy": "float64 source values; no coordinate, energy, force, or stress quantization",
            }
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            handle.attrs["structures"] = int(selected_omat_rows + large_structures)
            handle.attrs["omat24_structures"] = int(selected_omat_rows)
            handle.attrs["large_structures"] = int(large_structures)
            handle.attrs["atoms"] = int(selected_omat_atoms + large_atoms)
            handle.attrs["elements_json"] = json.dumps(
                sorted(set(reference_elements) | set(large_elements))
            )
        temporary.replace(output)
    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

    result = inspect_composite_dataset(str(output), verify_sources=False)
    output_sha256 = sha256_file(str(output))
    result.update({
        "output": str(output),
        "sha256": output_sha256,
        "tier": selected_tier,
        "duplicate_configuration_ids_skipped": int(duplicate_rows),
        "corpora": corpus_reports,
    })
    if report_path:
        report = Path(report_path).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def embed_neo_large_in_composite(
    path: str,
    large_hdf5: str,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Atomically upgrade an external-Large Plus/Max file to self-contained HDF5."""
    composite = Path(path).expanduser().resolve()
    large = Path(large_hdf5).expanduser().resolve()
    if not composite.is_file() or not large.is_file():
        raise FileNotFoundError(f"Composite or Large dataset is unavailable: {composite}, {large}")
    temporary = composite.with_name(composite.name + ".embedding")
    if temporary.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace temporary file: {temporary}")
        temporary.unlink()
    try:
        shutil.copy2(composite, temporary)
        with h5py.File(large, "r") as large_handle, h5py.File(temporary, "r+") as handle:
            if str(handle.attrs.get("schema_version", "")) != COMPOSITE_HDF5_SCHEMA_VERSION:
                raise ValueError("Target is not an E3MU Plus/Max composite")
            if str(large_handle.attrs.get("schema_version", "")) != HDF5_SCHEMA_VERSION:
                raise ValueError("Neo Large has an unsupported HDF5 schema")
            large_group = handle["sources/neo_large"]
            already_embedded = bool(large_group.attrs.get("embedded", False))
            present = [name for name in HDF5_CANONICAL_ROOT_GROUPS if name in handle]
            if (already_embedded or present) and not overwrite:
                raise FileExistsError("Composite already contains a canonical Large payload")
            for name in present:
                del handle[name]
            for name in HDF5_CANONICAL_ROOT_GROUPS:
                large_handle.copy(name, handle, name=name)
            source_sha = str(
                large_group.attrs.get(
                    "source_sha256", large_group.attrs.get("sha256", "")
                )
            ) or sha256_file(str(large))
            source_path = str(
                large_group.attrs.get(
                    "original_path", large_group.attrs.get("path", large.name)
                )
            )
            for obsolete in ("path", "sha256"):
                if obsolete in large_group.attrs:
                    del large_group.attrs[obsolete]
            large_group.attrs["embedded"] = True
            large_group.attrs["original_path"] = source_path
            large_group.attrs["source_sha256"] = source_sha
            large_group.attrs["source_bytes"] = int(large.stat().st_size)
            large_group.attrs["payload_groups_json"] = json.dumps(
                HDF5_CANONICAL_ROOT_GROUPS
            )
            handle.attrs["large_storage"] = "embedded-canonical-root"
            handle.attrs["large_embedded_at"] = _now()
            handle.flush()
        validation = inspect_composite_dataset(str(temporary), verify_sources=False)
        if not validation["valid"] or not validation["embedded_large"]:
            raise ValueError(
                "Embedded Large validation failed: "
                + json.dumps(validation.get("embedded_errors", []))
            )
        temporary.replace(composite)
    finally:
        temporary.unlink(missing_ok=True)
    return inspect_composite_dataset(str(composite), verify_sources=False)


def _neo_release_safe_value(value: Any, neo_root: Path) -> Any:
    """Remove workstation-only paths and transport details from release JSON."""
    if isinstance(value, dict):
        return {
            str(key): _neo_release_safe_value(item, neo_root)
            for key, item in value.items()
            if str(key) not in {"proxy_used", "notice_source"}
        }
    if isinstance(value, (list, tuple)):
        return [_neo_release_safe_value(item, neo_root) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("Datasets/Neo/"):
        return value[len("Datasets/Neo/") :]
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return value
    resolved = candidate.resolve(strict=False)
    root = neo_root.resolve()
    project_root = root.parent.parent if root.parent.name == "Datasets" else None
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        pass
    if project_root is not None:
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            pass
    return f"<external-source>/{resolved.name}"


def _copy_neo_release_metadata(source: Path, destination: Path, neo_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        portable = _neo_release_safe_value(payload, neo_root)
        destination.write_text(
            json.dumps(_checkpoint_safe(portable), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        shutil.copy2(source, destination)


def _neo_release_rights_issues(
    registry: Dict[str, Any],
    tier_sources: Dict[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    aliases: Dict[str, Dict[str, Any]] = {}
    for entry in registry.get("sources", []):
        for source_id in entry.get("neo_source_ids", []):
            aliases[str(source_id)] = dict(entry)
    source_tiers: Dict[str, List[str]] = {}
    for tier, sources in tier_sources.items():
        for source_id in sources:
            source_tiers.setdefault(str(source_id), []).append(str(tier))
    issues: List[Dict[str, Any]] = []
    for source_id, tiers in sorted(source_tiers.items()):
        entry = aliases.get(source_id)
        if entry is None:
            issues.append({
                "source_id": source_id,
                "registry_id": None,
                "status": "unregistered",
                "tiers": sorted(set(tiers)),
                "reason": "No source-rights registry entry exists.",
            })
            continue
        status = str(entry.get("redistribution_status", "unregistered"))
        if status not in _NEO_HF_ALLOWED_RIGHTS_STATUSES:
            issues.append({
                "source_id": source_id,
                "registry_id": str(entry.get("id", "unknown")),
                "status": status,
                "tiers": sorted(set(tiers)),
                "reason": str(
                    entry.get(
                        "release_reason",
                        "Source terms do not currently authorize an unqualified public release.",
                    )
                ),
            })
    return issues


def prepare_neo_huggingface_release(
    neo_root: str,
    output_directory: str,
    *,
    tiers: Sequence[str] = ("tiny", "small", "standard", "large"),
    acknowledge_rights_review: bool = False,
    validate_hdf5: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a portable, checksummed Hugging Face staging directory.

    The rights acknowledgement only permits local staging. It never changes a
    source status, declares permission, authenticates to Hugging Face, or
    uploads a file.
    """
    if not HAS_H5PY:
        raise RuntimeError("Hugging Face release preparation requires h5py")
    root = Path(neo_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Neo dataset root does not exist: {root}")
    if output == root or output in root.parents:
        raise ValueError("Release output must not be the Neo root or its parent")

    selected_tiers: List[str] = []
    for raw_name in tiers:
        name = str(raw_name).strip().lower()
        if name not in NEO_HF_TIER_PATHS:
            raise ValueError(
                f"Unknown Neo release tier {raw_name!r}; choose from "
                f"{sorted(NEO_HF_TIER_PATHS)}"
            )
        if name not in selected_tiers:
            selected_tiers.append(name)
    if not selected_tiers:
        raise ValueError("At least one Neo release tier is required")

    required_paths = [root / name for name in NEO_HF_REQUIRED_DOCUMENTS]
    required_paths.extend((
        root / "manifest.json",
        root / ".gitattributes",
        root / "provenance/source_registry.json",
    ))
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release metadata is incomplete: {missing}")

    registry = json.loads(
        (root / "provenance/source_registry.json").read_text(encoding="utf-8")
    )
    tier_sources: Dict[str, List[str]] = {}
    for name in selected_tiers:
        source_path = root / NEO_HF_TIER_PATHS[name]
        if not source_path.is_file():
            raise FileNotFoundError(f"Neo tier is missing: {source_path}")
        with h5py.File(source_path, "r") as handle:
            schema = str(handle.attrs.get("schema_version", ""))
            if schema == HDF5_SCHEMA_VERSION:
                tier_sources[name] = sorted(
                    set(str(value) for value in handle["metadata/source"].asstr()[:])
                )
            elif schema == COMPOSITE_HDF5_SCHEMA_VERSION:
                metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
                sources = set(
                    str(value)
                    for value in dict(metadata.get("neo_large", {}).get("sources", {}))
                )
                sources.update(
                    f"OMat24/{item.get('name', 'unknown')}"
                    for item in metadata.get("omat24", {}).get("corpora", [])
                )
                tier_sources[name] = sorted(sources)
            else:
                raise ValueError(f"Unsupported HDF5 schema in {source_path}")
    rights_issues = _neo_release_rights_issues(registry, tier_sources)
    if rights_issues and not acknowledge_rights_review:
        raise PermissionError(
            "Public-release rights are not clear for selected Neo tiers. "
            "Resolve the source registry issues or pass "
            "--acknowledge-rights-review for local staging only: "
            + json.dumps(rights_issues, sort_keys=True)
        )

    used_sources = {source for values in tier_sources.values() for source in values}
    for entry in registry.get("sources", []):
        aliases = {str(value) for value in entry.get("neo_source_ids", [])}
        if not (aliases & used_sources) or not entry.get("required_notice"):
            continue
        source_notice = root / str(entry.get("notice_source", ""))
        if not source_notice.is_file():
            raise FileNotFoundError(
                f"Required source notice is missing for {entry.get('id')}: {source_notice}"
            )

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Release output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    metadata_sources: List[Path] = list(required_paths[:-1])
    metadata_sources.append(root / "provenance/source_registry.json")
    for directory in ("config", "presets", "provenance"):
        metadata_sources.extend(sorted((root / directory).glob("*.json")))
    metadata_sources.extend(sorted((root / "reports").glob("*.json")))
    unique_metadata: Dict[str, Path] = {}
    for source in metadata_sources:
        relative = source.relative_to(root).as_posix()
        if "plus_half" in relative.lower() or "plus-half" in relative.lower():
            continue
        unique_metadata[relative] = source
    for relative, source in sorted(unique_metadata.items()):
        _copy_neo_release_metadata(source, output / relative, root)

    (output / ".gitignore").write_text(
        ".DS_Store\n.cache/\n*.partial\n*.building\n",
        encoding="utf-8",
    )

    for entry in registry.get("sources", []):
        aliases = {str(value) for value in entry.get("neo_source_ids", [])}
        required_notice = entry.get("required_notice")
        if not (aliases & used_sources) or not required_notice:
            continue
        source_notice = root / str(entry["notice_source"])
        destination = output / str(required_notice)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_notice, destination)

    source_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in source_manifest.get("canonical_datasets", [])
    }
    release_status = (
        "blocked_pending_rights_confirmation" if rights_issues else "ready_for_upload"
    )
    staged_tiers: List[Dict[str, Any]] = []
    staged_hdf5: Dict[str, Dict[str, Any]] = {}
    for name in selected_tiers:
        relative = NEO_HF_TIER_PATHS[name]
        source = root / relative
        source_sha256 = sha256_file(str(source))
        expected_sha256 = expected_hashes.get(relative)
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise ValueError(
                f"Canonical checksum mismatch for {relative}: "
                f"expected {expected_sha256}, got {source_sha256}"
            )
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        with h5py.File(destination, "r+") as handle:
            metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
            units = json.loads(str(handle.attrs.get("units_json", "{}")))
            units.update(HDF5_UNITS)
            handle.attrs["metadata_json"] = json.dumps(
                _checkpoint_safe(_neo_release_safe_value(metadata, root)),
                sort_keys=True,
            )
            handle.attrs["units_json"] = json.dumps(units, sort_keys=True)
            handle.attrs["release_source_sha256"] = source_sha256
            handle.attrs["release_source_path"] = relative
            handle.attrs["release_transform"] = (
                "metadata-only portability normalization; numerical HDF5 datasets unchanged"
            )
            handle.attrs["release_rights_status"] = release_status
            attribute_text = "\n".join(str(value) for value in handle.attrs.values())
            private_endpoint = re.compile(
                r"(?i)(?:https?://)?(?:127\.0\.0\.1|localhost):[0-9]{2,5}"
            )
            if (
                str(Path.home()) + "/" in attribute_text
                or private_endpoint.search(attribute_text)
            ):
                raise ValueError(f"Private workstation metadata remains in {relative}")
        validation = (
            validate_neo_hdf5(str(destination))
            if validate_hdf5
            else {
                "valid": True,
                "sha256": sha256_file(str(destination)),
                "warnings": [],
                "errors": [],
                **hdf5_dataset_summary(str(destination)),
            }
        )
        if not bool(validation.get("valid")):
            raise ValueError(
                f"Staged HDF5 validation failed for {relative}: {validation.get('errors')}"
            )
        tier_entry = {
            "name": name,
            "path": relative,
            "source_sha256": source_sha256,
            "release_sha256": str(validation["sha256"]),
            "bytes": int(destination.stat().st_size),
            "structures": int(validation["structures"]),
            "atoms": int(validation["atoms"]),
            "groups": int(validation["groups"]),
            "splits": dict(validation["splits"]),
            "sources": dict(validation["sources"]),
            "warnings": list(validation.get("warnings", [])),
        }
        staged_tiers.append(tier_entry)
        staged_hdf5[relative] = tier_entry

    private_home = str(Path.home()) + "/"
    private_endpoint = re.compile(
        r"(?i)(?:https?://)?(?:127\.0\.0\.1|localhost):[0-9]{2,5}"
    )
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() == ".h5":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if private_home in text or private_endpoint.search(text):
            raise ValueError(f"Private workstation metadata remains in {path}")

    release_files: List[Dict[str, Any]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in {"RELEASE_MANIFEST.json", "checksums.sha256"}:
            continue
        item: Dict[str, Any] = {
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": (
                staged_hdf5[relative]["release_sha256"]
                if relative in staged_hdf5
                else sha256_file(str(path))
            ),
        }
        if relative in staged_hdf5:
            item["source_sha256"] = staged_hdf5[relative]["source_sha256"]
            item["role"] = "canonical_hdf5_tier"
        else:
            item["role"] = "documentation_or_provenance"
        release_files.append(item)

    release_manifest = {
        "schema": "e3mu-neo-huggingface-release-v1",
        "created_at": _now(),
        "dataset_schema": HDF5_SCHEMA_VERSION,
        "huggingface_license": "other",
        "technical_staging": "complete",
        "release_status": release_status,
        "rights_review_acknowledged_for_local_staging": bool(
            acknowledge_rights_review and rights_issues
        ),
        "rights_acknowledgement_scope": (
            "Local staging only; not evidence of permission and not authorization to upload."
        ),
        "rights_issues": rights_issues,
        "tiers": staged_tiers,
        "files": release_files,
        "transformations": [
            "Sanitized absolute local paths in release JSON and HDF5 root metadata.",
            "Completed units_json from the e3mu-hdf5-v1 schema registry.",
            "Added release source checksum and metadata-only transformation attributes.",
            "Did not modify any numerical HDF5 dataset, mask, ID, or split.",
        ],
        "excluded": [
            "raw archives and VASP outputs",
            "combined static/response extXYZ archives",
            "developer smoke intermediates",
            "candidate and selection JSONL files",
            "checkpoints, training outputs, caches, and local proxy settings",
        ],
    }
    manifest_path = output / "RELEASE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(_checkpoint_safe(release_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(str(path))}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "release_status": release_status,
        "rights_issues": rights_issues,
        "tiers": staged_tiers,
        "files": len(checksum_paths) + 1,
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "release_manifest": str(manifest_path),
        "checksums": str(checksum_path),
        "checksums_sha256": sha256_file(str(checksum_path)),
        "uploaded": False,
    }


def _base_magnetic_structures() -> Dict[str, Atoms]:
    from ase.build import bulk, make_supercell

    fe = bulk("Fe", "bcc", a=2.866, cubic=True).repeat((2, 2, 2))
    fe.set_pbc((True, True, True))

    nio = bulk("NiO", "rocksalt", a=4.17, cubic=True).repeat((1, 1, 2))
    nio.set_pbc((True, True, True))

    fe_unit = bulk("Fe", "bcc", a=2.866, cubic=True)
    transform = np.asarray([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)
    fe_slab = make_supercell(fe_unit, transform).repeat((1, 1, 3))
    nio_slab = bulk("NiO", "rocksalt", a=4.17, cubic=True).repeat((1, 1, 2))
    target_xy = np.asarray(nio_slab.cell.array[:2], dtype=float)
    fe_cell = np.asarray(fe_slab.cell.array, dtype=float)
    scale = np.linalg.norm(target_xy, axis=1) / np.linalg.norm(fe_cell[:2], axis=1)
    scaled = fe_cell.copy()
    scaled[0] *= scale[0]
    scaled[1] *= scale[1]
    fe_slab.set_cell(scaled, scale_atoms=True)
    nio_z = float(nio_slab.cell.lengths()[2])
    fe_slab.positions[:, 2] -= float(fe_slab.positions[:, 2].min())
    fe_slab.positions[:, 2] += nio_z + 2.0
    interface = nio_slab + fe_slab
    interface.set_cell(
        [target_xy[0], target_xy[1], [0.0, 0.0, nio_z + float(fe_slab.cell.lengths()[2]) + 17.0]],
        scale_atoms=False,
    )
    interface.center(vacuum=7.5, axis=2)
    interface.set_pbc((True, True, False))
    return {"bcc_fe": fe, "nio_afm2": nio, "fe_nio_001": interface}


def _initial_spin_vectors(atoms: Atoms, system_id: str) -> np.ndarray:
    spins = np.zeros((len(atoms), 3), dtype=float)
    symbols = atoms.get_chemical_symbols()
    if system_id == "bcc_fe":
        spins[:, 2] = 1.0
        return spins
    ni_indices = [index for index, symbol in enumerate(symbols) if symbol == "Ni"]
    in_plane_lengths = np.linalg.norm(np.asarray(atoms.cell.array[:2], dtype=float), axis=1)
    nio_lattice = float(np.mean(in_plane_lengths))
    ni_plane_coordinates = {
        index: float(np.sum(atoms.positions[index]) / max(nio_lattice, 1e-12))
        for index in ni_indices
    }
    plane_origin = min(ni_plane_coordinates.values()) if ni_plane_coordinates else 0.0
    for index, symbol in enumerate(symbols):
        if symbol == "Fe":
            spins[index, 2] = 1.0
        elif symbol == "Ni":
            # Type-II AFM: each (111) Ni plane is ferromagnetic and adjacent
            # planes, separated by a/sqrt(3), have opposite orientation.
            layer = int(np.rint(ni_plane_coordinates[index] - plane_origin))
            spins[index, 2] = 1.0 if layer % 2 == 0 else -1.0
    return spins


def _vasp_species_order(system_id: str) -> Tuple[str, ...]:
    if system_id == "bcc_fe":
        return ("Fe",)
    if system_id == "nio_afm2":
        return ("Ni", "O")
    return ("Ni", "O", "Fe")


def _order_atoms_for_vasp(atoms: Atoms, system_id: str) -> Atoms:
    symbols = atoms.get_chemical_symbols()
    order = _vasp_species_order(system_id)
    indices = [index for symbol in order for index, value in enumerate(symbols) if value == symbol]
    if len(indices) != len(atoms):
        missing = sorted(set(symbols) - set(order))
        raise ValueError(f"No VASP species order for {system_id}: {missing}")
    return atoms[indices]


def _spin_family(base: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    """Eight states separating exchange, SOC anisotropy, DMI, and time reversal."""
    reference = np.asarray(base, dtype=float).copy()
    magnetic_indices = np.flatnonzero(np.linalg.norm(reference, axis=1) > 0.0)
    if magnetic_indices.size == 0:
        return [reference.copy() for _ in range(8)]
    reference[magnetic_indices] /= np.linalg.norm(
        reference[magnetic_indices], axis=1, keepdims=True
    ).clip(min=1e-12)

    flipped = reference.copy()
    flip_index = int(magnetic_indices[int(rng.integers(0, magnetic_indices.size))])
    flipped[flip_index] *= -1.0

    axis_x = reference.copy()
    signs_x = np.sign(reference[magnetic_indices, 2])
    signs_x[signs_x == 0.0] = 1.0
    axis_x[magnetic_indices] = 0.0
    axis_x[magnetic_indices, 0] = signs_x

    angles = np.linspace(0.0, 2.0 * np.pi, magnetic_indices.size, endpoint=False)
    cone_z = 0.60
    cone_xy = math.sqrt(1.0 - cone_z * cone_z)
    chiral_plus = reference.copy()
    chiral_minus = reference.copy()
    signs = np.sign(reference[magnetic_indices, 2])
    signs[signs == 0.0] = 1.0
    chiral_plus[magnetic_indices, 0] = cone_xy * np.cos(angles)
    chiral_plus[magnetic_indices, 1] = cone_xy * np.sin(angles)
    chiral_plus[magnetic_indices, 2] = cone_z * signs
    chiral_minus[magnetic_indices, 0] = cone_xy * np.cos(angles)
    chiral_minus[magnetic_indices, 1] = -cone_xy * np.sin(angles)
    chiral_minus[magnetic_indices, 2] = cone_z * signs
    return [
        reference,
        -reference,
        flipped,
        -flipped,
        reference.copy(),
        axis_x,
        chiral_plus,
        chiral_minus,
    ]


def _vasp_incar_text(
    system_id: str,
    spins: np.ndarray,
    *,
    soc: bool,
    species: Optional[Sequence[str]] = None,
) -> str:
    atom_species = list(species or (["Fe"] * len(spins)))
    moment_scale = {"Fe": 2.2, "Ni": 1.7, "O": 0.0}
    moments_array = np.asarray(spins, dtype=float) * np.asarray(
        [moment_scale.get(symbol, 1.0) for symbol in atom_species], dtype=float
    ).reshape(-1, 1)
    moments = " ".join(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in moments_array)
    lines = [
        "SYSTEM = E3MU_" + system_id,
        "ENCUT = 520",
        "PREC = Accurate",
        "EDIFF = 1E-7",
        "EDIFFG = -1E-3",
        "IBRION = -1",
        "NSW = 0",
        "ISYM = 0",
        "LASPH = .TRUE.",
        "LREAL = .FALSE.",
        "LCHARG = .FALSE.",
        "LWAVE = .FALSE.",
        "LNONCOLLINEAR = .TRUE.",
        "I_CONSTRAINED_M = 1",
        "LAMBDA = 10",
        "M_CONSTR = " + moments,
        "MAGMOM = " + moments,
    ]
    if system_id == "bcc_fe":
        lines.extend(["ISMEAR = 1", "SIGMA = 0.10"])
    else:
        ordered_species = _vasp_species_order(system_id)
        ldau_l = ["2" if symbol == "Ni" else "-1" for symbol in ordered_species]
        ldau_u = ["5.0" if symbol == "Ni" else "0.0" for symbol in ordered_species]
        ldau_j = ["0.0" for _ in ordered_species]
        lines.extend(
            [
                "ISMEAR = 0",
                "SIGMA = 0.05",
                "LDAU = .TRUE.",
                "LDAUTYPE = 2",
                "LDAUL = " + " ".join(ldau_l),
                "LDAUU = " + " ".join(ldau_u),
                "LDAUJ = " + " ".join(ldau_j),
                "LMAXMIX = 4",
            ]
        )
    if soc:
        lines.extend(["LSORBIT = .TRUE.", "GGA_COMPAT = .FALSE.", "SAXIS = 0 0 1"])
    if system_id == "fe_nio_001":
        lines.extend(["LDIPOL = .TRUE.", "IDIPOL = 3"])
    return "\n".join(lines) + "\n"


def _vasp_kpoints_text(system_id: str) -> str:
    mesh = (6, 6, 6) if system_id == "bcc_fe" else ((4, 4, 4) if system_id == "nio_afm2" else (5, 5, 1))
    return "Automatic mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*mesh)


def generate_vasp_magnetic_jobs(
    output_dir: str,
    *,
    total_jobs: int = 360,
    seed: int = 20260718,
    overwrite_metadata: bool = False,
) -> Dict[str, Any]:
    """Generate constrained non-collinear Fe/NiO/interface jobs without POTCAR data."""
    from ase.io import write as ase_write

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    systems = _base_magnetic_structures()
    family_size = 8
    per_system = int(total_jobs) // len(systems)
    if per_system < family_size:
        raise ValueError("total_jobs must provide at least eight jobs per magnetic system")
    per_system -= per_system % family_size
    rng = np.random.default_rng(seed)
    manifest: Dict[str, Any] = {
        "schema": "e3mu-vasp-jobs-v1",
        "seed": int(seed),
        "potcar_distributed": False,
        "nio_u_eff_ev": 5.0,
        "jobs": [],
    }
    for system_id, base_atoms in systems.items():
        base_atoms = _order_atoms_for_vasp(base_atoms, system_id)
        parent_count = per_system // family_size
        base_spins = _initial_spin_vectors(base_atoms, system_id)
        for parent_index in range(parent_count):
            atoms = base_atoms.copy()
            strain = rng.normal(0.0, 0.008, size=(3, 3))
            strain = 0.5 * (strain + strain.T)
            if not bool(atoms.pbc[2]):
                strain[2, :] = 0.0
                strain[:, 2] = 0.0
            atoms.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain), scale_atoms=True)
            atoms.positions += rng.normal(0.0, 0.025, size=atoms.positions.shape)
            parent_id = f"{system_id}-parent-{parent_index:03d}"
            split = stable_split(parent_id, train=60, val=20)
            spin_family = _spin_family(base_spins, rng)
            for variant, spins in enumerate(spin_family):
                soc = variant >= 4
                sample_id = f"{parent_id}-spin-{variant}"
                job_dir = root / system_id / sample_id
                job_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = job_dir / "metadata.json"
                if metadata_path.exists() and not overwrite_metadata:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                else:
                    metadata = {
                        "sample_id": sample_id,
                        "parent_id": parent_id,
                        "system_id": system_id,
                        "split": split,
                        "soc": bool(soc),
                        "spin_variant": int(variant),
                        "spin_design": (
                            "reference" if variant == 0 else
                            "time_reversed_reference" if variant == 1 else
                            "single_site_flip" if variant == 2 else
                            "time_reversed_single_site_flip" if variant == 3 else
                            "soc_axis_z" if variant == 4 else
                            "soc_axis_x" if variant == 5 else
                            "positive_chirality" if variant == 6 else
                            "negative_chirality"
                        ),
                        "spins": spins.tolist(),
                        "method_id": "PBE+U5+SOC" if (system_id != "bcc_fe" and soc) else ("PBE+SOC" if soc else ("PBE+U5" if system_id != "bcc_fe" else "PBE")),
                        "potcar_policy": "resolve from VASP_PP_PATH; never package POTCAR",
                    }
                    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
                    ase_write(job_dir / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
                    (job_dir / "INCAR").write_text(
                        _vasp_incar_text(
                            system_id,
                            spins,
                            soc=soc,
                            species=atoms.get_chemical_symbols(),
                        ),
                        encoding="ascii",
                    )
                    (job_dir / "KPOINTS").write_text(_vasp_kpoints_text(system_id), encoding="ascii")
                manifest["jobs"].append(
                    {"directory": job_dir.relative_to(root).as_posix(), **metadata}
                )
    manifest["generated_jobs"] = len(manifest["jobs"])
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": str(manifest_path), "jobs": len(manifest["jobs"]), "root": str(root)}


def run_vasp_job(job_dir: str, *, executable: str = "vasp_std", mpi_ranks: int = 1) -> int:
    directory = Path(job_dir).expanduser().resolve()
    if not (directory / "INCAR").exists() or not (directory / "POSCAR").exists():
        raise FileNotFoundError(f"Not a generated VASP job: {directory}")
    if not (directory / "POTCAR").exists():
        raise FileNotFoundError(
            f"{directory / 'POTCAR'} is missing. Resolve licensed PAW datasets locally before running."
        )
    command = [executable] if int(mpi_ranks) <= 1 else ["mpirun", "-np", str(int(mpi_ranks)), executable]
    with (directory / "vasp.stdout").open("w", encoding="utf-8") as stdout, (
        directory / "vasp.stderr"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr, check=False)
    return int(process.returncode)


def _spin_mapping_features(
    atoms: Atoms,
    spins: np.ndarray,
    *,
    cutoff: float = 4.5,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return aggregate Heisenberg, uniaxial, and chiral design features."""
    spin_array = np.asarray(spins, dtype=float).reshape(len(atoms), 3)
    magnetic = np.linalg.norm(spin_array, axis=1) > 1e-10
    src_all, dst_all = neighbor_list("ij", atoms, cutoff=float(cutoff))
    keep = (src_all < dst_all) & magnetic[src_all] & magnetic[dst_all]
    src, dst = src_all[keep], dst_all[keep]
    if src.size:
        dots = np.sum(spin_array[src] * spin_array[dst], axis=1)
        crosses = np.cross(spin_array[src], spin_array[dst])
        x_j = -float(np.sum(dots))
        x_dmi = np.sum(crosses, axis=0)
        pairs = np.stack([src, dst], axis=1).astype(np.int64)
    else:
        x_j = 0.0
        x_dmi = np.zeros(3, dtype=float)
        pairs = np.zeros((0, 2), dtype=np.int64)
    x_di = float(np.sum(spin_array[magnetic, 2] ** 2 - 1.0 / 3.0))
    return x_j, x_di, np.asarray(x_dmi, dtype=float), pairs


def _mapped_effective_field(
    spins: np.ndarray,
    pairs: np.ndarray,
    *,
    j_effective: float,
    dmi_effective: np.ndarray,
    di_effective: np.ndarray,
) -> np.ndarray:
    spin_array = np.asarray(spins, dtype=float)
    field_array = np.zeros_like(spin_array)
    dmi = np.asarray(dmi_effective, dtype=float).reshape(3)
    di = np.asarray(di_effective, dtype=float).reshape(3, 3)
    for src, dst in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        field_array[src] += float(j_effective) * spin_array[dst] - np.cross(spin_array[dst], dmi)
        field_array[dst] += float(j_effective) * spin_array[src] - np.cross(dmi, spin_array[src])
    field_array -= 2.0 * np.einsum("ab,nb->na", di, spin_array)
    return field_array


def attach_spin_energy_mappings(
    configurations: Sequence[Configuration],
    *,
    cutoff: float = 4.5,
) -> Dict[str, Any]:
    """Fit effective J, uniaxial Di, and projected DMI from each eight-state family."""
    families: Dict[str, List[Configuration]] = {}
    for cfg in configurations:
        group_id = str(cfg.properties.get("group_id", "unknown"))
        families.setdefault(group_id, []).append(cfg)
    mapped = 0
    diagnostics: List[Dict[str, Any]] = []
    for group_id, family in families.items():
        by_variant = {
            int(round(float(cfg.properties.get("spin_variant", -1)))): cfg for cfg in family
        }
        if any(index not in by_variant for index in range(8)):
            continue
        ordered = [by_variant[index] for index in range(8)]
        energies = np.asarray([float(cfg.properties["energy"]) for cfg in ordered], dtype=float)
        features: List[Tuple[float, float, np.ndarray, np.ndarray]] = []
        for cfg in ordered:
            atoms = Atoms(
                numbers=cfg.atomic_numbers,
                positions=cfg.positions,
                cell=cfg.cell,
                pbc=cfg.pbc,
            )
            features.append(
                _spin_mapping_features(atoms, np.asarray(cfg.properties["spins"]), cutoff=cutoff)
            )
        x_j = np.asarray([item[0] for item in features])
        x_di = np.asarray([item[1] for item in features])
        x_dmi = np.stack([item[2] for item in features])

        e_reference = 0.5 * (energies[0] + energies[1])
        e_flipped = 0.5 * (energies[2] + energies[3])
        xj_reference = 0.5 * (x_j[0] + x_j[1])
        xj_flipped = 0.5 * (x_j[2] + x_j[3])
        denominator_j = xj_flipped - xj_reference
        j_effective = (e_flipped - e_reference) / denominator_j if abs(denominator_j) > 1e-12 else 0.0

        delta_chiral = x_dmi[6] - x_dmi[7]
        delta_energy = (energies[6] - energies[7]) - j_effective * (x_j[6] - x_j[7])
        denominator_dmi = float(np.dot(delta_chiral, delta_chiral))
        dmi_effective = (
            delta_energy * delta_chiral / denominator_dmi
            if denominator_dmi > 1e-16
            else np.zeros(3, dtype=float)
        )

        denominator_di = x_di[5] - x_di[4]
        residual_di = (
            energies[5]
            - energies[4]
            - j_effective * (x_j[5] - x_j[4])
            - float(np.dot(dmi_effective, x_dmi[5] - x_dmi[4]))
        )
        k_effective = residual_di / denominator_di if abs(denominator_di) > 1e-12 else 0.0
        di_effective = k_effective * (np.diag([0.0, 0.0, 1.0]) - np.eye(3) / 3.0)

        reduced_energy = energies - j_effective * x_j - k_effective * x_di - x_dmi @ dmi_effective
        offset_non_soc = float(np.mean(reduced_energy[:4]))
        offset_soc = float(np.mean(reduced_energy[4:]))
        offsets = np.asarray([offset_non_soc] * 4 + [offset_soc] * 4)
        predicted = offsets + j_effective * x_j + k_effective * x_di + x_dmi @ dmi_effective
        rmse = float(np.sqrt(np.mean((predicted - energies) ** 2)))
        tr_error = float(max(abs(energies[0] - energies[1]), abs(energies[2] - energies[3])))
        for cfg, (_, _, _, pairs) in zip(ordered, features):
            spins = np.asarray(cfg.properties["spins"], dtype=float)
            magnetic = np.linalg.norm(spins, axis=1) > 1e-10
            di_atoms = np.zeros((len(spins), 3, 3), dtype=float)
            di_atoms[magnetic] = di_effective
            cfg.properties.update(
                {
                    "J_effective": float(j_effective),
                    "DMI_effective": dmi_effective.copy(),
                    "Di_effective": di_effective.copy(),
                    "Di": di_atoms,
                    "effective_field": _mapped_effective_field(
                        spins,
                        pairs,
                        j_effective=j_effective,
                        dmi_effective=dmi_effective,
                        di_effective=di_effective,
                    ),
                    "spin_mapping_rmse": rmse,
                }
            )
            for name in (
                "J_effective", "DMI_effective", "Di_effective", "Di",
                "effective_field", "spin_mapping_rmse",
            ):
                cfg.property_weights[name] = 1.0
        mapped += 1
        diagnostics.append(
            {
                "group_id": group_id,
                "J_effective_eV": float(j_effective),
                "DMI_effective_eV": dmi_effective.tolist(),
                "K_effective_eV": float(k_effective),
                "offset_non_soc_eV": offset_non_soc,
                "offset_soc_eV": offset_soc,
                "mapping_rmse_eV": rmse,
                "time_reversal_pair_error_eV": tr_error,
            }
        )
    return {"mapped_families": mapped, "diagnostics": diagnostics}


def collect_vasp_magnetic_jobs(
    jobs_root: str,
    output_hdf5: str,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    from ase.io import read as ase_read

    root = Path(jobs_root).expanduser().resolve()
    metadata_files = sorted(root.glob("*/*/metadata.json"))
    configs: List[Configuration] = []
    failures: List[Dict[str, str]] = []
    for metadata_path in metadata_files:
        directory = metadata_path.parent
        outcar = directory / "OUTCAR"
        if not outcar.exists():
            failures.append({"directory": str(directory), "reason": "missing OUTCAR"})
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            atoms = ase_read(outcar, index=-1)
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            try:
                magmoms = np.asarray(atoms.get_magnetic_moments(), dtype=float)
                if magmoms.ndim == 1:
                    magmoms = np.pad(magmoms.reshape(-1, 1), ((0, 0), (0, 2)))
            except Exception:
                magmoms = np.zeros((len(atoms), 3), dtype=float)
            props = {
                "energy": energy,
                "forces": forces,
                "spins": np.asarray(metadata["spins"], dtype=float),
                "magnetic_moments": magmoms,
                "field": np.zeros(3, dtype=float),
                "total_charge": 0.0,
                "source": "VASP-local",
                "method_id": str(metadata["method_id"]),
                "system_id": str(metadata["system_id"]),
                "group_id": str(metadata["parent_id"]),
                "split": str(metadata["split"]),
                "spin_variant": float(metadata["spin_variant"]),
                "soc": float(bool(metadata["soc"])),
            }
            configs.append(
                Configuration(
                    atomic_numbers=np.asarray(atoms.numbers, dtype=int),
                    positions=np.asarray(atoms.positions, dtype=float),
                    properties=props,
                    property_weights={
                        "energy": 1.0,
                        "forces": 1.0,
                        "spins": 1.0,
                        "magnetic_moments": 1.0,
                        "spin_variant": 1.0,
                        "soc": 1.0,
                    },
                    cell=np.asarray(atoms.cell.array, dtype=float),
                    pbc=tuple(bool(x) for x in atoms.pbc),
                    head=str(metadata["method_id"]),
                )
            )
        except Exception as exc:
            failures.append({"directory": str(directory), "reason": f"{type(exc).__name__}: {exc}"})
    if not configs:
        return {"collected": 0, "failed": len(failures), "failures": failures, "output": None}
    mapping = attach_spin_energy_mappings(configs)
    output = write_hdf5_dataset(
        configs,
        output_hdf5,
        metadata={
            "dataset": "Fe-NiO-spin",
            "jobs_root": str(root),
            "failures": failures,
            "energy_mapping": mapping,
        },
        overwrite=overwrite,
    )
    return {
        "collected": len(configs),
        "failed": len(failures),
        "failures": failures,
        "mapped_families": int(mapping["mapped_families"]),
        "output": output,
    }

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline dataset acquisition, conversion, validation, release, "
            "and VASP data-generation tools for E3-miu-GNN."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("dataset-extxyz", help="Convert extXYZ to canonical HDF5")
    command.add_argument("input"); command.add_argument("output")
    command.add_argument("--max-frames", type=int); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-qm7x", help="Rebuild canonical QM7-X from raw HDF5")
    command.add_argument("raw"); command.add_argument("output")
    command.add_argument("--max-frames", type=int); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-so3lr", help="Convert an official SO3LR HDF5 shard")
    command.add_argument("raw"); command.add_argument("output"); command.add_argument("--dataset-name")
    command.add_argument("--max-frames", type=int); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-deepspin", help="Convert official DeepSPIN NiO raw files")
    command.add_argument("raw_directory"); command.add_argument("output")
    command.add_argument("--max-frames", type=int); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-mptrj-scan", help="Index multi-element magnetic MPtrj materials")
    command.add_argument("raw_json"); command.add_argument("index")
    command.add_argument("--min-elements", type=int, default=2); command.add_argument("--min-abs-moment", type=float, default=0.05); command.add_argument("--max-atoms", type=int, default=160)

    command = subparsers.add_parser("dataset-mptrj-select", help="Select balanced magnetic MPtrj materials")
    command.add_argument("index"); command.add_argument("selection"); command.add_argument("--target", type=int, default=12000)

    command = subparsers.add_parser("dataset-mptrj-build", help="Build a selected magnetic MPtrj HDF5 shard")
    command.add_argument("raw_json"); command.add_argument("selection"); command.add_argument("output")
    command.add_argument("--min-abs-moment", type=float, default=0.05); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-mptrj-large-build", help="Build a trajectory-rich MPtrj shard")
    command.add_argument("raw_json"); command.add_argument("output"); command.add_argument("--required-hdf5", action="append", default=[])
    command.add_argument("--max-per-material", type=int, default=4); command.add_argument("--min-elements", type=int, default=2); command.add_argument("--max-atoms", type=int, default=160); command.add_argument("--min-abs-moment", type=float, default=0.05)
    command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-static-mptrj-select", help="Recover balanced MPtrj IDs from Parquet")
    command.add_argument("parquet_directory"); command.add_argument("selection"); command.add_argument("--source-rows", type=int, default=200000); command.add_argument("--target", type=int, default=40000); command.add_argument("--min-elements", type=int, default=2); command.add_argument("--max-atoms", type=int, default=160)

    command = subparsers.add_parser("dataset-static-mptrj-build", help="Build a recovered static MPtrj HDF5 shard")
    command.add_argument("static_extxyz"); command.add_argument("parquet_directory"); command.add_argument("selection"); command.add_argument("output"); command.add_argument("--overwrite", action="store_true")

    for name, help_text in (("dataset-scfnn", "Recover SCFNN field frames"), ("dataset-bec", "Recover local DFPT BEC frames")):
        command = subparsers.add_parser(name, help=help_text); command.add_argument("response_extxyz"); command.add_argument("output"); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-jarvis-scan", help="Index complex multi-element JARVIS structures")
    command.add_argument("raw_json"); command.add_argument("index"); command.add_argument("--min-elements", type=int, default=2); command.add_argument("--max-atoms", type=int, default=160)
    command = subparsers.add_parser("dataset-jarvis-select", help="Select balanced complex JARVIS structures")
    command.add_argument("index"); command.add_argument("selection"); command.add_argument("--target", type=int, default=24000)
    command = subparsers.add_parser("dataset-jarvis-build", help="Build a selected JARVIS shard")
    command.add_argument("raw_json"); command.add_argument("selection"); command.add_argument("output"); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-jarvis-dfpt-select", help="Select complex JARVIS DFPT archives")
    command.add_argument("raw_json"); command.add_argument("selection"); command.add_argument("--target", type=int, default=300); command.add_argument("--min-elements", type=int, default=2); command.add_argument("--max-atoms", type=int, default=80)
    command = subparsers.add_parser("dataset-jarvis-dfpt-download", help="Resume-download JARVIS DFPT archives")
    command.add_argument("selection"); command.add_argument("output_directory"); command.add_argument("--proxy"); command.add_argument("--retries", type=int, default=4)
    command = subparsers.add_parser("dataset-jarvis-dfpt-build", help="Build a multi-element JARVIS BEC shard")
    command.add_argument("selection"); command.add_argument("archive_directory"); command.add_argument("output"); command.add_argument("--max-abs-bec", type=float, default=50.0); command.add_argument("--max-asr-residual", type=float, default=0.5); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-mixed-build", help="Build a mixed corpus from a source policy")
    command.add_argument("policy"); command.add_argument("output"); command.add_argument("--seed", type=int, default=20260719); command.add_argument("--overwrite", action="store_true"); command.add_argument("--report")
    command = subparsers.add_parser("dataset-smoke-build", help="Build a split-balanced smoke corpus")
    command.add_argument("policy"); command.add_argument("output"); command.add_argument("--per-source-split", type=int, default=2); command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("dataset-tier-build", help="Build a coverage-stratified Neo tier")
    command.add_argument("input"); command.add_argument("output"); command.add_argument("--target-mib", type=float, required=True); command.add_argument("--seed", type=int, default=20260720); command.add_argument("--source-temperature", type=float, default=0.5); command.add_argument("--min-per-source", type=int, default=3); command.add_argument("--max-per-group", type=int, default=2); command.add_argument("--preserve-sources-up-to", type=int, default=128); command.add_argument("--size-tolerance", type=float, default=0.05); command.add_argument("--max-calibration-rounds", type=int, default=4); command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-tier-audit", help="Audit ordered nested Neo tiers")
    command.add_argument("--tier", action="append", required=True); command.add_argument("--output")

    for name, help_text in (("dataset-validate", "Strictly validate canonical Neo HDF5"), ("dataset-summary", "Inspect canonical or composite HDF5")):
        command = subparsers.add_parser(name, help=help_text); command.add_argument("input"); command.add_argument("--output")

    command = subparsers.add_parser("dataset-omat-composite-build", help="Build Neo Plus/Max from OMat24 + Large")
    command.add_argument("omat_root"); command.add_argument("large_hdf5"); command.add_argument("output"); command.add_argument("--tier", required=True, choices=("plus", "max")); command.add_argument("--seed", type=int, default=20260722); command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-composite-half-build", help="Build local self-contained Standard + 1/N OMat24 corpus")
    command.add_argument("parent"); command.add_argument("standard"); command.add_argument("output"); command.add_argument("--seed", type=int, default=20260723); command.add_argument("--omat-denominator", type=int, default=180); command.add_argument("--max-mb", type=float, default=800.0); command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-composite-embed-omat", help="Embed all OMat24 Parquet shards inside one composite HDF5")
    command.add_argument("input"); command.add_argument("--output"); command.add_argument("--chunk-mb", type=float, default=8.0); command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-composite-materialize-omat", help="Embed only selected OMat24 rows as internal Parquet shards")
    command.add_argument("input"); command.add_argument("--output"); command.add_argument("--chunk-mb", type=float, default=8.0); command.add_argument("--batch-rows", type=int, default=32768); command.add_argument("--row-group-rows", type=int, default=4096); command.add_argument("--compression-level", type=int, default=3); command.add_argument("--report"); command.add_argument("--no-resume", action="store_true"); command.add_argument("--restart", action="store_true"); command.add_argument("--sha256", action="store_true"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-composite-pack-omat", help="Convert embedded OMat24 rows to lossless packed HDF5 arrays")
    command.add_argument("input"); command.add_argument("--output"); command.add_argument("--report"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-composite-validate", help="Verify a Plus/Max composite and source checksums")
    command.add_argument("input"); command.add_argument("--output")
    command = subparsers.add_parser("dataset-composite-embed-large", help="Embed complete Large arrays in Plus/Max")
    command.add_argument("input"); command.add_argument("large_hdf5"); command.add_argument("--overwrite", action="store_true"); command.add_argument("--output")

    command = subparsers.add_parser("dataset-hf-prepare", help="Create a rights-aware Hugging Face staging directory")
    command.add_argument("neo_root"); command.add_argument("output_directory"); command.add_argument("--tier", action="append", choices=sorted(NEO_HF_TIER_PATHS)); command.add_argument("--acknowledge-rights-review", action="store_true"); command.add_argument("--skip-hdf5-validation", action="store_true"); command.add_argument("--overwrite", action="store_true")
    command = subparsers.add_parser("dataset-download", help="Download and checksum a raw dataset")
    command.add_argument("url"); command.add_argument("output"); command.add_argument("--sha256")

    command = subparsers.add_parser("vasp-generate", help="Generate Fe/NiO magnetic VASP jobs")
    command.add_argument("output_dir"); command.add_argument("--total-jobs", type=int, default=360); command.add_argument("--seed", type=int, default=20260718); command.add_argument("--overwrite-metadata", action="store_true")
    command = subparsers.add_parser("vasp-run", help="Run one generated VASP job or job tree")
    command.add_argument("path"); command.add_argument("--executable", default="vasp_std"); command.add_argument("--mpi-ranks", type=int, default=1); command.add_argument("--limit", type=int); command.add_argument("--fail-fast", action="store_true")
    command = subparsers.add_parser("vasp-collect", help="Collect OUTCAR files into canonical HDF5")
    command.add_argument("jobs_root"); command.add_argument("output"); command.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_policy(path: str) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    policy_path = Path(path).expanduser().resolve()
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        policy = list(payload.get("sources", [])); extras = dict(payload)
    elif isinstance(payload, list):
        policy = list(payload); extras = {}
    else:
        raise ValueError("Dataset policy must be a source list or an object with 'sources'.")
    if not policy:
        raise ValueError("Dataset policy contains no sources.")
    for source in policy:
        raw = Path(str(source["path"])).expanduser()
        if not raw.is_absolute():
            candidate = (policy_path.parent / raw).resolve()
            source["path"] = str(candidate if candidate.exists() else raw.resolve())
    return policy_path, policy, extras


def _write_json(path: Optional[str], payload: Any) -> None:
    if not path:
        return
    output = Path(path).expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_checkpoint_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_cli_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    command = args.command
    if command == "dataset-extxyz": result = {"output": extxyz_to_hdf5(args.input, args.output, overwrite=args.overwrite, max_frames=args.max_frames)}
    elif command == "dataset-qm7x": result = {"output": rebuild_qm7x_hdf5(args.raw, args.output, max_frames=args.max_frames, overwrite=args.overwrite)}
    elif command == "dataset-so3lr": result = {"output": rebuild_so3lr_hdf5(args.raw, args.output, dataset_name=args.dataset_name, max_frames=args.max_frames, overwrite=args.overwrite)}
    elif command == "dataset-deepspin": result = {"output": rebuild_deepspin_hdf5(args.raw_directory, args.output, max_frames=args.max_frames, overwrite=args.overwrite)}
    elif command == "dataset-mptrj-scan": result = scan_mptrj_magnetic_candidates(args.raw_json, args.index, min_elements=args.min_elements, min_abs_moment=args.min_abs_moment, max_atoms=args.max_atoms)
    elif command == "dataset-mptrj-select": result = select_mptrj_magnetic_candidates(args.index, args.selection, target_structures=args.target)
    elif command == "dataset-mptrj-build": result = {"output": rebuild_mptrj_magnetic_hdf5(args.raw_json, args.selection, args.output, min_abs_moment=args.min_abs_moment, overwrite=args.overwrite)}
    elif command == "dataset-mptrj-large-build": result = rebuild_mptrj_large_hdf5(args.raw_json, args.output, required_hdf5=args.required_hdf5, max_per_material=args.max_per_material, min_elements=args.min_elements, max_atoms=args.max_atoms, min_abs_moment=args.min_abs_moment, report_path=args.report, overwrite=args.overwrite)
    elif command == "dataset-static-mptrj-select": result = select_static_mptrj_parquet_rows(args.parquet_directory, args.selection, source_rows=args.source_rows, target_materials=args.target, min_elements=args.min_elements, max_atoms=args.max_atoms)
    elif command == "dataset-static-mptrj-build": result = {"output": rebuild_static_mptrj_hdf5(args.static_extxyz, args.parquet_directory, args.selection, args.output, overwrite=args.overwrite)}
    elif command == "dataset-scfnn": result = {"output": rebuild_scfnn_from_combined_extxyz(args.response_extxyz, args.output, overwrite=args.overwrite)}
    elif command == "dataset-bec": result = {"output": rebuild_bec_from_combined_response(args.response_extxyz, args.output, overwrite=args.overwrite)}
    elif command == "dataset-jarvis-scan": result = scan_jarvis_multi_element_candidates(args.raw_json, args.index, min_elements=args.min_elements, max_atoms=args.max_atoms)
    elif command == "dataset-jarvis-select": result = select_jarvis_multi_element_candidates(args.index, args.selection, target_structures=args.target)
    elif command == "dataset-jarvis-build": result = {"output": rebuild_jarvis_multi_element_hdf5(args.raw_json, args.selection, args.output, overwrite=args.overwrite)}
    elif command == "dataset-jarvis-dfpt-select": result = select_jarvis_dfpt_candidates(args.raw_json, args.selection, target_structures=args.target, min_elements=args.min_elements, max_atoms=args.max_atoms)
    elif command == "dataset-jarvis-dfpt-download": result = download_jarvis_dfpt_archives(args.selection, args.output_directory, proxy=args.proxy, retries=args.retries)
    elif command == "dataset-jarvis-dfpt-build": result = {"output": rebuild_jarvis_dfpt_hdf5(args.selection, args.archive_directory, args.output, overwrite=args.overwrite, max_abs_bec=args.max_abs_bec, max_acoustic_sum_residual=args.max_asr_residual)}
    elif command == "dataset-mixed-build":
        _, policy, extras = _resolve_policy(args.policy)
        result = build_neo_mixed_dataset(policy, args.output, seed=args.seed, overwrite=args.overwrite, dataset_name=str(extras.get("dataset_name", "Neo balanced L1-L3 mixed-granularity training corpus")), corpus_role=str(extras.get("corpus_role", "balanced-training")), metadata_extra={k: v for k, v in extras.items() if k not in {"sources", "dataset_name", "corpus_role"}}); _write_json(args.report, result)
    elif command == "dataset-smoke-build": _, policy, _ = _resolve_policy(args.policy); result = build_neo_smoke_dataset(policy, args.output, per_source_split=args.per_source_split, overwrite=args.overwrite)
    elif command == "dataset-tier-build": result = build_neo_stratified_tier(args.input, args.output, target_mib=args.target_mib, seed=args.seed, source_temperature=args.source_temperature, min_per_source=args.min_per_source, max_per_group=args.max_per_group, preserve_sources_up_to=args.preserve_sources_up_to, tolerance_fraction=args.size_tolerance, max_calibration_rounds=args.max_calibration_rounds, overwrite=args.overwrite, report_path=args.report)
    elif command == "dataset-tier-audit":
        tiers = []
        for value in args.tier:
            if "=" not in value: raise ValueError("Each --tier must use NAME=HDF5 syntax")
            name, path = value.split("=", 1); tiers.append((name.strip(), path.strip()))
        result = audit_neo_tier_hierarchy(tiers, output_path=args.output)
    elif command == "dataset-validate": result = validate_neo_hdf5(args.input); _write_json(args.output, result)
    elif command == "dataset-summary": result = inspect_composite_dataset(args.input, verify_sources=False) if _core._is_composite_hdf5_path(args.input) else hdf5_dataset_summary(args.input); _write_json(args.output, result)
    elif command == "dataset-omat-composite-build": result = build_neo_omat24_composite(args.omat_root, args.large_hdf5, args.output, tier=args.tier, seed=args.seed, overwrite=args.overwrite, report_path=args.report)
    elif command == "dataset-composite-half-build": result = build_neo_composite_half(args.parent, args.standard, args.output, seed=args.seed, omat_denominator=args.omat_denominator, max_bytes=int(float(args.max_mb) * 1_000_000), overwrite=args.overwrite, report_path=args.report)
    elif command == "dataset-composite-embed-omat": result = embed_omat24_parquet_in_composite(args.input, output_path=args.output, chunk_bytes=int(float(args.chunk_mb) * 1024 * 1024), overwrite=args.overwrite, report_path=args.report)
    elif command == "dataset-composite-materialize-omat": result = materialize_omat24_selection_in_composite(args.input, output_path=args.output, chunk_bytes=int(float(args.chunk_mb) * 1024 * 1024), batch_rows=args.batch_rows, row_group_rows=args.row_group_rows, compression_level=args.compression_level, resume=not args.no_resume, restart=args.restart, overwrite=args.overwrite, compute_sha256=args.sha256, report_path=args.report)
    elif command == "dataset-composite-pack-omat": result = pack_omat24_selection_in_composite(args.input, output_path=args.output, overwrite=args.overwrite, report_path=args.report)
    elif command == "dataset-composite-validate": result = inspect_composite_dataset(args.input, verify_sources=True); _write_json(args.output, result)
    elif command == "dataset-composite-embed-large": result = embed_neo_large_in_composite(args.input, args.large_hdf5, overwrite=args.overwrite); _write_json(args.output, result)
    elif command == "dataset-hf-prepare": result = prepare_neo_huggingface_release(args.neo_root, args.output_directory, tiers=args.tier or ("tiny", "small", "standard", "large"), acknowledge_rights_review=args.acknowledge_rights_review, validate_hdf5=not args.skip_hdf5_validation, overwrite=args.overwrite)
    elif command == "dataset-download": result = {"output": download_with_sha256(args.url, args.output, expected_sha256=args.sha256)}
    elif command == "vasp-generate": result = generate_vasp_magnetic_jobs(args.output_dir, total_jobs=args.total_jobs, seed=args.seed, overwrite_metadata=args.overwrite_metadata)
    elif command == "vasp-run":
        path = Path(args.path).expanduser().resolve(); jobs = [path] if (path / "INCAR").exists() else [item.parent for item in sorted(path.glob("*/*/metadata.json"))]
        if args.limit is not None: jobs = jobs[:max(0, args.limit)]
        runs = []
        for job in jobs:
            code = run_vasp_job(str(job), executable=args.executable, mpi_ranks=args.mpi_ranks); runs.append({"directory": str(job), "return_code": code})
            if code and args.fail_fast: break
        result = {"jobs": len(runs), "failed": sum(item["return_code"] != 0 for item in runs), "results": runs}
    elif command == "vasp-collect": result = collect_vasp_magnetic_jobs(args.jobs_root, args.output, overwrite=args.overwrite)
    else: raise RuntimeError(f"Unhandled command: {command}")
    print(json.dumps(_checkpoint_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
