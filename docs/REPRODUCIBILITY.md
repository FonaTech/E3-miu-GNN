# Reproducibility Guide

This guide defines the minimum procedure for reproducing the implemented
E(3)-GNN checks and starting a controlled training run. Run commands from the
repository root.

## Software environment

Python 3.10 or newer is recommended. Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies include PyTorch, PyTorch Geometric, ASE, h5py, SciPy,
Matplotlib, PyQt6, `torch-pme`, `tad-dftd4`, and the official `dftd4` reference
package. The pinned physics interfaces in `requirements.txt` are intentional.

Record the exact environment used for a result:

```bash
python --version
python -m pip freeze > environment.freeze.txt
python - <<'PY'
import platform
import torch

print(platform.platform())
print("torch", torch.__version__)
print("mps", torch.backends.mps.is_available())
print("cuda", torch.cuda.is_available())
PY
```

Do not commit `environment.freeze.txt` without reviewing platform-specific
paths and package indexes.

## Source integrity

The project is currently a local source tree and does not yet contain `.git`
metadata. Before a public release, initialize the repository, make the first
reviewed commit, and record its commit hash in every experiment report.
`CITATION.cff` intentionally contains no fabricated DOI or version.

The executable implementation is a single file:

```text
E3_miu_GNN.py
```

Historical `Dual_Layer_Atomic_E3_GNN_L*.py` snapshots are excluded by
`.gitignore`; they are not independent release implementations.

## Fast source checks

```bash
python -m py_compile E3_miu_GNN.py
pytest -q
python E3_miu_GNN.py self-test \
  --seed 7 \
  --output Validation/self_test.json
```

For the same source and float64 CPU operations, the deterministic self-test
should reproduce the invariance and finite-difference errors within ordinary
numerical variation. Different PyTorch, BLAS, or device backends may change the
last few digits.

## Canonical dataset validation

The Tiny file is available from the GitHub repository. Download the larger
tiers from the Hugging Face dataset before validation or training:

```bash
python -m pip install --upgrade huggingface_hub
hf download FonaTech/E3-miu-GNN \
  canonical/neo_small_l1_l2_l3.h5 \
  canonical/neo_mixed_l1_l2_l3.h5 \
  canonical/neo_large_l1_l2_l3.h5 \
  --repo-type dataset --local-dir Datasets/Neo
```

Then validate the canonical file used by the chosen config:

```bash
python Datasets_Preparation.py dataset-summary \
  Datasets/Neo/canonical/neo_tiny_l1_l2_l3.h5

python Datasets_Preparation.py dataset-validate \
  Datasets/Neo/canonical/neo_tiny_l1_l2_l3.h5 \
  --output Validation/neo_tiny_validation.json
```

Audit portable-tier nesting with ordered name/path pairs:

```bash
python Datasets_Preparation.py dataset-tier-audit \
  --tier tiny=Datasets/Neo/canonical/neo_tiny_l1_l2_l3.h5 \
  --tier small=Datasets/Neo/canonical/neo_small_l1_l2_l3.h5 \
  --tier standard=Datasets/Neo/canonical/neo_mixed_l1_l2_l3.h5 \
  --output Validation/neo_tier_audit.json
```

Use canonical sample IDs, source checksums, and fixed splits from the Neo
manifest. Do not regenerate a split for a published benchmark.

## Minimal training checks

CPU smoke configuration:

```bash
python E3_miu_GNN.py train \
  --config Datasets/Neo/presets/smoke_cpu.json
```

Portable Apple MPS smoke configuration:

```bash
python E3_miu_GNN.py train \
  --config Datasets/Neo/presets/portable_tier_smoke_mps.json
```

The preset paths are repository-relative. Run them from the project root or
override `dataset` and `out_ckpt` on the command line. A smoke preset checks
execution and gradients; it is not an accuracy-training schedule.

For an explicit small base run:

```bash
python E3_miu_GNN.py train \
  --dataset path/to/canonical.h5 \
  --mode base \
  --device cpu \
  --epochs 2 \
  --batch-size 2 \
  --channels 8 \
  --interactions 1 \
  --radial-basis 4 \
  --w-energy 1 \
  --w-forces 10 \
  --no-sevennet \
  --out-ckpt Validation/base_smoke.pt
```

## Evaluation

Evaluate only a finite validated checkpoint:

```bash
python E3_miu_GNN.py evaluate \
  Validation/base_smoke.pt \
  path/to/canonical.h5 \
  --split test \
  --device cpu \
  --output Validation/base_smoke_test.json
```

Report all of the following with a result:

- source commit and clean/dirty state;
- environment and device;
- complete configuration JSON;
- dataset file SHA-256 and schema version;
- fixed split and physical grouping policy;
- active target masks and loss weights;
- epoch selected by normalized validation score;
- MAE units and aggregation convention;
- solver residuals and stability shifts; and
- peak and post-cleanup memory measurements when relevant.

## GUI workflow

```bash
python E3_miu_GNN.py gui
```

The PyQt6 interface uses the same configuration builders and trainer as the
CLI. After choosing a canonical HDF5 file, wait for the dataset capability scan
before selecting architecture switches. Hover documentation describes purpose,
physical principle, dependencies, and recommended ranges. Auto Research never
applies its winner automatically; use `Apply Best` after reviewing the retained
configuration and score.

GUI defaults are stored outside the repository in
`~/.dual_layer_field_gui.defaults.json`. Archive a reviewed experiment config
instead of relying on that mutable user-level file.

## Determinism and graph construction

`seed` controls model initialization, data subsetting, batching, and search
sampling. Canonical splits are metadata-defined and stable independently of
training seed. Non-periodic KD-tree candidate generation verifies cutoff
membership in float64 and preserves the exact cutoff graph; it changes search
cost, not neighborhood precision. Periodic neighbor lists use ASE or the
optional configured backend.

GPU and MPS kernels are not guaranteed bitwise deterministic across library
versions. Reproducibility claims should use tolerances and recorded software
versions rather than exact binary equality of floating-point checkpoints.

## Apple MPS notes

- Use float32; Apple MPS does not support general float64 tensors.
- QEq uses an analytic Helmert neutral basis and reduced positive-definite
  Cholesky solve, avoiding unsupported QR and unstable indefinite KKT backward.
- The QEq eigenvalue diagnostic runs on CPU float64 and returns an on-device
  differentiable Rayleigh quotient.
- D4 runs as a differentiable CPU sublayer because its reference tables require
  float64.
- Edge-balanced batches cap higher-order force/BEC graph size.
- Each epoch reports RSS, active MPS, driver, and reclaimable-cache memory.

Do not set a global CPU fallback as the primary reproducibility strategy; it
can silently change performance and device placement. Use the implemented
device-specific paths and inspect the log.

## Checkpoints and artifacts

Native checkpoints contain a plain configuration, element table, atomic
references, and state dictionaries in a weights-only-compatible structure.
Legacy full-object pickle loading requires an explicit unsafe opt-in and should
not be used for untrusted artifacts.

When epoch artifacts are enabled, output is written under
`<checkpoint parent>/train/<checkpoint stem>/`. Keep the final config, metric
JSON, plots, memory history, dataset hash, and selected checkpoint together.
SevenNet TorchScript export represents only the ground-state branch, not the
complete Layer-2/Layer-3 solver system.

## Dataset release reproducibility

The published dataset is
[FonaTech/E3-miu-GNN](https://huggingface.co/datasets/FonaTech/E3-miu-GNN).
The local Hugging Face staging command is documented in
[`Datasets/Neo/HUGGINGFACE_UPLOAD.md`](../Datasets/Neo/HUGGINGFACE_UPLOAD.md).
It validates checksums and strips workstation metadata, but it does not grant
redistribution rights or upload automatically. The supplied BEC archive rows
retain an unresolved archive-level rights review.

## Publication checklist

1. Run compilation, regression tests, and the deterministic self-test.
2. Validate the exact canonical HDF5 and record its SHA-256.
3. Verify no physical group crosses train, validation, and test.
4. Preserve target masks and source-specific energy-reference policy.
5. Record normalized checkpoint selection and all raw held-out metrics.
6. Inspect QEq, polarization, FiLM, and memory histories.
7. State unsupported or unsupervised outputs explicitly.
8. Verify source-level dataset licenses before distributing binaries.
9. Scan tracked text for credentials, absolute workstation paths, and concrete
   proxy endpoints.
10. Archive the source commit and environment with the final result.
