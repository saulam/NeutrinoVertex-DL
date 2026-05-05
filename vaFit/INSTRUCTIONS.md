# Vertex Fitting Instructions (`vaFit`)

This document explains how to run and tune the vertex fitting workflow in `vaFit` without changing the model code.

## 1) What this module does

`vaFit` performs gradient-based fitting of particle parameters to match a target vertex-activity image.

Main flow:
- Load pretrained models:
  - Transformer (for initial particle and vertex predictions)
  - CNF generators (`proton_contained`, `proton_exiting`, `muon`) for rendering charge templates
- Build fit parameters (direction, kinetic energy, vertex, optional weights/background)
- Optimize parameters with Adam (optionally followed by LBFGS)
- Return fitted kinematics/vertex and reconstructed image

Core files:
- `config.py`: fitting hyperparameters (`FitConfig`)
- `model_loader.py`: model and test-data loading
- `run_fit.py`: top-level `VertexFitter` interface
- `fit.py`: optimization loops (`fit_vertex_event`, `fit_single_track`)
- `forward.py`: rendering from kinematics to voxelized charge
- `losses.py`: loss terms (`poisson`, `robust_l2`, `KL_combined`, etc.)
- `params.py`: constrained parameterization used during optimization

## 2) Environment assumptions

The code is written for GPU-first execution and currently uses CUDA memory diagnostics (`torch.cuda.synchronize`, peak-memory prints) inside optimization loops.

Recommended:
- Python environment with project dependencies installed (`torch`, `numpy`, `tqdm`, `nflows`, project modules under `datasets/`, `models/`, `utils/`)
- Run from the `vaFit` folder (or ensure `PYTHONPATH` includes it), because several imports are local-style (`from config import ...`, `from fit import ...`)
- Existing checkpoints and metadata at the hard-coded paths in `model_loader.py`

Important path defaults in `model_loader.py`:
- Metadata: `/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl`
- Transformer ckpt root: `/pscratch/sd/b/botaoli/SFGD_VA/Results/checkpoints`
- CNF ckpt root: `/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/checkpoints_v2`

## 3) Minimal usage pattern

```python
import torch
from config import FitConfig
from model_loader import GlobalModelLoader
from run_fit import VertexFitter

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = FitConfig(
    device=device,
    data_term="KL_combined",
    adam_steps=100,
    n_sample_for_template=1,
    n_fit_per_event=1,
)

model_loader = GlobalModelLoader(device)
model_loader.load_models()

fitter = VertexFitter(model_loader, cfg)
```

After this, choose one of two fit modes:
- `fit_type="event"`: multi-particle event fit
- `fit_type="single_track"`: per-track fit for one particle type

## 4) Event fitting (`fit_type="event"`)

### Required input dataset keys

Pass a Python dictionary where each value is a list-like collection over events:

- `hits`: flattened input image per event (typically generator crop, size `img_size^3`)
- `N_pred`: predicted number of contained tracks per event
- `ekin_pred`: predicted contained-track kinetic energies per event
- `dir_pred`: predicted contained-track directions per event
- `vtx_pred`: predicted vertex per event
- `exit_particle`: exiting-particle info per event
- `ekin_true`, `dir_true`, `vtx_true`: optional truth arrays used in reporting/logging

Call:

```python
result = fitter.fit(dataset, n_events=1, fit_type="event")
```

### Output fields

`fit_vertex_event` returns a dictionary including:
- `x_recon_final`, `x_true`
- `dir_pred`, `ekin_pred`, `vtx_pred`
- `weights_pred`, `background_pred`
- `dir_true`, `ekin_true`, `vtx_true`
- `active`, `loss_history`

## 5) Single-track fitting (`fit_type="single_track"`)

Use when fitting one particle category (commonly `proton_contained`) from track-level samples.

### Required input dataset keys

- `hits`: input charge image(s), flattened
- `parameters`: true/seed parameters with shape `[N, 7]` as `[vtx(3), E(1), dir(3)]`

Call:

```python
result = fitter.fit(
    dataset,
    n_events=1,
    fit_type="single_track",
    particle="proton_contained",  # or "muon" / "proton_exiting" if supported by your data
)
```

### Output fields

`fit_single_track` returns:
- `x_recon_final`, `x_true`
- `dir_pred`, `ekin_pred`, `vtx_pred`
- `dir_initial`, `ekin_initial`, `vtx_initial`
- `weights_pred`, `background_pred`
- `dir_true`, `ekin_true`, `vtx_true`
- `active`, `loss_history`

## 6) Key `FitConfig` knobs

Most-used parameters from `config.py`:

- Geometry:
  - `va_size` (transformer image size, default `7`)
  - `img_size` (generator image size, default `5`)
- Optimization:
  - `adam_lr`, `adam_steps`
  - `use_lbfgs`, `lbfgs_steps`
- Rendering/statistics:
  - `n_sample_for_template` (Monte Carlo samples averaged per template)
  - `n_fit_per_event` (repeated fit attempts per event)
- Data term:
  - `data_term`: `"poisson"`, `"robust_l2"`, `"KL_combined"`, `"weighted_poisson"`, `"capped_log"`
- Optional structure penalties:
  - `lam_a`, `gamma_repel`, `gamma_drift`
- Optional background:
  - `background_mode`: `"none"`, `"scalar"`, `"map"`

## 7) Practical workflow

1. Start with:
   - `data_term="KL_combined"`
   - `n_sample_for_template=1`
   - `n_fit_per_event=1`
   - `use_lbfgs=False`
2. Verify fit convergence from `loss_history`.
3. Increase `n_sample_for_template` if reconstructions are too noisy.
4. Enable `use_lbfgs=True` for a second-stage refinement after Adam.
5. Compare fitted vs truth (`vtx_*`, `ekin_*`, `dir_*`) to diagnose bias.

## 8) Common pitfalls and checks

- **Import/path issues**: local imports in `vaFit` expect execution context to include the folder on `PYTHONPATH`.
- **Checkpoint mismatch**: if loading fails, validate hard-coded checkpoint/metadata paths in `model_loader.py`.
- **Shape mismatches**: ensure dataset arrays match expected event-wise structure and dimensions (especially `N_pred`, `exit_particle`, and flattened `hits` length).
- **GPU-only debug calls**: optimization loops contain CUDA synchronization and memory print statements; for CPU workflows, review these calls first.
- **Particle consistency**: in `single_track`, choose `particle` consistent with how `dataset['parameters']` and charge normalization were generated.

## 9) Optional notebook-driven workflow

Existing notebooks in this folder (`fit_vertex.ipynb`, `fit_vertex_test.ipynb`, `fit_vertex_single_part.ipynb`) already demonstrate:
- Config creation (`FitConfig(...)`)
- Model loading (`GlobalModelLoader(...).load_models()`)
- Fitter usage (`VertexFitter(...).fit(...)`)

Use them as references to construct dataset dictionaries and visualize fit outputs.
