# VoxelScribe — Implementation Summary

Post-implementation notes for the skeleton built against
[`Implementation_Guide.md`](./Implementation_Guide.md). Covers what was built,
what surprised us, deviations from the spec, and how to run and read the
artefacts manually.

---

## 1. What was built

A complete end-to-end Gibbs-sampling pipeline for binary ink detection on
papyrus CT slices. Every required module from section 6 of the guide exists,
the CLI in `run_scenario.py` runs scenarios end-to-end, and `comparison.png`
plus `comparison.md` are rebuilt automatically after every run.

| Module | Status | Notes |
|--------|--------|-------|
| `src/config.py` | done | Defaults adjusted to match the real dataset (see below). |
| `src/data_loader.py` | done | TIF and PNG loaders; cropping; horizontal train/test split. |
| `src/distributions.py` | done | `gaussian_log_pdf` written from scratch in log-space. |
| `src/learning.py` | done | Per-class mean / std with Bessel correction; vectorised log-unary table. |
| `src/energy.py` | done | MRF energy diagnostic; pairwise term counted once per neighbour pair. |
| `src/gibbs.py` | done | `count_neighbors_equal`, `gibbs_sweep`, `run_gibbs`. Checkerboard schedule. |
| `src/metrics.py` | done | Confusion matrix + accuracy, precision, recall, F1 (zero-safe). |
| `src/io_utils.py` | done | PNG, CSV, JSON writers. |
| `src/visualize.py` | done | Energy trace plot, multi-panel comparison plot. |
| `run_scenario.py` | done | CLI, plus automatic rebuild of `comparison.png` / `comparison.md`. |

All math is hand-written on top of NumPy. No `scipy`, no `scikit-*`, no PPL
libraries. The dependencies actually pulled in at runtime are exactly
`numpy`, `pillow`, `matplotlib`.

---

## 2. Findings during implementation

These are things that were **not** obvious from the spec alone and that future
maintainers should know.

### 2.1 The real dataset filenames differ from the spec example

The guide writes the CT file as `ct_layer32.tif`, but the file on disk is
`ctlayer_32.tif` (with the underscore between `r` and `32`). `config.py` was
updated to point at the real filename. If you swap in a different dataset,
remember to update `CT_FILE`.

### 2.2 The ink mask is a palette PNG with values {0, 1}, not {0, 255}

`load_ink_mask` initially used a threshold of `> 127` (assuming an 8-bit
grayscale mask). On this dataset that discarded every ink pixel. The loader
now binarises with `arr > 0`, which is correct for both palette PNGs (values
{0, 1}) and 8-bit grayscale PNGs (values {0, 255}). If you switch datasets,
sanity-check `np.unique(arr)` before trusting the loader.

### 2.3 The default crop (0, 0, 300, 300) is outside the papyrus

The full source images are roughly **8181 × 6330** pixels. The papyrus body
(and the labelled ink bounding box) starts around row 95, column 302. Cropping
at the origin landed in pure background — CT mean was `0.0` and the ink
fraction was `0.0`, so the per-class Gaussian estimator could not even run
because there were no ink pixels in the training half.

A small grid search picked a 300×300 ROI at **(top=1600, left=2200)** with:

- CT mean ≈ 0.39 (papyrus body is present)
- overall ink fraction ≈ 0.30
- training half (top 150 rows) ink fraction ≈ 0.44
- test half (bottom 150 rows) ink fraction ≈ 0.16

Both halves contain ink, which is required for the train/test split to be
meaningful. The crop is configurable in `src/config.py`.

### 2.4 The unary signal alone is weak

Maximum-likelihood per-class Gaussian parameters on the training half come out
as:

- `mu_ink ≈ 0.326`, `sigma_ink ≈ 0.154`
- `mu_noink ≈ 0.409`, `sigma_noink ≈ 0.157`

The class means differ by only about half a standard deviation, so the
pixel-level likelihood ratio is genuinely ambiguous for a large fraction of
the ROI. This is a property of the data, not a bug; do not expect the unary
classifier alone to be sharp.

### 2.5 The "energy must decrease" acceptance criterion can mask a stuck chain

`beta = 5.0` produces an energy trace that drops once and then stays flat at
`-930655.27` for every recorded sweep — the energy of the all-zeros
configuration. The Potts coupling is so strong relative to the unary
likelihood that the chain locks into the noink modal class and never escapes
within 700 sweeps. Sprint 5's "energy decreases visibly in the first 50
sweeps" check is satisfied, but the resulting marginals are degenerate (P =
0 everywhere). This is the *expected pathology* for the oversmoothed
scenario, not a bug, and matches what the assignment is trying to demonstrate.

### 2.6 Gibbs schedule choice

`gibbs_sweep` updates pixels in **checkerboard** order: pixels with `(i + j)`
even are updated together as one conditionally-independent group, then the
odd-parity pixels. The spec mentions this as the optimisation path when the
pure-Python random-permutation loop is too slow; given the scale of the
problem (300 × 300 grid, 700 sweeps × 3 scenarios) the checkerboard
implementation is essentially mandatory — it brings each scenario to ~8 s on
a laptop CPU, versus an estimated several minutes for a Python-level loop.

Both half-sweeps reuse a single `count_neighbors_equal` call against the
current state, so the public function from the spec is wired into the hot
path rather than being a dead stub.

### 2.7 `energy_history` is a 3-tuple, not a 2-tuple

The spec docstring says `list of (sweep_index, energy)` with "phase tagged in
the index". To keep the phase explicit and machine-readable, the actual
return value is

```python
[("burn", 10, -301034.7), ("burn", 20, ...), ..., ("sample", 1, ...), ...]
```

The CSV mirrors this with three columns: `phase, sweep, energy`. The energy
plot offsets sampling sweeps onto a single global x-axis and draws the
vertical line at the end of burn-in. If you write tooling against
`energy.csv`, expect three columns.

---

## 3. Warnings and known limitations

1. **The chain starts from all-zeros.** With moderate-to-high `beta`, this
   biases the burn-in trajectory toward the noink mode and can dominate the
   final marginals at the configured 200/500 sweep budget. A "warm-start"
   from the unary MAP would converge faster; the spec asked for all-zeros so
   that is what is implemented.
2. **`beta = 5.0` is degenerate.** See §2.5. The mask is all-zero and every
   F1/precision/recall is exactly 0. This is the expected illustration of
   over-smoothing; do not chase it as a bug.
3. **The Gaussian model is intentionally simple.** No log-prior, no per-pixel
   feature engineering. Adding a class prior `log π_k` to the unary term is
   the obvious first extension — at present the sampler implicitly assumes a
   uniform prior over `{ink, noink}` even though the dataset is roughly 70%
   noink.
4. **Test metrics look weak.** Test F1 sits around 0.20–0.24 in the `base`
   and `normal` scenarios. Most of the gap between train and test is because
   the test half is genuinely harder (only 16% ink versus 44% in training,
   and a different ink texture). It is not a code defect.
5. **`comparison.png` is rebuilt every run.** Each invocation of
   `run_scenario.py` rebuilds `results/comparison.png` and
   `results/comparison.md` from whatever `scenario_*` folders exist. If you
   delete one scenario folder, the next run drops it from the comparison.
6. **No automated test suite.** The acceptance criteria from section 7 of the
   guide are checked by inspection of the artefacts. A `pytest` layer is the
   natural next step.

---

## 4. Conclusions

- The skeleton is feature-complete against the guide. All ten expected
  artefacts per scenario are produced (six per `scenario_<tag>/` plus
  `comparison.png` and `comparison.md` at the `results/` root).
- The three scenarios reproduce the qualitative phenomena the assignment is
  meant to illustrate: salt-and-pepper noise at `beta = 0`, spatial
  coherence at `beta = 1.5`, and modal collapse at `beta = 5.0`.
- Runtime is ~8 s per scenario on a CPU, ~25 s total for the three.
- The most productive next steps are: warm-starting the chain from the unary
  MAP, adding a class prior, and running a small `pytest` suite that locks
  in the sprint-7 acceptance behaviour.

### 4.1 Test-region metrics (bottom half of the ROI)

| tag | beta | accuracy | precision | recall | F1 |
|-----|------|----------|-----------|--------|----|
| base | 0.0 | 0.522 | 0.160 | 0.455 | 0.236 |
| normal | 1.5 | 0.648 | 0.160 | 0.276 | 0.203 |
| oversmoothed | 5.0 | 0.838 | 0.000 | 0.000 | 0.000 |

Note the "oversmoothed" row: accuracy is misleadingly high because the all-zero
prediction matches the dominant class on the test half. F1 is the correct
metric to look at here.

---

## 5. Manual execution guide

This section reproduces the run from a clean checkout.

### 5.1 Prerequisites

- Windows with PowerShell (the project's reference platform).
- Python 3.10+ on `PATH`.
- A virtual environment in `.venv/` with `numpy`, `pillow`, `matplotlib`
  installed. The repo's `Setup_Windows.md` walks through this.
- The two raw inputs in place:
  - `data/raw/ctlayer_32.tif`
  - `data/raw/inklabels.png`

### 5.2 Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Bash (Git Bash / WSL):

```bash
source .venv/Scripts/activate
```

Confirm the right Python is active:

```powershell
python -c "import numpy, PIL, matplotlib; print(numpy.__version__, PIL.__version__, matplotlib.__version__)"
```

### 5.3 Run the three scenarios

Each command takes roughly 8 s and writes into `results/scenario_<tag>/`.

```powershell
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

After every run, `results/comparison.png` and `results/comparison.md` are
rebuilt from every `scenario_*` folder currently on disk.

Optional flags (defaults come from `src/config.py`):

```
--burn N               number of burn-in sweeps (default 200)
--samples N            number of sampling sweeps (default 500)
--seed N               RNG seed (default 42)
--energy-log-every N   record MRF energy every N sweeps (default 10)
```

For a quick smoke test before committing to a full run:

```powershell
python run_scenario.py --beta 1.5 --burn 20 --samples 30 --tag smoke
```

When you are done with the smoke run, delete `results/scenario_smoke/` so it
does not pollute the comparison artefacts:

```powershell
Remove-Item -Recurse -Force results\scenario_smoke
```

### 5.4 What each artefact contains

Per scenario, in `results/scenario_<tag>/`:

| File | What to look for |
|------|------------------|
| `final_mask.png` | Predicted binary ink mask (255 = ink, 0 = noink). Compare against `inklabels.png`. |
| `posterior.png` | Per-pixel marginal `P(HasInk = 1)` as a grayscale image. Soft edges = uncertain regions. |
| `energy.csv` | Three columns: `phase, sweep, energy`. Useful for replotting or convergence checks. |
| `energy.png` | Energy trace versus sweep index, with a red dashed line at end of burn-in. Should drop and then flatten. |
| `metrics.json` | Confusion-matrix counts and accuracy / precision / recall / F1 on three subsets: `full_roi`, `train`, `test`. |
| `params.json` | Hyperparameters used for that run, including the seed and the learned class Gaussian parameters. |

At the `results/` root after at least one run:

| File | What to look for |
|------|------------------|
| `comparison.png` | Side-by-side panels: CT slice, ground-truth mask, predicted mask per `beta`. |
| `comparison.md` | Table of test-region metrics across every scenario currently on disk. |

### 5.5 Inspecting results from the command line

```powershell
# Energy trace as plain text
Get-Content results\scenario_normal\energy.csv | Select-Object -First 5

# Pretty-print metrics
Get-Content results\scenario_normal\metrics.json | ConvertFrom-Json | Format-List

# Pretty-print params
Get-Content results\scenario_normal\params.json | ConvertFrom-Json | Format-List
```

Bash:

```bash
head results/scenario_normal/energy.csv
python -m json.tool results/scenario_normal/metrics.json
python -m json.tool results/scenario_normal/params.json
```

To open the PNGs from PowerShell:

```powershell
Invoke-Item results\comparison.png
Invoke-Item results\scenario_normal\energy.png
Invoke-Item results\scenario_normal\posterior.png
Invoke-Item results\scenario_normal\final_mask.png
```

### 5.6 What a healthy run looks like

- During execution, stdout prints `[load]`, `[crop]`, `[learn]`, and
  `[gibbs]` lines, ending in a metrics block for `full`, `train`, and
  `test`.
- Total wall-clock per scenario is in the single-digit-seconds range on a
  laptop CPU.
- The energy trace in `energy.png` is monotonically decreasing through
  burn-in, then flat (apart from sampling noise) during the sampling phase.
- The vertical red dashed line in the energy plot sits at the burn-in / sample
  boundary (sweep 200 with default settings).
- `final_mask.png` for `base` is visibly noisy, for `normal` is visibly
  cleaner, and for `oversmoothed` is solid black (modal collapse).

If any of these signs are absent, re-read §3 before assuming a bug.

### 5.7 Resetting between experiments

```powershell
Remove-Item -Recurse -Force results
```

Then re-run section 5.3. Reproducibility is governed entirely by `--seed`;
identical seeds and identical config produce bit-identical artefacts.
