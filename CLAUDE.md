# Project Context for Claude Code

## What this is

VoxelScribe is an ink detection system for carbonized papyrus fragments from the Vesuvius Challenge. It implements a Relational Probability Model (RPM) with Gibbs Sampling over a 2D Markov Random Field, applied to CT scan imagery.

Academic project for the course **Modelos Estocásticos**, Universidad Nacional de Colombia. Corresponds to Chapter 18 (Probabilistic Programming) of Russell & Norvig's *AIMA*, 4th ed.

## Full specification

The authoritative implementation spec lives in `docs/Implementation_Guide.md`. Read it before writing any code. This file is only the briefing.

## Critical constraints (non-negotiable)

**Allowed:**

- Python 3.10+
- Standard library
- `numpy` (the single permitted data-structures library)
- `pillow` (only to decode TIF and PNG)
- `matplotlib` (only for plotting)

**Forbidden:**

- `scipy` and any of its submodules
- `scikit-learn`, `scikit-image`
- `pymc`, `stan`, `pgmpy`, `pyro`, `numpyro`
- Any library that implements MCMC, Gaussian PDFs, Bayesian networks, MRFs, or classification metrics

All mathematics must be implemented from scratch with NumPy. If a function feels "too convenient to find in scipy", that is exactly the one to write by hand.

## Code style

- Code in English; comments in English.
- No emojis anywhere in code.
- Reproducibility through `np.random.default_rng(SEED)`, passed as parameter rather than created locally.
- Work in log-space throughout the sampler; apply `np.exp` only at normalization.
- Prefer explicit errors (`raise ValueError`, `raise RuntimeError`) over `assert` for control flow.
- Skeleton over polish: focus on a correct end-to-end pipeline before any optimization.

## How to run

After implementing, the three experimental scenarios run as:

```
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

Outputs land in `results/scenario_<tag>/`.

## Sprint order

Follow section 7 of `docs/Implementation_Guide.md` in order. Do not skip ahead. Each sprint has its own acceptance criterion; satisfy it before moving on.

## Project files

| File | Role |
|------|------|
| `docs/Implementation_Guide.md` | Full technical specification |
| `docs/implementation_summary.md` | Post-implementation notes, warnings, run guide |
| `Setup_Windows.md` | Environment setup reference |
| `README.md` | Human-facing overview |
| `requirements.txt` | Pinned dependency ranges |
| `src/` | All implementation modules |
| `run_scenario.py` | CLI entry point |
| `data/raw/` | Input TIF and PNG (already in place) |

## Common pitfalls to avoid

- Do not pad image borders artificially when counting 4-neighbors. Edge pixels naturally have fewer neighbors.
- Do not compute raw Gaussian PDFs in linear space; they underflow for typical CT intensities.
- Do not reseed the RNG inside loops.
- Do not overwrite existing files in `results/` silently; create them under tagged subfolders.
