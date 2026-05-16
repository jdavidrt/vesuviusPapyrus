"""CLI entry point: run one scenario end-to-end and rebuild comparisons."""

from __future__ import annotations

import argparse
import json
import os
import time
from glob import glob

import numpy as np

from src import config
from src.data_loader import crop_region, load_ct_image, load_ink_mask, spatial_split
from src.gibbs import run_gibbs
from src.io_utils import (
    ensure_dir,
    save_energy_csv,
    save_json,
    save_mask_png,
    save_posterior_png,
)
from src.learning import estimate_class_params
from src.metrics import compute_metrics
from src.visualize import plot_comparison, plot_energy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoxelScribe Gibbs scenario runner.")
    p.add_argument("--beta", type=float, required=True, help="Potts coupling strength.")
    p.add_argument("--burn", type=int, default=config.N_BURN, help="Burn-in sweeps.")
    p.add_argument("--samples", type=int, default=config.N_SAMPLE, help="Sampling sweeps.")
    p.add_argument("--seed", type=int, default=config.SEED, help="RNG seed.")
    p.add_argument("--tag", type=str, required=True, help="Subfolder tag under results/.")
    p.add_argument(
        "--energy-log-every",
        type=int,
        default=config.ENERGY_LOG_EVERY,
        help="Log MRF energy every N sweeps.",
    )
    return p.parse_args()


def _print_metric_block(title: str, m: dict) -> None:
    print(
        f"  {title:<8} acc={m['accuracy']:.4f} "
        f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
        f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})"
    )


def _rebuild_comparison(results_dir: str) -> None:
    """Aggregate every scenario_* subfolder into comparison.png and comparison.md."""
    ct_path = os.path.join(config.DATA_DIR, config.CT_FILE)
    ink_path = os.path.join(config.DATA_DIR, config.INK_FILE)
    ct_full = load_ct_image(ct_path)
    ink_full = load_ink_mask(ink_path)
    ct = crop_region(ct_full, config.CROP_TOP, config.CROP_LEFT, config.CROP_H, config.CROP_W)
    ink = crop_region(ink_full, config.CROP_TOP, config.CROP_LEFT, config.CROP_H, config.CROP_W)

    scenario_dirs = sorted(glob(os.path.join(results_dir, "scenario_*")))
    predictions: dict[str, np.ndarray] = {}
    rows: list[tuple[str, float, dict]] = []
    for sdir in scenario_dirs:
        params_path = os.path.join(sdir, "params.json")
        metrics_path = os.path.join(sdir, "metrics.json")
        mask_path = os.path.join(sdir, "final_mask.png")
        if not (os.path.isfile(params_path) and os.path.isfile(metrics_path) and os.path.isfile(mask_path)):
            continue
        with open(params_path, encoding="utf-8") as fh:
            scen_params = json.load(fh)
        with open(metrics_path, encoding="utf-8") as fh:
            scen_metrics = json.load(fh)
        beta = float(scen_params.get("beta", float("nan")))
        tag = scen_params.get("tag", os.path.basename(sdir).replace("scenario_", ""))
        # Load mask back as binary array for plotting.
        from PIL import Image
        mask = (np.asarray(Image.open(mask_path)) > 127).astype(np.uint8)
        predictions[f"{beta:.1f}"] = mask
        rows.append((tag, beta, scen_metrics.get("test", {})))

    if predictions:
        plot_comparison(
            ct, ink, predictions, os.path.join(results_dir, "comparison.png")
        )

    md_path = os.path.join(results_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Scenario comparison\n\n")
        fh.write("Metrics computed on the held-out test region (bottom half of the ROI).\n\n")
        fh.write("| tag | beta | accuracy | precision | recall | F1 |\n")
        fh.write("|-----|------|----------|-----------|--------|----|\n")
        for tag, beta, m in sorted(rows, key=lambda r: r[1]):
            fh.write(
                f"| {tag} | {beta:.2f} | "
                f"{m.get('accuracy', 0.0):.4f} | "
                f"{m.get('precision', 0.0):.4f} | "
                f"{m.get('recall', 0.0):.4f} | "
                f"{m.get('f1', 0.0):.4f} |\n"
            )


def main() -> None:
    args = _parse_args()
    t_start = time.time()

    # 1-2. Load CT image and ink mask.
    ct_path = os.path.join(config.DATA_DIR, config.CT_FILE)
    ink_path = os.path.join(config.DATA_DIR, config.INK_FILE)
    ct_full = load_ct_image(ct_path)
    ink_full = load_ink_mask(ink_path)
    print(f"[load] CT shape={ct_full.shape} dtype={ct_full.dtype}")
    print(f"[load] ink shape={ink_full.shape} dtype={ink_full.dtype}")

    # 3. Crop to the configured ROI.
    ct = crop_region(ct_full, config.CROP_TOP, config.CROP_LEFT, config.CROP_H, config.CROP_W)
    ink = crop_region(ink_full, config.CROP_TOP, config.CROP_LEFT, config.CROP_H, config.CROP_W)
    print(f"[crop] ROI shape={ct.shape}, ink positive fraction={float(ink.mean()):.3f}")

    # 4. Spatial split: top half trains, bottom half tests.
    train_rows, test_rows = spatial_split(config.CROP_H)

    # 5. Learn per-class Gaussian parameters on the training rows only.
    params = estimate_class_params(ct[train_rows], ink[train_rows])
    print(
        "[learn] mu_ink={mu_ink:.4f} sigma_ink={sigma_ink:.4f} "
        "mu_noink={mu_noink:.4f} sigma_noink={sigma_noink:.4f}".format(**params)
    )

    # 6-7. Run Gibbs (run_gibbs internally precomputes log unaries).
    result = run_gibbs(
        image=ct,
        params=params,
        beta=args.beta,
        n_burn=args.burn,
        n_sample=args.samples,
        seed=args.seed,
        energy_log_every=args.energy_log_every,
    )
    posterior = result["posterior"]
    final_mask = result["final_mask"]
    history = result["energy_history"]

    # 8. Metrics on the full ROI, the training rows, and the test rows.
    metrics_full = compute_metrics(ink, final_mask)
    metrics_train = compute_metrics(ink[train_rows], final_mask[train_rows])
    metrics_test = compute_metrics(ink[test_rows], final_mask[test_rows])

    # 9. Persist artefacts.
    out_dir = os.path.join(config.RESULTS_DIR, f"scenario_{args.tag}")
    ensure_dir(out_dir)
    save_mask_png(final_mask, os.path.join(out_dir, "final_mask.png"))
    save_posterior_png(posterior, os.path.join(out_dir, "posterior.png"))
    save_energy_csv(history, os.path.join(out_dir, "energy.csv"))
    plot_energy(history, os.path.join(out_dir, "energy.png"))
    save_json(
        {
            "full_roi": metrics_full,
            "train": metrics_train,
            "test": metrics_test,
        },
        os.path.join(out_dir, "metrics.json"),
    )
    save_json(
        {
            "beta": args.beta,
            "burn": args.burn,
            "samples": args.samples,
            "seed": args.seed,
            "tag": args.tag,
            "energy_log_every": args.energy_log_every,
            "crop": {
                "top": config.CROP_TOP,
                "left": config.CROP_LEFT,
                "h": config.CROP_H,
                "w": config.CROP_W,
            },
            "ct_file": config.CT_FILE,
            "ink_file": config.INK_FILE,
            "class_params": params,
        },
        os.path.join(out_dir, "params.json"),
    )

    # Rebuild comparison artefacts from whatever scenarios exist.
    _rebuild_comparison(config.RESULTS_DIR)

    # 10. Stdout summary.
    elapsed = time.time() - t_start
    print(f"\n[done] scenario={args.tag} beta={args.beta} in {elapsed:.1f}s")
    print(f"[done] artefacts in {out_dir}")
    _print_metric_block("full", metrics_full)
    _print_metric_block("train", metrics_train)
    _print_metric_block("test", metrics_test)


if __name__ == "__main__":
    main()
