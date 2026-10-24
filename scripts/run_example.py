#!/usr/bin/env python
"""Run a released model on an example chip (CPU).

Usage:
  python scripts/run_example.py --weights weights/prithvi_openet_tower_opt_f_e \
      --example examples/conus_T13TGE_2020-07.npz --out example_output
  python scripts/run_example.py --weights ... --example ... --selftest
      (compares the prediction with the one stored in the example; max |diff| < 1e-3 mm)

The example stores the integer values of all Opt.+F+E channels (x_int16, and inputs_raw in physical units for
inspection); each model selects its own channels by name (Opt.: 6, Opt.+F: 9, Opt.+F+E: 14) and normalises them with
the statistics in its config.json. Outputs: <id>_et_members.npy, <id>_et_mean.npy and a PNG quick-look.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prithvi_openet import example_inputs, load_model, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", default="example_output")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    wdir = Path(a.weights)
    model, cfg = load_model(wdir)
    ex = np.load(a.example, allow_pickle=False)
    inputs = example_inputs(ex, cfg)
    pred = predict(model, inputs)
    et = np.where(inputs["valid_mask"], pred["et_mean"], np.nan)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.example).stem
    np.save(out / f"{stem}_et_members.npy", pred["et_members"])
    np.save(out / f"{stem}_et_mean.npy", et)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(et, cmap="viridis")
        fig.colorbar(im, ax=ax, label="ET (mm month$^{-1}$)")
        ax.set_title(f"{cfg.get('display_name', cfg['name'])}  {stem}", fontsize=8)
        ax.axis("off")
        fig.savefig(out / f"{stem}_et_mean.png", dpi=150, bbox_inches="tight")
    except ImportError:
        pass
    print(f"{stem}: mean ET {np.nanmean(et):.1f} mm/month over {np.isfinite(et).sum()} valid pixels")

    if a.selftest:
        key = f"pred_{wdir.name}_mean"
        if key not in ex:
            sys.exit(f"selftest: {key} not stored in the example")
        diff = np.nanmax(np.abs(np.where(inputs["valid_mask"], pred["et_mean"] - ex[key], np.nan)))
        print(f"selftest max |diff| = {diff:.2e} mm")
        sys.exit(0 if diff < 1e-3 else 1)


if __name__ == "__main__":
    main()
