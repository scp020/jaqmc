#!/usr/bin/env python3
"""
Plot original-vs-broken training curves.

Run:

    python scripts/taskb_plot_curves.py \
      --original runs/taskb_li_original/train_stats.csv \
      --broken runs/taskb_li_broken/train_stats.csv \
      --outdir runs/taskb_plots
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find {label} column. Available columns are:\n{list(df.columns)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="runs/taskb_li_original/train_stats.csv")
    parser.add_argument("--broken", default="runs/taskb_li_broken/train_stats.csv")
    parser.add_argument("--outdir", default="runs/taskb_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_o = pd.read_csv(args.original)
    df_b = pd.read_csv(args.broken)

    step_candidates = ["step", "Step", "iteration", "iter"]
    energy_candidates = [
        "total_energy",
        "energy",
        "energy:total",
        "energy_total",
        "mean_energy",
    ]
    variance_candidates = [
        "total_energy_var",
        "variance",
        "energy:total_var",
        "energy_var",
        "var",
    ]

    # Some CSVs may not explicitly store step.
    try:
        step_o = find_column(df_o, step_candidates, "step")
        x_o = df_o[step_o]
    except ValueError:
        x_o = df_o.index

    try:
        step_b = find_column(df_b, step_candidates, "step")
        x_b = df_b[step_b]
    except ValueError:
        x_b = df_b.index

    energy_o = find_column(df_o, energy_candidates, "energy")
    energy_b = find_column(df_b, energy_candidates, "energy")

    plt.figure(figsize=(8, 5))
    plt.plot(x_o, df_o[energy_o], label="original FermiNet")
    plt.plot(x_b, df_b[energy_b], label="broken antisymmetry")
    plt.xlabel("Step")
    plt.ylabel("Energy / Ha")
    plt.title("Training Energy")
    plt.legend()
    plt.tight_layout()
    energy_path = outdir / "energy_comparison.png"
    plt.savefig(energy_path, dpi=200)
    print(f"Wrote {energy_path}")

    # Variance column names may differ. Plot only if both can be found.
    try:
        var_o = find_column(df_o, variance_candidates, "variance")
        var_b = find_column(df_b, variance_candidates, "variance")

        plt.figure(figsize=(8, 5))
        plt.plot(x_o, df_o[var_o], label="original FermiNet")
        plt.plot(x_b, df_b[var_b], label="broken antisymmetry")
        plt.xlabel("Step")
        plt.ylabel("Variance")
        plt.title("Training Variance")
        plt.legend()
        plt.tight_layout()
        var_path = outdir / "variance_comparison.png"
        plt.savefig(var_path, dpi=200)
        print(f"Wrote {var_path}")
    except ValueError as e:
        print("Variance plot skipped.")
        print(e)


if __name__ == "__main__":
    main()
