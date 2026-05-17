#!/usr/bin/env python3
"""
Task B: Check FermiNet antisymmetry.

Run from JaQMC repository root:

    python scripts/taskb_check_antisymmetry.py --fresh

Then, after breaking LogDet, run:

    python scripts/taskb_check_antisymmetry.py --no-train

Expected original FermiNet behavior:
    exchange two same-spin electrons:
        log|psi| almost unchanged
        sign flips

That is:
    psi(r1, r2, r3) = - psi(r2, r1, r3)
"""

import argparse
import dataclasses
import shutil
from pathlib import Path

import jax
from jax import numpy as jnp

from jaqmc.app.molecule import MoleculeTrainWorkflow
from jaqmc.utils.config import ConfigManager


def make_cfg(save_path: Path, batch_size: int, pretrain_steps: int, train_steps: int):
    return ConfigManager(
        {
            "system": {
                "atoms": [
                    {"symbol": "Li", "coords": [0.0, 0.0, 0.0]},
                ],
                # First two electrons are spin-up; the third is spin-down.
                # We will exchange electrons 0 and 1.
                "electron_spins": [2, 1],
            },
            "workflow": {
                "batch_size": batch_size,
                "save_path": str(save_path),
            },
            # Small network for a quick task-B test.
            # Antisymmetry is structural; it does not require long training.
            "wf": {
                "module": "ferminet",
                "hidden_dims_single": [32, 32],
                "hidden_dims_double": [8, 8],
                "ndets": 4,
            },
            "pretrain": {
                "run": {"iterations": pretrain_steps},
            },
            "train": {
                "run": {"iterations": train_steps},
            },
        }
    )


def to_float(x) -> float:
    return float(jax.device_get(x))


def one_walker_from_batch(batched_data, i: int):
    """Extract a single walker Data object from BatchedData."""
    return dataclasses.replace(
        batched_data.data,
        **{k: batched_data.data[k][i] for k in batched_data.fields_with_batch},
    )


def swap_first_two_same_spin_electrons(data):
    """Swap electron 0 and electron 1."""
    e = data.electrons
    swapped = e.at[0].set(e[1])
    swapped = swapped.at[1].set(e[0])
    return data.merge({"electrons": swapped})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="runs/taskb_li_original")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--num-checks", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fresh", action="store_true", help="Delete old workdir and rerun training.")
    parser.add_argument("--no-train", action="store_true", help="Only restore existing checkpoint and test.")
    args = parser.parse_args()

    workdir = Path(args.workdir)

    if args.fresh and workdir.exists():
        shutil.rmtree(workdir)

    cfg = make_cfg(
        save_path=workdir,
        batch_size=args.batch_size,
        pretrain_steps=args.pretrain_steps,
        train_steps=args.train_steps,
    )

    workflow = MoleculeTrainWorkflow(cfg)

    if not args.no_train:
        print(f"[1/3] Running a short Li/FermiNet computation in {workdir} ...")
        workflow()
    else:
        print(f"[1/3] Skipping training. Restoring from {workdir} ...")

    print("[2/3] Restoring checkpoint ...")
    state = workflow.restore_checkpoint(workdir)
    wf = workflow.train_stage.wavefunction
    params = state.params
    batched_data = state.batched_data

    print("[3/3] Checking same-spin electron exchange antisymmetry ...")
    print()
    print("i | logpsi(original) | sign(original) | logpsi(swapped) | sign(swapped) | |Δlogpsi| | sign product | verdict")
    print("--|------------------|----------------|-----------------|---------------|----------|--------------|--------")

    passed = 0
    checked = 0

    for i in range(min(args.num_checks, batched_data.batch_size)):
        data = one_walker_from_batch(batched_data, i)
        data_swapped = swap_first_two_same_spin_electrons(data)

        out0 = wf.apply(params, data)
        out1 = wf.apply(params, data_swapped)

        log0 = to_float(out0["logpsi"])
        log1 = to_float(out1["logpsi"])
        sign0 = to_float(out0["sign_logpsi"])
        sign1 = to_float(out1["sign_logpsi"])

        dlog = abs(log0 - log1)
        sign_product = sign0 * sign1

        if sign0 == 0 or sign1 == 0:
            verdict = "SKIP: near node"
        elif dlog < args.tol and sign_product < 0:
            verdict = "PASS"
            passed += 1
            checked += 1
        else:
            verdict = "FAIL"
            checked += 1

        print(
            f"{i:1d} | {log0:16.8f} | {sign0:14.0f} | "
            f"{log1:15.8f} | {sign1:13.0f} | {dlog:8.2e} | "
            f"{sign_product:12.0f} | {verdict}"
        )

    print()
    if checked == 0:
        print("No valid non-node samples were checked. Rerun with a different checkpoint.")
    elif passed == checked:
        print(f"RESULT: PASS. {passed}/{checked} valid samples satisfy antisymmetry.")
    else:
        print(f"RESULT: FAIL. {passed}/{checked} valid samples satisfy antisymmetry.")


if __name__ == "__main__":
    main()
