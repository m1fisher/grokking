#!/usr/bin/env python3
"""Ablation sweep: primes × noise levels × activations for MLP.

Intelligent scheduling: sorts jobs by estimated speed (small primes first),
limits to MAX_PER_GPU concurrent jobs per GPU, and backfills as slots open.

Usage: .venv/bin/python run_sweep.py
"""

import itertools
import subprocess
import time

PRIMES = [11, 31, 59, 97]
NOISES = [0.0, 0.1, 0.25, 0.5]
ACTIVATIONS = ["relu", "silu", "quadratic"]
MLP_GPUS = [0, 3, 4, 5, 7]
MAX_PER_GPU = 2

MLP_EPOCHS = 200_000
MLP_LOG_EVERY = 500
MLP_EARLY_STOP = 6
OUT_DIR = "results/sweep2"


def launch(cmd, tag):
    print(f"  [{tag}] launching", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PYTHONUNBUFFERED": "1", "PATH": ".venv/bin:/usr/bin:/bin"},
    )


def gpu_running(active, gpu):
    """Count how many active jobs are on a given GPU."""
    return sum(1 for _, _, g in active if g == gpu)


def run_mlp_sweep():
    # Build job list sorted by prime (small = fast first)
    jobs = sorted(
        itertools.product(PRIMES, NOISES, ACTIVATIONS),
        key=lambda x: x[0],
    )
    total = len(jobs)
    max_concurrent = MAX_PER_GPU * len(MLP_GPUS)
    print(f"MLP sweep: {total} runs, max {max_concurrent} concurrent "
          f"({MAX_PER_GPU}/GPU) across GPUs {MLP_GPUS}")
    print(f"  {MLP_EPOCHS} epochs, log every {MLP_LOG_EVERY}, "
          f"early stop {MLP_EARLY_STOP}\n")

    active = []  # list of (tag, Popen, gpu)
    completed = 0
    job_idx = 0

    while job_idx < total or active:
        # Reap finished processes
        still_active = []
        for tag, proc, gpu in active:
            if proc.poll() is not None:
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED ({rc})"
                completed += 1
                print(f"  [{tag}] {status} ({completed}/{total} done)", flush=True)
            else:
                still_active.append((tag, proc, gpu))
        active = still_active

        # Launch new jobs if slots available
        launched_any = False
        while job_idx < total:
            # Find a GPU with capacity
            gpu = None
            for g in MLP_GPUS:
                if gpu_running(active, g) < MAX_PER_GPU:
                    gpu = g
                    break
            if gpu is None:
                break  # all GPUs full

            p, noise, act = jobs[job_idx]
            job_idx += 1
            tag = f"mlp_{act}_p{p}_n{noise}"
            outdir = f"{OUT_DIR}/{tag}"
            cmd = [
                ".venv/bin/python", "-m", "grokking.main",
                "--model", "mlp",
                "--p", str(p),
                "--operation", "addition",
                "--data-frac", "0.5",
                "--noise-level", str(noise),
                "--activation", act,
                "--depth", "1",
                "--hidden", "500",
                "--optimizer", "sgd",
                "--lr", "50",
                "--wd", "0",
                "--loss", "mse",
                "--batch-size", "128",
                "--epochs", str(MLP_EPOCHS),
                "--log-every", str(MLP_LOG_EVERY),
                "--early-stop", str(MLP_EARLY_STOP),
                "--seed", "1",
                "--pair-seed", "420",
                "--device", f"cuda:{gpu}",
                "--out-dir", outdir,
            ]
            proc = launch(cmd, tag)
            active.append((tag, proc, gpu))
            launched_any = True
            time.sleep(0.2)

        if not launched_any and active:
            time.sleep(3)

    print(f"\nAll {total} runs complete.")


if __name__ == "__main__":
    print("=" * 60)
    print("GROKKING MLP SWEEP (200k epochs, intelligent scheduling)")
    print("=" * 60)
    print()
    run_mlp_sweep()
