#!/usr/bin/env python3
"""
Carrying capacity (K) sensitivity analysis.

Tests whether changing K from the default 100 to 10, 1000, or 10000
meaningfully changes the simulation output (distance matrices).

Design:
    1. Draw 50 parameter sets via Latin hypercube from the inference priors.
    2. For each set, run methdemon at K = 10, 100, 1000, 10000.
    3. Use the SAME RNG seed per parameter set across all K values.
    4. Compare inter-gland distance matrices via Frobenius norm relative
       to the K=100 baseline.

Three modes:
    generate  — Write parameter sets + seeds to a JSON file.
    run       — Run a single (param_index, K) simulation. Designed for
                Slurm array jobs: SLURM_ARRAY_TASK_ID indexes into the
                task list (n_sets * len(K_VALUES) tasks total).
    collect   — Gather all .npy results, compute statistics, plot.

Usage (local, from methabc root):
    python scripts/k_sensitivity.py generate -n 50
    python scripts/k_sensitivity.py run --local --cores 20
    python scripts/k_sensitivity.py collect

Usage (Slurm, from methabc root):
    python scripts/k_sensitivity.py generate -n 50
    sbatch scripts/k_sensitivity_slurm.sh   # submits array job
    # ... wait for completion ...
    python scripts/k_sensitivity.py collect

Output:
    outputs/k_sensitivity/params.json         (generate)
    outputs/k_sensitivity/matrices/*.npy      (run)
    outputs/k_sensitivity/k_sensitivity.json  (collect)
    outputs/k_sensitivity/k_sensitivity.pdf   (collect)
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────

K_VALUES = [10, 100, 1000, 10000]
K_BASELINE = 100

LOG_PRIORS = {
    'meth_rate': (-4, -2),
    'demeth_rate': (-4, -2),
    'init_migration_rate': (-3.3, -1),
    'mu_driver_birth': (-5, -2),
    's_driver_birth': (0, 0.2),
}

MODEL_PATH = "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = "resources/config_template.dat"
OUTPUT_DIR = Path("outputs/k_sensitivity")
MATRIX_DIR = OUTPUT_DIR / "matrices"


# ── Simulation ──────────────────────────────────────────────────────

_TEMPLATE_LINES = None


def read_template():
    with open(CONFIG_TEMPLATE, 'r') as f:
        return f.readlines()


def run_single_sim(params, K, seed):
    """Run one simulation. Returns 8x8 distance matrix or None."""
    tmpdir = tempfile.mkdtemp(prefix=f'ksens_K{K}_')
    try:
        lines = list(_TEMPLATE_LINES)
        full = {}
        for k, v in params.items():
            if k == 's_driver_birth':
                full[k] = v
            else:
                full[k] = 10 ** v
        full['deme_carrying_capacity'] = K
        full['seed'] = seed
        full['left_demes'] = 4
        full['right_demes'] = 4

        updated = []
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in full:
                val = full[parts[0]]
                if isinstance(val, (int, np.integer)):
                    updated.append(f"    {parts[0]} {val}\n")
                else:
                    updated.append(f"    {parts[0]} {val:.10f}\n")
            else:
                updated.append(line)

        config_path = os.path.join(tmpdir, 'config.dat')
        with open(config_path, 'w') as f:
            f.writelines(updated)

        result = subprocess.run(
            [MODEL_PATH, tmpdir, 'config.dat'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=1800,
        )
        if result.returncode != 0:
            return None

        data_path = os.path.join(tmpdir, 'final_demes.csv')
        if not os.path.exists(data_path):
            return None

        df = pd.read_csv(data_path, header=0)
        df.AverageArray = df.AverageArray.apply(
            lambda x: np.fromstring(x, sep=";"))
        df = df[np.floor(df.Generation) == np.floor(df.Generation.max())]
        df = df.reset_index(drop=True)
        if len(df) > 8:
            df = df.iloc[:8]
        if len(df) != 8:
            return None

        return compute_distance_matrix(df)

    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_distance_matrix(df):
    df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    n = len(df)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = df.iloc[i].AverageArray - df.iloc[j].AverageArray
            mat[i, j] = np.sum(diff ** 2) / len(diff)
    return mat


# ── Worker for local Pool ──────────────────────────────────────────

def _pool_worker(args):
    idx, K, params, seed = args
    mat = run_single_sim(params, K, seed)
    if mat is not None:
        out = MATRIX_DIR / f"mat_{idx:03d}_K{K}.npy"
        np.save(out, mat)
        return True
    return False


def _init_pool(template_lines):
    global _TEMPLATE_LINES
    _TEMPLATE_LINES = template_lines


# ── Commands ────────────────────────────────────────────────────────

def cmd_generate(args):
    """Generate LHS parameter sets and save to JSON."""
    from scipy.stats import qmc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    n = args.n_sets
    param_names = list(LOG_PRIORS.keys())
    n_params = len(param_names)

    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    unit_samples = sampler.random(n)
    lows = np.array([LOG_PRIORS[p][0] for p in param_names])
    highs = np.array([LOG_PRIORS[p][1] for p in param_names])
    scaled = qmc.scale(unit_samples, lows, highs)

    rng = np.random.default_rng(123)
    seeds = rng.integers(1, 2**15 - 1, size=n).tolist()

    param_sets = []
    for i in range(n):
        param_sets.append({p: float(scaled[i, j])
                           for j, p in enumerate(param_names)})

    config = {
        'n_sets': n,
        'K_values': K_VALUES,
        'param_names': param_names,
        'param_sets': param_sets,
        'seeds': seeds,
        'n_tasks': n * len(K_VALUES),
    }

    out_path = OUTPUT_DIR / "params.json"
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Generated {n} parameter sets × {len(K_VALUES)} K values "
          f"= {config['n_tasks']} tasks")
    print(f"Saved to {out_path}")
    print(f"\nFor Slurm: sbatch --array=0-{config['n_tasks']-1} "
          f"scripts/k_sensitivity_slurm.sh")


def cmd_run(args):
    """Run simulations — either a single Slurm task or all locally."""
    global _TEMPLATE_LINES
    _TEMPLATE_LINES = read_template()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run from methabc root.")
        sys.exit(1)

    params_path = OUTPUT_DIR / "params.json"
    if not params_path.exists():
        print("ERROR: params.json not found. Run 'generate' first.")
        sys.exit(1)

    with open(params_path) as f:
        config = json.load(f)

    param_sets = config['param_sets']
    seeds = config['seeds']
    n_sets = config['n_sets']

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    # Build task list: (param_index, K)
    tasks = []
    for i in range(n_sets):
        for K in K_VALUES:
            tasks.append((i, K))

    if args.local:
        # Run all tasks locally with multiprocessing
        print(f"Running {len(tasks)} tasks on {args.cores} cores...")
        pool_args = [(i, K, param_sets[i], seeds[i]) for i, K in tasks]

        t0 = time.time()
        with Pool(processes=args.cores, initializer=_init_pool,
                  initargs=(_TEMPLATE_LINES,)) as pool:
            results = pool.map(_pool_worker, pool_args)

        n_ok = sum(results)
        elapsed = time.time() - t0
        print(f"Done: {n_ok}/{len(tasks)} successful, {elapsed:.0f}s")

    else:
        # Single task from SLURM_ARRAY_TASK_ID
        task_id = args.task_id
        if task_id is None:
            task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if task_id is None:
            print("ERROR: No --task-id and no SLURM_ARRAY_TASK_ID set.")
            print("Use --local for local runs, or submit via Slurm.")
            sys.exit(1)

        task_id = int(task_id)
        if task_id >= len(tasks):
            print(f"Task {task_id} out of range (max {len(tasks)-1})")
            sys.exit(1)

        i, K = tasks[task_id]
        print(f"Task {task_id}: param_set={i}, K={K}")

        mat = run_single_sim(param_sets[i], K, seeds[i])
        if mat is not None:
            out = MATRIX_DIR / f"mat_{i:03d}_K{K}.npy"
            np.save(out, mat)
            print(f"  Saved: {out}")
        else:
            print(f"  FAILED (no output)")


def cmd_collect(args):
    """Gather results, compute stats, plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    params_path = OUTPUT_DIR / "params.json"
    with open(params_path) as f:
        config = json.load(f)

    n_sets = config['n_sets']

    # Load all matrices
    results = {K: {} for K in K_VALUES}
    for i in range(n_sets):
        for K in K_VALUES:
            path = MATRIX_DIR / f"mat_{i:03d}_K{K}.npy"
            if path.exists():
                results[K][i] = np.load(path)

    # Find indices where ALL K values succeeded
    valid = []
    for i in range(n_sets):
        if all(i in results[K] for K in K_VALUES):
            valid.append(i)

    print(f"Valid parameter sets (all K succeeded): {len(valid)}/{n_sets}")
    for K in K_VALUES:
        print(f"  K={K:>5d}: {len(results[K])}/{n_sets} completed")

    if len(valid) < 3:
        print("Not enough valid sets to analyse.")
        sys.exit(1)

    # Frobenius distances relative to K=100
    frob = {K: [] for K in K_VALUES if K != K_BASELINE}
    for K in frob:
        for i in valid:
            d = np.sqrt(np.sum((results[K_BASELINE][i] - results[K][i]) ** 2))
            frob[K].append(d)

    print(f"\n{'K':>6s}  {'mean':>8s}  {'median':>8s}  {'std':>8s}")
    summary = {'n_sets': n_sets, 'n_valid': len(valid), 'comparisons': {}}
    for K in sorted(frob.keys()):
        d = frob[K]
        m, med, s = np.mean(d), np.median(d), np.std(d)
        print(f"{K:>6d}  {m:8.4f}  {med:8.4f}  {s:8.4f}")
        summary['comparisons'][str(K)] = {
            'mean_frobenius': float(m), 'median_frobenius': float(med),
            'std_frobenius': float(s), 'n': len(d),
        }

    # Mean inter-gland divergence per K
    mean_div = {K: [] for K in K_VALUES}
    for K in K_VALUES:
        for i in valid:
            mean_div[K].append(np.mean(results[K][i]))

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    plot_data = [frob[K] for K in sorted(frob.keys())]
    plot_labels = [f'K={K}' for K in sorted(frob.keys())]
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Frobenius distance to K=100 baseline')
    ax.set_title('Distance matrix sensitivity to K')

    ax = axes[1]
    positions = range(len(K_VALUES))
    for pos, K in zip(positions, K_VALUES):
        ax.boxplot([mean_div[K]], positions=[pos], widths=0.6,
                   patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f'K={K}' for K in K_VALUES])
    ax.set_ylabel('Mean distance matrix entry')
    ax.set_title('Inter-gland divergence vs K')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'k_sensitivity.pdf'
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nFigure: {fig_path}")

    json_path = OUTPUT_DIR / 'k_sensitivity.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results: {json_path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description='K sensitivity analysis')
    sub = parser.add_subparsers(dest='command')

    gen = sub.add_parser('generate', help='Generate LHS parameter sets')
    gen.add_argument('-n', '--n_sets', type=int, default=50)

    run = sub.add_parser('run', help='Run simulations')
    run.add_argument('--local', action='store_true',
                     help='Run all tasks locally with multiprocessing')
    run.add_argument('--cores', type=int, default=20)
    run.add_argument('--task-id', type=int, default=None,
                     help='Single task index (overrides SLURM_ARRAY_TASK_ID)')

    sub.add_parser('collect', help='Gather results and plot')

    args = parser.parse_args()

    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'collect':
        cmd_collect(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
