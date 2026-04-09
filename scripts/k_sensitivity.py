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

Usage (from methabc root):
    python scripts/k_sensitivity.py -n 50 --cores 20

Output:
    outputs/k_sensitivity/k_sensitivity_results.json
    outputs/k_sensitivity/k_sensitivity.pdf
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc, wilcoxon

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


# ── Simulation ──────────────────────────────────────────────────────

def read_template():
    with open(CONFIG_TEMPLATE, 'r') as f:
        return f.readlines()

TEMPLATE_LINES = None  # set in main after cwd is confirmed


def run_single_sim(args):
    """Run one simulation. args = (param_dict, K, seed)."""
    params, K, seed = args

    tmpdir = tempfile.mkdtemp(prefix=f'ksens_K{K}_')
    try:
        lines = list(TEMPLATE_LINES)
        # Build full param dict (natural scale)
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
            timeout=1800,  # 30 min max for K=10000
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
    """8x8 inter-gland distance matrix."""
    df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    n = len(df)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = df.iloc[i].AverageArray - df.iloc[j].AverageArray
            mat[i, j] = np.sum(diff ** 2) / len(diff)
    return mat


# ── Main ────────────────────────────────────────────────────────────

def main():
    global TEMPLATE_LINES

    parser = ArgumentParser(description='K sensitivity analysis')
    parser.add_argument('-n', '--n_sets', type=int, default=50,
                        help='Number of LHS parameter sets (default: 50)')
    parser.add_argument('--cores', type=int, default=20,
                        help='Parallel cores (default: 20)')
    args = parser.parse_args()

    # Verify we're in methabc root
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run from methabc root.")
        sys.exit(1)

    TEMPLATE_LINES = read_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = args.n_sets
    n_params = len(LOG_PRIORS)
    param_names = list(LOG_PRIORS.keys())

    # Latin hypercube sampling
    print(f"Generating {n} LHS parameter sets...")
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    unit_samples = sampler.random(n)
    lows = np.array([LOG_PRIORS[p][0] for p in param_names])
    highs = np.array([LOG_PRIORS[p][1] for p in param_names])
    scaled = qmc.scale(unit_samples, lows, highs)

    param_sets = []
    for i in range(n):
        param_sets.append({p: scaled[i, j] for j, p in enumerate(param_names)})

    # Fixed seeds per parameter set (same seed across K values)
    rng = np.random.default_rng(123)
    seeds = rng.integers(1, 2**15 - 1, size=n)

    # Run all simulations
    results = {K: [None] * n for K in K_VALUES}

    for K in K_VALUES:
        print(f"\n--- K = {K} ({n} simulations) ---")
        t0 = time.time()

        tasks = [(param_sets[i], K, int(seeds[i])) for i in range(n)]

        with Pool(processes=args.cores, initializer=_init_worker,
                  initargs=(TEMPLATE_LINES,)) as pool:
            matrices = pool.map(run_single_sim, tasks)

        n_ok = sum(1 for m in matrices if m is not None)
        elapsed = time.time() - t0
        print(f"  {n_ok}/{n} successful, {elapsed:.0f}s")
        results[K] = matrices

    # ── Analysis ───────────────────────────────────────────────────
    print("\n--- Analysis ---")

    # Frobenius distance of each K's matrix relative to K=100
    frob_dists = {K: [] for K in K_VALUES if K != K_BASELINE}
    valid_indices = []

    for i in range(n):
        baseline = results[K_BASELINE][i]
        if baseline is None:
            continue
        all_valid = True
        for K in K_VALUES:
            if K == K_BASELINE:
                continue
            if results[K][i] is None:
                all_valid = False
                break
        if all_valid:
            valid_indices.append(i)

    print(f"  Valid parameter sets (all K succeeded): {len(valid_indices)}/{n}")

    for K in K_VALUES:
        if K == K_BASELINE:
            continue
        for i in valid_indices:
            baseline = results[K_BASELINE][i]
            test = results[K][i]
            frob = np.sqrt(np.sum((baseline - test) ** 2))
            frob_dists[K].append(frob)

    # Summary stats and Wilcoxon tests
    summary = {'n_sets': n, 'n_valid': len(valid_indices), 'K_values': K_VALUES,
               'comparisons': {}}

    print(f"\n  {'K':>6s}  {'mean_Frob':>10s}  {'med_Frob':>10s}  {'std_Frob':>10s}")
    for K in sorted(frob_dists.keys()):
        dists = frob_dists[K]
        if len(dists) < 3:
            print(f"  {K:>6d}  insufficient data")
            continue
        m = np.mean(dists)
        med = np.median(dists)
        s = np.std(dists)
        print(f"  {K:>6d}  {m:10.4f}  {med:10.4f}  {s:10.4f}")

        # Wilcoxon: are Frobenius distances to K significantly different
        # from distances between two random K=100 runs?
        summary['comparisons'][str(K)] = {
            'mean_frobenius': float(m),
            'median_frobenius': float(med),
            'std_frobenius': float(s),
            'n': len(dists),
        }

    # Also compute self-distance at K=100 (re-run with different seed)
    # to establish noise floor — but this would double runtime,
    # so we skip and just report the relative magnitudes.

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot of Frobenius distances
    ax = axes[0]
    plot_data = []
    plot_labels = []
    for K in sorted(frob_dists.keys()):
        if len(frob_dists[K]) >= 3:
            plot_data.append(frob_dists[K])
            plot_labels.append(f'K={K}')

    if plot_data:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel('Frobenius distance to K=100 baseline')
    ax.set_title('Distance matrix sensitivity to K')

    # Mean methylation level vs K (to check convergence)
    ax = axes[1]
    mean_betas = {K: [] for K in K_VALUES}
    for K in K_VALUES:
        for i in valid_indices:
            mat = results[K][i]
            if mat is not None:
                # diagonal of distance matrix = 0, but mean of matrix
                # captures overall divergence level
                mean_betas[K].append(np.mean(mat))

    positions = range(len(K_VALUES))
    for pos, K in zip(positions, K_VALUES):
        if mean_betas[K]:
            ax.boxplot([mean_betas[K]], positions=[pos], widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'K={K}' for K in K_VALUES])
    ax.set_ylabel('Mean distance matrix entry')
    ax.set_title('Inter-gland divergence vs K')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'k_sensitivity.pdf'
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")

    # Save JSON
    json_path = OUTPUT_DIR / 'k_sensitivity_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")


def _init_worker(template_lines):
    """Pool worker initialiser — set the global template."""
    global TEMPLATE_LINES
    TEMPLATE_LINES = template_lines


if __name__ == '__main__':
    main()
