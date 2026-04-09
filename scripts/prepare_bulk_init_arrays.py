#!/usr/bin/env python3
"""
Pre-process bulk methylation arrays into binary initial arrays for methdemon.

Reads the filtered fCpG data (1164 loci) with bulk side A and B columns,
averages them to get a tumour-level initial methylation profile, then converts
to the 2400-value binary format that methdemon's manual_array expects.

Internal representation:
    methdemon uses 2 * fCpG_loci_per_cell binary values per cell.
    Locus j has two alleles: site[j] and site[j + n_loci].
    Output beta for locus j = (site[j] + site[j + n_loci]) / 2.

Conversion:
    For each locus with beta=b, each allele is independently set to 1
    with probability b (Bernoulli draw). This gives E[output_beta] = b.

Output:
    data/bulk_init_arrays/tumour_{ID}_init.txt  (2400 ints, one per line)
    data/bulk_init_arrays/manifest.json         (metadata per tumour)

Usage:
    python scripts/prepare_bulk_init_arrays.py

Requires access to the crcfcpg data directory for the source bulk arrays.
The output files are committed to methabc so the inference script can use
them without needing crcfcpg on the HPC.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────

CRCFCPG_DATA = Path.home() / "Documents" / "github_repos" / "crcfcpg" / "data"
FULL_ARRAYS_DIR = CRCFCPG_DATA / "full_arrays"
INDIV_DIR = CRCFCPG_DATA / "individual"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bulk_init_arrays"

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]
N_LOCI = 1164   # fCpG loci after filtering
N_INTERNAL = N_LOCI * 2  # methdemon internal array size
RNG_SEED = 2026  # reproducible conversion


def find_bulk_columns(df_full, df_indiv, tid):
    """Identify the two bulk columns (not in individual file)."""
    indiv_cols = set(df_indiv.columns)
    bulk = [c for c in df_full.columns if c not in indiv_cols]
    if len(bulk) != 2:
        raise ValueError(
            f"Tumour {tid}: expected 2 bulk columns, got {bulk}")
    return bulk


def beta_to_binary(beta_array, rng):
    """Convert beta values to 2*N_LOCI binary array."""
    n = len(beta_array)
    arr = np.zeros(2 * n, dtype=int)
    for j in range(n):
        b = beta_array[j]
        arr[j] = 1 if rng.random() < b else 0
        arr[j + n] = 1 if rng.random() < b else 0
    return arr


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    manifest = {}

    for tid in TUMOUR_IDS:
        print(f"Processing tumour {tid}...")

        # Load individual file to get fCpG labels and gland column names
        indiv = pd.read_csv(INDIV_DIR / f"tumour_{tid}.csv", index_col=0)
        keep_labels = set(indiv.index)

        # Load full array and filter to fCpG loci
        full = pd.read_csv(FULL_ARRAYS_DIR / f"{tid}_norm.csv", index_col=0)
        full = full[full["Labels"].isin(keep_labels)].copy()
        full = full.set_index("Labels")

        # Find bulk columns
        bulk_cols = find_bulk_columns(full, indiv, tid)
        bulk_a = full[bulk_cols[0]].values
        bulk_b = full[bulk_cols[1]].values

        # Check for NaN
        mask = ~(np.isnan(bulk_a) | np.isnan(bulk_b))
        if mask.sum() != N_LOCI:
            print(f"  WARNING: {N_LOCI - mask.sum()} NaN loci dropped")
            bulk_a = bulk_a[mask]
            bulk_b = bulk_b[mask]

        # Average bulk A and B
        avg_beta = (bulk_a + bulk_b) / 2.0

        # Compute correlation between bulk and gland averages (for flagging)
        gland_avg = indiv.mean(axis=1).values
        if len(gland_avg) == len(avg_beta):
            from scipy.stats import pearsonr
            r_bulk_gland, _ = pearsonr(avg_beta, gland_avg)
        else:
            r_bulk_gland = float('nan')

        # Convert to binary
        binary_arr = beta_to_binary(avg_beta, rng)

        # Verify
        reconstructed_beta = (binary_arr[:N_LOCI] + binary_arr[N_LOCI:]) / 2.0
        reconstruction_corr = np.corrcoef(avg_beta, reconstructed_beta)[0, 1]

        # Save
        out_path = OUTPUT_DIR / f"tumour_{tid}_init.txt"
        np.savetxt(out_path, binary_arr, fmt='%d')

        # Flag tumour M
        flag = ""
        if abs(r_bulk_gland) < 0.3:
            flag = "WARNING: bulk arrays poorly correlated with gland averages"

        manifest[tid] = {
            'file': f"tumour_{tid}_init.txt",
            'n_loci': int(len(avg_beta)),
            'n_internal': int(len(binary_arr)),
            'mean_beta_A': float(np.mean(bulk_a)),
            'mean_beta_B': float(np.mean(bulk_b)),
            'mean_beta_avg': float(np.mean(avg_beta)),
            'r_bulk_AB': float(np.corrcoef(bulk_a, bulk_b)[0, 1]),
            'r_bulk_vs_gland_avg': float(r_bulk_gland),
            'reconstruction_r': float(reconstruction_corr),
            'bulk_col_A': bulk_cols[0],
            'bulk_col_B': bulk_cols[1],
            'flag': flag,
        }

        status = f"  mean_beta={np.mean(avg_beta):.3f}  " \
                 f"r(A,B)={np.corrcoef(bulk_a, bulk_b)[0, 1]:.3f}  " \
                 f"r(bulk,gland)={r_bulk_gland:.3f}"
        if flag:
            status += f"  *** {flag} ***"
        print(status)

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(TUMOUR_IDS)} init arrays written to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
