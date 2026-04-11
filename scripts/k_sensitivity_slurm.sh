#!/bin/bash

#SBATCH --job-name=k_sensitivity
#SBATCH -D /users/adby616/methABC
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH -o /users/adby616/archive/abc-temp/%x_%j.o
#SBATCH -e /users/adby616/archive/abc-temp/%x_%j.e

# Usage:
#   python scripts/k_sensitivity.py generate -n 50
#   sbatch scripts/k_sensitivity_slurm.sh              # full run
#   sbatch scripts/k_sensitivity_slurm.sh --dry-run    # 1-sim sanity check
#   # After completion:
#   python scripts/k_sensitivity.py collect
#
# Logs go to /users/adby616/archive/abc-temp/ (must exist).

set -eo pipefail
set -x

echo "=== Job diagnostics ==="
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "PWD:      $(pwd)"
echo "JobID:    ${SLURM_JOB_ID:-none}"
echo "Nodelist: ${SLURM_JOB_NODELIST:-none}"
echo "CPUs:     ${SLURM_CPUS_ON_NODE:-none}"
echo "Python:   $(which python || echo MISSING)"
echo "Version:  $(python --version 2>&1 || echo MISSING)"
echo "======================="

# Fail fast if prerequisites are missing — otherwise the Pool workers
# silently return None and you get an empty results dir.
for path in \
    resources/methdemon/bin/methdemon \
    resources/config_template.dat \
    scripts/k_sensitivity.py \
    outputs/k_sensitivity/params.json
do
    if [[ ! -e "$path" ]]; then
        echo "FATAL: missing $path" >&2
        echo "Run 'python scripts/k_sensitivity.py generate -n 50' first." >&2
        exit 1
    fi
done

mkdir -p outputs/k_sensitivity/matrices

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== DRY RUN: one simulation only ==="
    python -u scripts/k_sensitivity.py run --dry-run
    exit $?
fi

python -u scripts/k_sensitivity.py run --local --cores 48
