#!/bin/bash

#SBATCH --job-name=k_sensitivity
#SBATCH -D /users/adby616/methABC
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH -o outputs/k_sensitivity/%x_%j.o
#SBATCH -e outputs/k_sensitivity/%x_%j.e

# Usage:
#   python scripts/k_sensitivity.py generate -n 50
#   mkdir -p outputs/k_sensitivity
#   sbatch scripts/k_sensitivity_slurm.sh
#   # After completion:
#   python scripts/k_sensitivity.py collect

python scripts/k_sensitivity.py run --local --cores 48
