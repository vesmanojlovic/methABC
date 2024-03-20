#!/bin/bash

# slurm settings
#SBATCH -D /users/adby616/methABC
#SBATCH --partition=week
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
#SBATCH --time=168:00:00
#SBATCH -o /users/adby616/archive/abc-temp/%x_%j.o
#SBATCH -e /users/adby616/archive/abc-temp/%x_%j.e

# prepare environment, e.g. set path

# run
abc-redis-worker --host=10.10.0.21 --port=2166 --runtime=7d --processes=192

