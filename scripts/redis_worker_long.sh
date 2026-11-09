#!/bin/bash

# usage:
# sbatch -D <methabc_dir> --partition=<partition> --nodes=<nodes> --ntasks-per-node=<tasks> \
#     --mem=<mem> --time=<time> -o <log_dir>/%x_%j.o -e <log_dir>/%x_%j.e \
#     scripts/redis_worker_long.sh <host> <port> <runtime> <processes>

if [ "$#" -ne 4 ]; then
    echo "usage: redis_worker_long.sh <host> <port> <runtime> <processes>"
    exit 1
fi

host=$1
port=$2
runtime=$3
processes=$4

# prepare environment, e.g. set path

# run
abc-redis-worker --host=$host --port=$port --runtime=$runtime --processes=$processes
