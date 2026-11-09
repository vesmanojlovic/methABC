# methABC
Approximate Bayesian Computation framework for inferring evolutionary parameters of colorectal cancer from methylation arrays.

## Installation
Clone repo recursively to include submodule
```
git clone https://github.com/vesmanojlovic/methABC --recursive
```
Create virtual environment and install dependencies
```
conda create -n methabc python=3.11
conda activate methabc
pip install -e .
```
Compile methdemon (requires the `boost` C++ library, see the [methdemon README](https://github.com/vesmanojlovic/methdemon))
```
make -C resources/methdemon all
```
To check everything works, run
```
pip install -e ".[test]"
pytest
```

Versions used for the paper runs: Python 3.11, pyabc 0.12.13, numpy 1.26.4, pandas 2.2.1, scipy 1.12.0, redis 5.0.1.

## Running inference
Inference on a single tumour is run with `scripts/log_model.py`, e.g. for tumour D on 8 local processes
```
python scripts/log_model.py -d D -n 8
```
The output database is written to `tmp/`. Run `python scripts/log_model.py -h` for all options (carrying capacity, turnover, distance, epsilon schedule, population size, etc.).

Inference is slow on a laptop. On a cluster, start a Redis server, run `log_model.py` without `-n` (pointing `--host` and `--port` to the server), and start workers with `scripts/redis_worker_long.sh` or `scripts/redis_worker_short.sh` (see the usage at the top of each script).
