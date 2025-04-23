# methABC
Approximate Bayesian Computation framework for inferring evolutionary parameters of colorectal cancer from methylation arrays.

## Installation
Clone repo recursively to include submodule
```
git clone https://github.com/vesmanojlovic/methABC --recursive
```
Create virtual environment and install dependencies
```
pyenv virtualenv 3.11 methabc
pyenv activate methabc
pip install -e .
pip install -r requirements.txt
```

If you'd like to recompile methdemon from source (or have to because the included one throws an error), edit the `Makefile` in `resources/methdemon/` to include the path to your copy of the `boost` C++ library. Then run `make all` to compile the binary.
