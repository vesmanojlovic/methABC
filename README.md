# methABC
Approximate Bayesian Computation framework for inferring evolutionary parameters of colorectal cancer from methylation arrays.

## Installation
Clone repo recursively to include submodule
```
git clone https://github.com/vesmanojlovic/methABC --recursive
cd methABC/resources/methdemon
```
Change the `Makefile` to include the path to your boost library. Then run
```
make all
cd ../..
```
Finally, install the python dependencies
```
pyenv virtualenv -m methabc
pip install -e .
pip install -r requirements.txt
```
