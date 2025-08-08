# Linear Response Properties
A package for the calculation of polarizabilities of molecules.

## Dependencies
This package builds on top of the quantum chemistry programs from the
[chem](https://github.com/the-pawel-wojcik/chem) package. `chem`, in turn,
depends on [psi4](https://psicode.org/), this dependency is not listed in the
`pyproject.toml` as `psi4` is unavailable from pypi.

## Install
1. Please follow the installation instructions for the
   [chem](https://github.com/the-pawel-wojcik/chem) package.
2. Use pip to install this package
```bash
git clone git@github.com:the-pawel-wojcik/rspn.git
cd rspn
python -m pip install -e .
```

## Use
The `examples` directory contains a complete input and output of a calculation
of the polarizability of water in the STO-3G basis set.

## Test
Tests use `pytest`. Run them with
```bash
python -m pip install pytest
cd tests
pytest -v
```

## Bottleneck
The GHF-CCSD-LR polarizability works, but it is not optimized. The current
bottleneck is the construction of the two-electron matrices like the CC
Jacobian. The script in `memuse` shows where this bottleneck happens. The way
around it is to implement the action of the CC Jacobian, which would lift the
need for storing this matrix.

## Docs
The `docs` directory contains a summary of the theory needed for the
implementation of this method.

## Lock files
This package was developed with Python 3.12.10, Psi4 1.10a1.dev11, numpy 2.2.6,
and scipy 1.15.3.
