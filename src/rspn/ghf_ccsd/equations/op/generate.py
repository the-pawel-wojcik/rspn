r""" The perturbation operator needed as given by Eq. (60) from Ref. [1]
$$
\xi _{\nu} (\omega)
=
\bra{0} \tau _{\nu} e ^{-T} V ^{\omega} e ^{T} \ket{0}
$$


The version of this prorgam at the time of writing this comment works only with
the electric dipole moment operator.

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
import argparse

import pdaggerq
from rspn.ghf_ccsd.equations.printer import (
    DefineSections,
    print_to_numpy,
    print_imports,
)
from chem.meta.coordinates import CARTESIAN


def singles():
    print_imports()
    for component in CARTESIAN:
        pq = pdaggerq.pq_helper('fermi')
        pq.add_st_operator(1.0, ['e1(i,a)', 'h'], ['t1', 't2'])
        pq.simplify()
        extra_definitions = (
            f'h = ghf_data.mu[Descartes.{component}]',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}_singles',
            tensor_subscripts=('a', 'i'),
            defines_exclude={
                DefineSections.IDENTITY,
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
                DefineSections.LAMBDA_AMPS,
            },
            extra_definitions=extra_definitions,
        )


def doubles():
    print_imports()
    for component in CARTESIAN:
        pq = pdaggerq.pq_helper('fermi')
        pq.add_st_operator(1.0, ['e2(i,j,b,a)', 'h'], ['t1', 't2'])
        pq.simplify()
        extra_definitions = (
            f'h = ghf_data.mu[Descartes.{component}]',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}_doubles',
            tensor_subscripts=('a', 'b', 'j', 'i'),
            defines_exclude={
                DefineSections.IDENTITY,
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
                DefineSections.LAMBDA_AMPS,
            },
            extra_definitions=extra_definitions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--singles', default=False, action='store_true')
    options.add_argument('--doubles', default=False, action='store_true')
    args = parser.parse_args()

    if args.singles:
        singles()

    if args.doubles:
        doubles()
