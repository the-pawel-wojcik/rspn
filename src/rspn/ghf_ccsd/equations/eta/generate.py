r""" Build a matrix defined in Eq. (76) of Ref. [1].

η _μ = <Λ| [V^{\omega _1}, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operator. The operator V^{\omega _1} is a Fourier
transform of the interaction operator, see Ref. [2] for details and examples.
In case of a static electric field perturbation $-μE$, the operaotr
$V ^{\omega _1}$ reduces to the dipole operator $\mu$.

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
[2] J. Olsen and P. Jørgensen, Linear and nonlinear response functions for an
exact state and for an MCSCF state, The Journal of Chemical Physics 82, 3235
(1985).
"""
import argparse

import pdaggerq
from rspn.ghf_ccsd.equations.printer import (
    DefineSections,
    print_to_numpy,
    print_imports,
)


def build_singles_block():
    """ Builds eta _{ai} """
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # commutator [A, \tau] -- A is a generic one-particle operator, i.e., the
    # dipole moment operator \mu
    pq.add_st_operator(1.0, ['h', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(a,i)', 'h'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_block():
    """ Builds eta _{abji} """
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # commutator [A, \tau] -- A is a generic one-particle operator, i.e., the
    # dipole moment operator \mu
    pq.add_st_operator(1.0, ['h', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(a,b,j,i)', 'h'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--singles', default=False, action='store_true')
    options.add_argument('--doubles', default=False, action='store_true')
    args = parser.parse_args()

    if args.singles:
        pq = build_singles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='eta',
            tensor_subscripts=('a', 'i'),
            defines_exclude={
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
            },
            extra_arguments=[
                'operator: NDArray',
            ],
            extra_definitions=[
                'h = operator',
            ],
        )

    elif args.doubles:
        pq = build_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='eta',
            tensor_subscripts=('a', 'b', 'i', 'j'),
            defines_exclude={
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
            },
            extra_arguments=[
                'operator: NDArray',
            ],
            extra_definitions=[
                'h = operator',
            ],
        )


if __name__ == "__main__":
    main()
