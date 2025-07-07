r""" Build the matrix elements of the CC Jacobian, as defined in Eq. (33) of
Ref. [1].

A _μν = <μ| e^{-T} [H, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operators.

The above can be translated to

A _μν = <HF| tau * _μ [\bar{H}, tau_ν] |HF>

which can be implemented with pdaggerq

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
import argparse

from rspn.ghf_ccsd.equations.printer import (
        DefineSections, print_to_numpy, print_imports,
)
import pdaggerq


def build_singles_singles_block():
    """ Builds A _{ai bj}"""
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a_a
    pq.set_left_operators([['e1(i,a)']])  # (Replace i with a )*

    # commutator
    pq.add_st_operator(1.0, ['f', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(b,j)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(b,j)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_singles_doubles_block():
    """ Builds A _{ai bckj} """
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a_a
    pq.set_left_operators([['e1(i,a)']])  # (Replace i with a)*

    # commutator
    pq.add_st_operator(1.0, ['f', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(b,c,k,j)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(b,c,k,j)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_singles_block():
    """ Builds A _{abji ck} """
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a*_j a_b a_a
    pq.set_left_operators([['e2(i,j,b,a)']])

    # commutator
    pq.add_st_operator(1.0, ['f', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(c,k)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(c,k)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_doubles_block():
    """ Builds A _{abji cdlk} """
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a*_j a_b a_a
    pq.set_left_operators([['e2(i,j,b,a)']])

    # commutator
    pq.add_st_operator(1.0, ['f', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--ss', default=False, action='store_true')
    options.add_argument('--sd', default=False, action='store_true')
    options.add_argument('--ds', default=False, action='store_true')
    options.add_argument('--dd', default=False, action='store_true')
    args = parser.parse_args()

    if args.ss:
        pq = build_singles_singles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_singles_singles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'i', 'b', 'j'),
        )

    elif args.sd:
        pq = build_singles_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_singles_doubles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'i', 'b', 'c', 'k', 'j'),
        )

    elif args.ds:
        pq = build_doubles_singles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_doubles_singles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'b', 'j', 'i', 'c', 'k'),
        )

    elif args.dd:
        pq = build_doubles_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_doubles_doubles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k'),
        )


if __name__ == "__main__":
    main()
