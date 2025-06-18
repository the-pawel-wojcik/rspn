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
from rspn.uhf_ccs.equations.printer import (
        DefineSections, print_to_numpy, print_imports,
)
import pdaggerq


def build_singles_singles_block():
    """ Builds A _{ai bj}"""
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a_a
    pq.set_left_operators([['e1(i,a)']])  # Replace the MO#i with an MO#a

    # commutator
    pq.add_st_operator(1.0, ['f', 'e1(b,j)'], ['t1'])
    pq.add_st_operator(-1.0, ['e1(b,j)', 'f'], ['t1'])

    pq.add_st_operator(1.0, ['v', 'e1(b,j)'], ['t1'])
    pq.add_st_operator(-1.0, ['e1(b,j)', 'v'], ['t1'])

    pq.simplify()

    return pq


def main():
    do_singles_singles = True
    if do_singles_singles:
        excludes = {DefineSections.LAMBDA_AMPS}
        pq = build_singles_singles_block()
        print_imports(defines_exclude=excludes)
        print_to_numpy(
            pq,
            tensor_name='cc_j_singles_singles',
            defines_exclude=excludes,
            tensor_subscripts=('a', 'i', 'b', 'j'),
        )


if __name__ == "__main__":
    main()
