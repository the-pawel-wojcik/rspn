r""" Build the matrix elements of the CC Jacobian, as defined in Eq. (33) of
Ref. [1].

A _μν = <μ| e^{-T} [H, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operators.

The above can be translated to

A _μν = <HF| tau * _μ [\bar{H}, tau_ν] |HF>

A _μν = <HF| tau * _μ \bar{H} tau_ν |HF>
        -
        <HF| tau * _μ tau_ν \bar{H} |HF>

which can be implemented with pdaggerq

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
from rspn.uhf_ccsd.equations.printer import (
        DefineSections, print_to_numpy, print_imports,
)
import pdaggerq


def build_singles_block():
    r""" Builds (A w)_{ai} 

    (A _w) _μ = <HF| e1(i,a) \bar{H} (r1 + r2) |HF>
                -
                <HF| e1(i, a) (r1 + r2) \bar{H} |HF>
    """
    pq = pdaggerq.pq_helper('fermi')

    projection = 'e1(i,a)'
    vector = ['r1', 'r2']

    # The f part of H
    pq.add_st_operator(1.0, [projection, 'f', *vector], ['t1', 't2'])
    pq.add_st_operator(-1.0, [projection, *vector, 'f'], ['t1', 't2'])

    # The v part of H
    pq.add_st_operator(1.0, [projection, 'v', *vector], ['t1', 't2'])
    pq.add_st_operator(-1.0, [projection, *vector, 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_block():
    r""" Builds (A w)_{abji} 

    (A _w) _μ = <HF| e2(i,j,b,a) \bar{H} (r1 + r2) |HF>
                -
                <HF| e2(i,j,b,a) (r1 + r2) \bar{H} |HF>
    """
    pq = pdaggerq.pq_helper('fermi')

    projection = 'e2(i,j,b,a)'
    vector = ['r1', 'r2']

    # The f part of H
    pq.add_st_operator(1.0, [projection, 'f', *vector], ['t1', 't2'])
    pq.add_st_operator(-1.0, [projection, *vector, 'f'], ['t1', 't2'])

    # The v part of H
    pq.add_st_operator(1.0, [projection, 'v', *vector], ['t1', 't2'])
    pq.add_st_operator(-1.0, [projection, *vector, 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():

    do_singles = True
    do_doubles = False

    if do_singles:
        pq = build_singles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_w_singles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'i'),
            extra_arguments=[
                'r1_aa: NDArray',
                'r1_bb: NDArray',
                'r2_aaaa: NDArray',
                'r2_abab: NDArray',
                'r2_baba: NDArray',
                'r2_bbbb: NDArray',
            ],
        )

    elif do_doubles:
        pq = build_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_singles_doubles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'i', 'b', 'c', 'k', 'j'),
        )


if __name__ == "__main__":
    main()
