r""" Build the action of the CC Jacobian matrix, as defined in Eq. (33) of
Ref. [1].

A _μν = <μ| e^{-T} [H, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operators.

The above can be translated to

A _μν = <HF| tau * _μ [\bar{H}, tau_ν] |HF>

which can be implemented with pdaggerq. 

The action of the CC Jacobian means that for any input vector w, I return the
vector Aw. Because 
(Aw) _μ 
= A _μν w_ν 
= <HF| tau * _μ [\bar{H}, tau_ν] |HF> w_ν 
= <HF| tau * _μ [\bar{H}, (w_ν tau_ν)] |HF> 
= <HF| tau * _μ [\bar{H}, w] |HF> 
this may be implemented with pdaggerq thanks to the fact that w has the same
structure as the r vector.

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""

from rspn.ghf_ccsd.equations.printer import (
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
    pq.set_left_operators([[projection]])

    # The f part of H
    pq.add_st_operator(1.0, ['f', 'r1'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['f', 'r2'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r1', 'f'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r2', 'f'], ['t1', 't2'])

    # The v part of H
    pq.add_st_operator(1.0, ['v', 'r1'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v', 'r2'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r1', 'v'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r2', 'v'], ['t1', 't2'])

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
    pq.set_left_operators([[projection]])

    # The f part of H
    pq.add_st_operator(1.0, ['f', 'r1'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['f', 'r2'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r1', 'f'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r2', 'f'], ['t1', 't2'])

    # The v part of H
    pq.add_st_operator(1.0, ['v', 'r1'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v', 'r2'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r1', 'v'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['r2', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():
    do_singles = False
    do_doubles = True

    if do_singles:
        pq = build_singles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_w_singles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'i'),
            extra_arguments=[
                'vector: GHF_CCSD_MBE',
            ],
            extra_definitions=[
                'r1 = vector.singles',
                'r2 = vector.doubles',
            ],
        )

    elif do_doubles:
        pq = build_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_w_doubles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'b', 'i', 'j'),
            extra_arguments=[
                'vector: GHF_CCSD_MBE',
            ],
            extra_definitions=[
                'r1 = vector.singles',
                'r2 = vector.doubles',
            ],
        )


if __name__ == "__main__":
    main()
