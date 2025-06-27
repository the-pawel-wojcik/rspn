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

A _μν = <HF| tau * _ \bar{H} tau_ν |HF>
        -
        \delta _{μν} E_{CC}

which can be implemented with pdaggerq.

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
from rspn.uhf_ccsd.equations.printer import (
    DefineSections, print_to_numpy, print_imports,
)
import pdaggerq


def build_singles_block():
    r""" Builds (A w) _{ai} 

    (A _w) _μ =   1.0 * <HF| e1(i,a) \bar{H} (r1 + r2) |HF> 
                - 1.0 * E_CC * <HF| e1(i,a) (r1 + r2) |HF>

    Here, I generate only the first term. I call the second term the diagonal
    part and I add it manually, in the application.
    """
    pq = pdaggerq.pq_helper('fermi')

    projection = 'e1(i,a)'
    vector = ['r1', 'r2']

    # The f part of H
    pq.add_st_operator(1.0, [projection, 'f', *vector], ['t1', 't2'])
    pq.add_st_operator(1.0, [projection, 'v', *vector], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_block():
    """ See the comment from the singles block. """
    pq = pdaggerq.pq_helper('fermi')
    projection = 'e2(i,j,b,a)'
    vector = ['r1', 'r2']
    pq.add_st_operator(1.0, [projection, 'f', *vector], ['t1', 't2'])
    pq.add_st_operator(1.0, [projection, 'v', *vector], ['t1', 't2'])
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
                'vector: Spin_MBE',
                'E_CC: float',
            ],
            extra_definitions=[
                'r1_aa = vector.singles[E1_spin.aa]',
                'r1_bb = vector.singles[E1_spin.bb]',
                'r2_aaaa = vector.doubles[E2_spin.aaaa]',
                'r2_abab = vector.doubles[E2_spin.abab]',
                'r2_baba = vector.doubles[E2_spin.baba]',
                'r2_bbbb = vector.doubles[E2_spin.bbbb]',
            ],
        )

    elif do_doubles:
        pq = build_doubles_block()
        print_imports()
        print_to_numpy(
            pq,
            tensor_name='cc_j_w_doubles',
            defines_exclude={DefineSections.LAMBDA_AMPS},
            tensor_subscripts=('a', 'b', 'j', 'i'),
            extra_arguments=[
                'vector: Spin_MBE',
                'E_CC: float',
            ],
            extra_definitions=[
                'r1_aa = vector.singles[E1_spin.aa]',
                'r1_bb = vector.singles[E1_spin.bb]',
                'r2_aaaa = vector.doubles[E2_spin.aaaa]',
                'r2_abab = vector.doubles[E2_spin.abab]',
                'r2_baba = vector.doubles[E2_spin.baba]',
                'r2_bbbb = vector.doubles[E2_spin.bbbb]',
            ],
        )


if __name__ == "__main__":
    main()
