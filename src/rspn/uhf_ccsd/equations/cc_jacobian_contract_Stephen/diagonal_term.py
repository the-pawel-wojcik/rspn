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

which can be implemented with pdaggerq

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
from rspn.uhf_ccsd.equations.printer import (
    DefineSections, print_to_numpy, print_imports,
)
import pdaggerq


def try_out_diagonal_term():
    r"""
    The diagonal part

     - E_CC * <HF| e1(i,a) (r1 + r2) |HF>
    """
    pq = pdaggerq.pq_helper('fermi')

    projection = 'e1(i,a)'

    pq.add_operator_product(1.0, [projection, 'r1'])
    pq.add_operator_product(1.0, [projection, 'r2'])

    pq.simplify()

    return pq


def main():

    pq = try_out_diagonal_term()
    print_imports()
    print_to_numpy(
        pq,
        tensor_name='diagonal_part',
        defines_exclude={DefineSections.LAMBDA_AMPS},
        tensor_subscripts=('a', 'i'),
        extra_arguments=[
            'vector: Spin_MBE',
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
