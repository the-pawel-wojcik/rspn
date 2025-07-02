r""" Build a matrix defined in Eq. (77) of Ref. [1].

           l    H    e       e     cc
F _{νγ} = <Λ| [[H, tau_ν], tau_γ] |CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operator. The operator H is the molecular Hamiltonian.

Instead of building the whole F, which would be a large object to store,
calcaulate instead it's action on any test vector; aka the sigma build


                       l    H    t      w     cc
σ _ν = F _{νγ} w _γ = <Λ| [[H, tau_ν], tau_γ] |CC> w_γ
                    = <Λ| [[H, tau_ν], w_γ tau_γ] |CC>
                    = <Λ| [[H, tau_ν], w] |CC>

The object `w` has the same structure as the `r` vector

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
import pdaggerq
from rspn.uhf_ccsd.equations.printer import print_imports, print_to_numpy


def build_lHtwcc(tau: str):
    """ Tau is the un-contracted excitation operator: either 'e1(a,i') for
    singles or 'e2(a,b,j,i)' for doubles. """

    # [H, e1(a,i)] = H e1(a,i) - e1(a,i)H

    #  [[H, e1(a,i)], r]
    #
    #     = [H, e1(a,i)] r
    #     - r [H, e1(a,i)]
    # 
    #     = [H, e1(a,i)] r
    #     - r [H, e1(a,i)]
    #
    #     = (H e1(a,i) - e1(a,i)H) r
    #     - r (H e1(a,i) - e1(a,i)H)
    #
    #     =     H e1(a,i) r
    #         - e1(a,i) H r
    #         - r H e1(a,i)
    #         + r e1(a,i) H

    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    for r in ['r1', 'r2']:
        pq.add_st_operator(1.0, ['v', tau, r], ['t1', 't2'])
        pq.add_st_operator(-1., [tau, 'v', r], ['t1', 't2'])
        pq.add_st_operator(-1., [r, 'v', tau], ['t1', 't2'])
        pq.add_st_operator(1.0, [r, tau, 'v'], ['t1', 't2'])

        pq.add_st_operator(1.0, ['f', tau, r], ['t1', 't2'])
        pq.add_st_operator(-1., [tau, 'f', r], ['t1', 't2'])
        pq.add_st_operator(-1., [r, 'f', tau], ['t1', 't2'])
        pq.add_st_operator(1.0, [r, tau, 'f'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():

    do_e1 = False
    do_e2 = True
    extra_arguments = ['vector: Spin_MBE',]
    extra_definitions = [
        'r1_aa = vector.singles[E1_spin.aa]',
        'r1_bb = vector.singles[E1_spin.bb]',
        'r2_aaaa = vector.doubles[E2_spin.aaaa]',
        'r2_abab = vector.doubles[E2_spin.abab]',
        'r2_abba = vector.doubles[E2_spin.abba]',
        'r2_baab = vector.doubles[E2_spin.baab]',
        'r2_baba = vector.doubles[E2_spin.baba]',
        'r2_bbbb = vector.doubles[E2_spin.bbbb]',
    ]

    print_imports()

    if do_e1:
        pq = build_lHtwcc('e1(a,i)')
        print_to_numpy(
            pq,
            tensor_name='lHtauwCC_singles',
            tensor_subscripts=('a', 'i'),
            extra_arguments=extra_arguments,
            extra_definitions=extra_definitions,
        )

    elif do_e2:
        pq = build_lHtwcc('e2(a,b,j,i)')
        print_to_numpy(
            pq,
            tensor_name='lHtauwCC_doubles',
            tensor_subscripts=('a', 'b', 'j', 'i'),
            extra_arguments=extra_arguments,
            extra_definitions=extra_definitions,
        )


if __name__ == "__main__":
    main()
