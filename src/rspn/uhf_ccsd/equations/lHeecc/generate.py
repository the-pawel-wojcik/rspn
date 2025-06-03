r""" Build a matrix defined in Eq. (77) of Ref. [1].

           l    H    e       e     cc
F _{νγ} = <Λ| [[H, tau_ν], tau_ν] |CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operator. The operator H is the molecular Hamiltonian.

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal of
Chemical Physics 93, 3333 (1990).
"""
import pdaggerq
from rspn.uhf_ccsd.equations.printer import print_to_numpy


def build_lHe1e1cc():
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # [H, e1(a,i)] = H e1(a,i) - e1(a,i)H

    #  [[H, e1(a,i)], e1(b,j)]
    #
    #     = [H, e1(a,i)] e1(b,j)
    #     - e1(b,j) [H, e1(a,i)]
    #
    #     = (H e1(a,i) - e1(a,i)H) e1(b,j)
    #     - e1(b,j) (H e1(a,i) - e1(a,i)H)
    #
    #     =     H e1(a,i) e1(b,j)
    #         - e1(a,i) H e1(b,j)
    #         - e1(b,j) H e1(a,i) 
    #         + e1(b,j) e1(a,i) H

    pq.add_st_operator(1.0, ['v', 'e1(a,i)', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(a,i)', 'v', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(b,j)', 'v', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e1(b,j)', 'e1(a,i)', 'v'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['f', 'e1(a,i)', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(a,i)', 'f', 'e1(b,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(b,j)', 'f', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e1(b,j)', 'e1(a,i)', 'f'], ['t1', 't2'])


    pq.simplify()

    return pq


def main():

    do_e1e1 = True
    do_e1e2 = False
    do_e2e1 = False
    do_e2e2 = False

    if do_e1e1:
        pq = build_lHe1e1cc()
        print_to_numpy(
            pq, tensor_name='lhe1e1cc', tensor_subscripts=('a', 'i', 'b', 'j'),
        )
    # else: # TODO


if __name__ == "__main__":
    main()
