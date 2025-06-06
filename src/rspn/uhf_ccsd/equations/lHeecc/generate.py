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
from rspn.uhf_ccsd.equations.printer import print_imports, print_to_numpy


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


def build_lHe1e2cc():
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # [H, e1(a,i)] = H e1(a,i) - e1(a,i)H

    #  [[H, e1(a,i)], e2(b,c,k,j)]
    #
    #     = [H, e1(a,i)] e2(b,c,k,j)
    #     - e2(b,c,k,j) [H, e1(a,i)]
    #
    #     = (H e1(a,i) - e1(a,i)H) e2(b,c,k,j)
    #     - e2(b,c,k,j) (H e1(a,i) - e1(a,i)H)
    #
    #     =     H e1(a,i) e2(b,c,k,j)
    #         - e1(a,i) H e2(b,c,k,j)
    #         - e2(b,c,k,j) H e1(a,i) 
    #         + e2(b,c,k,j) e1(a,i) H

    pq.add_st_operator(1.0, ['v', 'e1(a,i)', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(a,i)', 'v', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(b,c,k,j)', 'v', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e2(b,c,k,j)', 'e1(a,i)', 'v'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['f', 'e1(a,i)', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(a,i)', 'f', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(b,c,k,j)', 'f', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e2(b,c,k,j)', 'e1(a,i)', 'f'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_lHe2e1cc():
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # [H, e2(a,b,j,i)] = H e2(a,b,j,i) - e2(a,b,j,i)H

    #  [[H, e2(a,b,j,i)], e1(c,k)]
    #
    #     = [H, e2(a,b,j,i)] e1(c,k)
    #     - e1(c,k) [H, e2(a,b,j,i)]
    #
    #     = (H e2(a,b,j,i) - e2(a,b,j,i)H) e1(c,k)
    #     - e1(c,k) (H e2(a,b,j,i) - e2(a,b,j,i)H)
    #
    #     =     H e2(a,b,j,i) e1(c,k)
    #         - e2(a,b,j,i) H e1(c,k)
    #         - e1(c,k) H e2(a,b,j,i) 
    #         + e1(c,k) e2(a,b,j,i) H

    pq.add_st_operator(1.0, ['v', 'e2(a,b,j,i)', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(a,b,j,i)', 'v', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(c,k)', 'v', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e1(c,k)', 'e2(a,b,j,i)', 'v'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['f', 'e2(a,b,j,i)', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(a,b,j,i)', 'f', 'e1(c,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e1(c,k)', 'f', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e1(c,k)', 'e2(a,b,j,i)', 'f'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_lHe2e2cc():
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # [H, e2(a,b,j,i)] = H e2(a,b,j,i) - e2(a,b,j,i)H

    #  [[H, e2(a,b,j,i)], e2(c,d,l,k)]
    #
    #     = [H, e2(a,b,j,i)] e2(c,d,l,k)
    #     - e2(c,d,l,k) [H, e2(a,b,j,i)]
    #
    #     = (H e2(a,b,j,i) - e2(a,b,j,i)H) e2(c,d,l,k)
    #     - e2(c,d,l,k) (H e2(a,b,j,i) - e2(a,b,j,i)H)
    #
    #     =     H e2(a,b,j,i) e2(c,d,l,k)
    #         - e2(a,b,j,i) H e2(c,d,l,k)
    #         - e2(c,d,l,k) H e2(a,b,j,i) 
    #         + e2(c,d,l,k) e2(a,b,j,i) H

    pq.add_st_operator(1.0, ['v', 'e2(a,b,j,i)', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(a,b,j,i)', 'v', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(c,d,l,k)', 'v', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e2(c,d,l,k)', 'e2(a,b,j,i)', 'v'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['f', 'e2(a,b,j,i)', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(a,b,j,i)', 'f', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1., ['e2(c,d,l,k)', 'f', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['e2(c,d,l,k)', 'e2(a,b,j,i)', 'f'], ['t1', 't2'])

    pq.simplify()

    return pq


def main():

    do_e1e1 = False
    do_e1e2 = False
    do_e2e1 = False
    do_e2e2 = True

    print_imports()

    if do_e1e1:
        pq = build_lHe1e1cc()
        print_to_numpy(
            pq,
            tensor_name='lhe1e1cc',
            tensor_subscripts=('a', 'i', 'b', 'j'),
        )

    elif do_e1e2:
        pq = build_lHe1e2cc()
        print_to_numpy(
            pq,
            tensor_name='lhe1e2cc',
            tensor_subscripts=('a', 'i', 'b', 'c', 'k', 'j'),
        )

    elif do_e2e1:
        pq = build_lHe2e1cc()
        print_to_numpy(
            pq,
            tensor_name='lhe2e1cc',
            tensor_subscripts=('a', 'b', 'j', 'i', 'c', 'k'),
        )

    elif do_e2e2:
        pq = build_lHe2e2cc()
        print_to_numpy(
            pq,
            tensor_name='lhe2e2cc',
            tensor_subscripts=('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k'),
        )


if __name__ == "__main__":
    main()
