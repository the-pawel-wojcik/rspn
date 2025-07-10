r""" Build the matrix elements of the CC Jacobian, as defined in Eq. (33) of
Ref. [1].

A _μν = <μ| e^{-T} [H, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operators. In this script it will be only doubles.

The above expression is implemented in pdaggerq through
A _μν = <HF| tau * _μ [\bar{H}, tau_ν] |HF>

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms

pq = pdaggerq.pq_helper('fermi')

# <mu| = <HF| a*_i a*_j a_b a_a
pq.set_left_operators([['e2(i,j,b,a)']])

# commutator [Hbar, a*_c a*_d a_l a_k]
pq.add_st_operator(1.0, ['f', 'e2(c,d,l,k)'], ['t1', 't2'])
pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'f'], ['t1', 't2'])

pq.add_st_operator(1.0, ['v', 'e2(c,d,l,k)'], ['t1', 't2'])
pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'v'], ['t1', 't2'])

pq.simplify()

terms = pq.strings()
tensor_terms = contracted_strings_to_tensor_terms(terms)
for my_term in tensor_terms:
    einsum_terms = my_term.einsum_string(
        output_variables=('a', 'b', 'i', 'j', 'c', 'd', 'k', 'l'),
        update_val='cc_j_doubles_doubles',
    )
    for print_term in einsum_terms.split('\n'):
        print(print_term)
