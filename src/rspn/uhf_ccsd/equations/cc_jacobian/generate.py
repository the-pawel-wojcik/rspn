r""" Build the matrix elements of the CC Jacobian, as defined in Eq. (33) of
Ref. [1].

A _μν = <μ| e^{-T} [H, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operators.

The above can be translated to 

A _μν = <HF| tau * _μ [\bar{H}, tau_ν] |HF>

which can be implemented with pdaggerq

Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal of Chemical Physics 93, 3333 (1990).
"""
import itertools
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


TAB='    '

def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print('from chem.ccsd.uhf_ccsd import UHF_CCSD_Data')


def print_function_header(quantity: str, spin_subscript: str = '') -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    '''
    print(body)


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


def numpy_print_singles_singles(pq):
    block_name = 'singles_singles'
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
            'b': spin_mix[2],
            'j': spin_mix[3],
        }

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_function_header(
            quantity=block_name,
            spin_subscript=spin_subscript,
        )

        out_var = block_name + '_' + spin_subscript
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=tuple(spin_labels),
                update_val=out_var,
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")

        print(f'{TAB}return {out_var}')


def build_singles_doubles_block():
    """ Builds A _{ai bckj} """
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a_a
    pq.set_left_operators([['e1(i,a)']])  # (Replace i with a )*

    # commutator
    pq.add_st_operator(1.0, ['f', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(b,c,k,j)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e2(b,c,k,j)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(b,c,k,j)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def numpy_print_singles_doubles(pq):
    block_name = 'singles_doubles' 
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=6):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
            'b': spin_mix[2],
            'c': spin_mix[3],
            'k': spin_mix[4],
            'j': spin_mix[5],
        }

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_function_header(
            quantity=block_name,
            spin_subscript=spin_subscript,
        )

        out_var = block_name + '_' + spin_subscript
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=tuple(spin_labels),
                update_val=out_var,
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")

        print(f'{TAB}return {out_var}')


def build_doubles_doubles_block():
    """ Builds A _{abji cdlk} """
    pq = pdaggerq.pq_helper('fermi')

    # <mu| = <HF| a*_i a_a
    pq.set_left_operators([['e2(i,j,b,a)']])

    # commutator
    pq.add_st_operator(1.0, ['f', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'f'], ['t1', 't2'])

    pq.add_st_operator(1.0, ['v', 'e2(c,d,l,k)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(c,d,l,k)', 'v'], ['t1', 't2'])

    pq.simplify()

    return pq


def numpy_print_doubles_doubles(pq):
    block_name = 'doubles_doubles' 
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=8):
        spin_labels = {
            'a': spin_mix[0],
            'b': spin_mix[1],
            'j': spin_mix[5],
            'i': spin_mix[4],
            'c': spin_mix[2],
            'd': spin_mix[3],
            'l': spin_mix[7],
            'k': spin_mix[6],
        }

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_function_header(
            quantity=block_name,
            spin_subscript=spin_subscript,
        )

        out_var = block_name + '_' + spin_subscript
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=tuple(spin_labels),
                update_val=out_var,
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")

        print(f'{TAB}return {out_var}')


def main():

    do_singles_singles = False
    do_singles_doubles = False
    do_doubles_doubles = True

    if do_singles_singles:
        pq = build_singles_singles_block()
        numpy_print_singles_singles(pq)

    elif do_singles_doubles:
        pq = build_singles_doubles_block()
        numpy_print_singles_doubles(pq)

    elif do_doubles_doubles:
        pq = build_doubles_doubles_block()
        numpy_print_doubles_doubles(pq)


if __name__ == "__main__":
    main()
