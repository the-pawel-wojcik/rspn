""" Transform the HF dipole operator matrix elements to the CC dipole operators
matrix elements. """
import itertools
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms
from pdaggerq.algebra import TensorTerm
from rspn.uhf_ccsd.equations.generate import (
    print_imports,
    TAB,
)

CART = {'x', 'y', 'z'}
def print_dipole_gen_header(
    component: str,
    spin_subscript: str = '',
) -> None:
    if component not in CART:
        raise ValueError(f'The component has to be one of {CART}.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_mu{component}{spin_subscript}(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_{component}
    h_bb = uhf_scf_data.mub_{component}
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


def do_singles(component: str):
    if component not in CART:
        raise ValueError(f'The component has to be one of {CART}.')
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['a*(i)', 'a(a)', 'h'], ['t1', 't2'])
    pq.simplify()

    block_name = 'dipole_singles'
    for spin_mix in itertools.product(['a', 'b'], repeat=2):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
        }
        strings = pq.strings(spin_labels=spin_labels)
        terms: list[TensorTerm] = contracted_strings_to_tensor_terms(strings)
        if len(terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_dipole_gen_header(
            component=component,
            spin_subscript=spin_subscript,
        )

        out_var = block_name + '_' + spin_subscript
        for term in terms:
            einsum_term = term.einsum_string(
                output_variables=tuple(spin_labels),
                update_val=out_var,
            )
            print(f'{TAB}{einsum_term}')

        print(f'{TAB}return {out_var}')


def do_doubles(component: str):
    if component not in CART:
        raise ValueError(f'The component has to be one of {CART}.')
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['e2(i,j,b,a)', 'h'], ['t1', 't2'])
    pq.simplify()

    block_name = 'dipole_doubles'
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
        spin_labels = {
            'a': spin_mix[0],
            'b': spin_mix[1],
            'j': spin_mix[2],
            'i': spin_mix[3],
        }
        strings = pq.strings(spin_labels=spin_labels)
        terms: list[TensorTerm] = contracted_strings_to_tensor_terms(strings)
        if len(terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_dipole_gen_header(
            component=component,
            spin_subscript=spin_subscript,
        )

        out_var = block_name + '_' + spin_subscript
        for term in terms:
            einsum_term = term.einsum_string(
                output_variables=tuple(spin_labels),
                update_val=out_var,
            )
            print(f'{TAB}{einsum_term}')

        print(f'{TAB}return {out_var}')


if __name__ == "__main__":
    # Print
    print_imports()
    singles_time = True
    doubles_time = False
    if singles_time:
        for component in CART:
            do_singles(component)

    elif doubles_time:
        for component in CART:
            do_doubles(component)
