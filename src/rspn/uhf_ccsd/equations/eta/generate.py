r""" Build a matrix defined in Eq. (77) of Ref. [1].

η _μ = <Λ| [V^{\omega _1}, tau_ν] | CC>

The operator tau _ν is an excitation operator that can be a single, double, and
higher order excitation operator. The operator V^{\omega _1} is a Fourier
transform of the interaction operator, see Ref. [2] for details and examples. In
case of a static electric field perturbation $-μE$, the operaotr
$V ^{\omega _1}$ reduces to the dipole operator $\mu$.


Refs:
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal of
Chemical Physics 93, 3333 (1990).
[2] J. Olsen and P. Jørgensen, Linear and nonlinear response functions for an
exact state and for an MCSCF state, The Journal of Chemical Physics 82, 3235
(1985).
"""
import itertools
from collections.abc import Sequence
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
    h_aa: NDArray,
    h_bb: NDArray,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    '''
    print(body)


def build_singles_block():
    """ Builds eta _{ai} """
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # commutator [A, \tau] -- A is a generic one-particle operator, i.e., the
    # dipole moment operator \mu
    pq.add_st_operator(1.0, ['h', 'e1(a,i)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e1(a,i)', 'h'], ['t1', 't2'])

    pq.simplify()

    return pq


def build_doubles_block():
    """ Builds eta _{abji} """
    pq = pdaggerq.pq_helper('fermi')

    # <bra| = <HF| (1 + l1 + l2)
    pq.set_left_operators([['1'], ['l1'], ['l2']])

    # commutator [A, \tau] -- A is a generic one-particle operator, i.e., the
    # dipole moment operator \mu
    pq.add_st_operator(1.0, ['h', 'e2(a,b,j,i)'], ['t1', 't2'])
    pq.add_st_operator(-1.0, ['e2(a,b,j,i)', 'h'], ['t1', 't2'])

    pq.simplify()

    return pq


def print_to_numpy(pq, tensor_name: str, tensor_subscripts: Sequence[str]):
    print_imports()
    subscripts_count = len(tensor_subscripts)
    for spin_mix in itertools.product(['a', 'b'], repeat=subscripts_count):
        spin_labels = {
            subscript: spin_mix[idx] for idx, subscript
            in enumerate(tensor_subscripts)
        }

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        spin_subscript = ''.join(spin_labels.values())
        print_function_header(
            quantity=tensor_name,
            spin_subscript=spin_subscript,
        )

        out_var = tensor_name + '_' + spin_subscript
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=tuple(tensor_subscripts),
                update_val=out_var,
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")

        print(f'{TAB}return {out_var}')


def main():

    do_singles = False
    do_doubles = True

    if do_singles:
        pq = build_singles_block()
        print_to_numpy(
            pq, tensor_name='eta', tensor_subscripts=('a', 'i'),
        )

    elif do_doubles:
        pq = build_doubles_block()
        print_to_numpy(
            pq, tensor_name='eta', tensor_subscripts=('a', 'b', 'j', 'i'),
        )


if __name__ == "__main__":
    main()
