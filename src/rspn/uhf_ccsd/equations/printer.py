import itertools
from collections.abc import Iterable, Sequence
from pdaggerq.parser import contracted_strings_to_tensor_terms
from enum import Enum, auto

TAB = '    '


class DefineSections(Enum):
    FOCK = auto()
    FLUCTUATION = auto()
    IDENTITY = auto()
    SLICES = auto()
    CC_AMPS = auto()
    LAMBDA_AMPS = auto()


def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print('from chem.ccsd.uhf_ccsd import UHF_CCSD_Data')
    print('from chem.ccsd.containers import Spin_MBE, E1_spin, E2_spin')


def print_function_header(
    quantity: str,
    comment: str = '',
    spin_subscript: str = '',
    defines_exclude: set[DefineSections] | None = None,
    extra_definitions: None | Iterable[str] = None,
    extra_arguments: None | Iterable[str] = None,
) -> None:
    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,'''

    if extra_arguments is not None:
        for argument in extra_arguments:
            body += f'\n{TAB}{argument},'

    body += '''
) -> NDArray:'''

    if comment != '':
        body += f'''
    """ {comment} """'''

    if extra_definitions is not None:
        for definition in extra_definitions:
            body += f'\n{TAB}{definition}'

    if defines_exclude is None:
        defines_exclude = set()

    if DefineSections.FOCK not in defines_exclude:
        body += '''
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb'''

    if DefineSections.FLUCTUATION not in defines_exclude:
        body += '''
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb'''

    if DefineSections.IDENTITY not in defines_exclude:
        body += '''
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb'''

    if DefineSections.SLICES not in defines_exclude:
        body += '''
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob'''

    if DefineSections.CC_AMPS not in defines_exclude:
        body += '''
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb'''

    if DefineSections.LAMBDA_AMPS not in defines_exclude:
        body += '''
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb'''

    body += f'\n{TAB}'

    print(body)


def print_to_numpy(
    pq,
    tensor_name: str,
    tensor_subscripts: Sequence[str],
    defines_exclude: set[DefineSections] | None = None,
    extra_definitions: Iterable[str] | None = None,
    extra_arguments: Iterable[str] | None = None,
):
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
            comment=f'tensor_subscripts: {tensor_subscripts}',
            spin_subscript=spin_subscript,
            defines_exclude=defines_exclude,
            extra_definitions=extra_definitions,
            extra_arguments=extra_arguments,
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
