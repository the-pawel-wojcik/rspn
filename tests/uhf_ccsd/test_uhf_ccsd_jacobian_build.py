import pickle
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._jacobian import (
    cc_jacobian_singles_singles,
    cc_jacobian_singles_doubles,
    cc_jacobian_doubles_singles,
    cc_jacobian_doubles_doubles,
)


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def test_cc_jacobian_build():
    with open('pickles/uhf_ccsd_lambda.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(uhf_scf_data=ccsd.scf_data, uhf_ccsd_data=ccsd.data,)
    builders_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    print('Building the UHF CCSD Jacobian matrix.')

    print('Singles-Singles:')
    singles_singles = cc_jacobian_singles_singles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(singles_singles.nbytes)}.')  # 282 KiB
    print(f'Shape = {singles_singles.shape}')  # (190, 190)

    print()
    print('Singles-Doubles:')
    singles_doubles = cc_jacobian_singles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(singles_doubles.nbytes)}.')  # 78.5 MiB
    print(f'Shape = {singles_doubles.shape}')  # (190, 54150)

    print()
    print('Doubles-Singles:')
    doubles_singles = cc_jacobian_doubles_singles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(doubles_singles.nbytes)}.')  # 78.5 MiB
    print(f'Shape = {doubles_singles.shape}')  # (54150, 190)

    print()
    print('Doubles-Doubles:')
    doubles_doubles = cc_jacobian_doubles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(doubles_doubles.nbytes)}.')  # ??? 23 GiB
    print(f'Shape = {doubles_doubles.shape}')  # (54150, 54150)


if __name__ == "__main__":
    test_cc_jacobian_build()
