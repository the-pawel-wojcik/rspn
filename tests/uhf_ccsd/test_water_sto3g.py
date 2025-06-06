import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._jacobian import (
    cc_jacobian_singles_singles,
    cc_jacobian_singles_doubles,
    cc_jacobian_doubles_singles,
    cc_jacobian_doubles_doubles,
    build_cc_jacobian,
)


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def test_cc_jacobian_build_helpers():
    with open('pickles/water_sto3g.pkl', 'rb') as bak_file:
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
    print(f'Size = {humanify(singles_singles.nbytes)}.')
    print(f'Shape = {singles_singles.shape}')
    assert singles_singles.nbytes == 3200
    assert singles_singles.shape == (20, 20)

    print()
    print('Singles-Doubles:')
    singles_doubles = cc_jacobian_singles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(singles_doubles.nbytes)}.')
    print(f'Shape = {singles_doubles.shape}')
    assert singles_doubles.nbytes == 96000
    assert singles_doubles.shape == (20, 600)

    print()
    print('Doubles-Singles:')
    doubles_singles = cc_jacobian_doubles_singles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(doubles_singles.nbytes)}.')
    print(f'Shape = {doubles_singles.shape}')
    assert doubles_singles.nbytes == 96000
    assert doubles_singles.shape == (600, 20)

    print()
    print('Doubles-Doubles:')
    doubles_doubles = cc_jacobian_doubles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(doubles_doubles.nbytes)}.')
    print(f'Shape = {doubles_doubles.shape}')
    assert doubles_doubles.nbytes == 2880000
    assert doubles_doubles.shape == (600, 600)


def test_cc_jacobian_build():
    with open('pickles/water_sto3g.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(uhf_scf_data=ccsd.scf_data, uhf_ccsd_data=ccsd.data,)
    builders_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    print('Build the whole CC Jacobian:')
    cc_jacobian = build_cc_jacobian(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    print('Ready.')
    print(f'Size = {humanify(cc_jacobian.nbytes)}.')
    print(f'Shape = {cc_jacobian.shape}')
    assert cc_jacobian.nbytes == 3075200
    assert cc_jacobian.shape == (620, 620)


def test_polarizability():
    with open('pickles/water_sto3g.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(uhf_scf_data=ccsd.scf_data, uhf_ccsd_data=ccsd.data,)
    pol = lr.find_polarizabilities()
    print(pol)


if __name__ == "__main__":
    test_cc_jacobian_build_helpers()
    test_cc_jacobian_build()
    test_polarizability()
