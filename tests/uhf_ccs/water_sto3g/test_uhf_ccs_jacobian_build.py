import pickle

from chem.ccs.uhf_ccs import UHF_CCS, UHF_CCS_InputPair
from chem.ccs.equations.util import UHF_CCS_InputPair
from rspn.uhf_ccs._jacobian import (
    cc_jacobian_singles_singles,
    build_cc_jacobian,
)
import numpy as np


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def test_cc_jacobian_build():
    with open('../pickles/water_uhf_ccs_lambda_sto3g.pkl', 'rb') as bak_file:
        ccs: UHF_CCS = pickle.load(bak_file)

    builders_input = UHF_CCS_InputPair(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
    )
    print('Building the UHF CCSD Jacobian matrix.')
    print('Singles-Singles:')
    scf_data = ccs.scf_data
    nmo = scf_data.nmo
    noa = scf_data.noa
    nob = scf_data.nob
    nva = nmo - noa
    nvb = nmo - nob
    dims = {
        'aa': nva * noa,
        'bb': nvb * nob,
    }
    singles_singles = cc_jacobian_singles_singles(
        kwargs=builders_input,
        dims=dims,
    )
    print('Ready.')
    print(f'UHF-CCS Jacobian size = {humanify(singles_singles.nbytes)}.')
    assert singles_singles.shape == (20, 20)

    with np.printoptions(precision=3, suppress=True):
        print(singles_singles)
    print(f'{
        np.allclose(
            np.zeros_like(singles_singles),
            singles_singles - singles_singles.T,
            atol=1e-6,
        )=}')
    

    cc_jacobian = build_cc_jacobian(kwargs=builders_input, dims=dims)
    assert np.allclose(cc_jacobian, singles_singles, atol=1e-6)


if __name__ == "__main__":
    test_cc_jacobian_build()
