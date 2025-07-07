import pickle
import pytest

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from rspn.ghf_ccsd.equations.cc_jacobian.singles_singles import (
    get_cc_j_singles_singles,
)
from rspn.ghf_ccsd.equations.cc_jacobian.singles_doubles import (
    get_cc_j_singles_doubles,
)
from rspn.ghf_ccsd.equations.cc_jacobian.doubles_singles import (
    get_cc_j_doubles_singles
)
from rspn.ghf_ccsd.equations.cc_jacobian.doubles_doubles import (
    get_cc_j_doubles_doubles
)


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


# this one takes too long for a regular test
# @pytest.mark.skip
def test_cc_jacobian_build():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    builders_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    print('Building the GHF CCSD Jacobian matrix.')

    print('Singles-Singles:')
    singles_singles = get_cc_j_singles_singles(**builders_input,)
    print('Ready.')
    print(f'Size = {humanify(singles_singles.nbytes)}.')  # 282 KiB
    print(f'Shape = {singles_singles.shape}')
    assert singles_singles.shape == (4, 10, 4, 10)
    print()
    print('Singles-Doubles:')
    singles_doubles = get_cc_j_singles_doubles(**builders_input,)
    print('Ready.')
    print(f'Size = {humanify(singles_doubles.nbytes)}.')  # 78.5 MiB
    print(f'Shape = {singles_doubles.shape}')
    assert singles_doubles.shape == (4, 10, 4, 4, 10, 10)
    print()
    print()
    print('Doubles-Singles:')
    doubles_singles = get_cc_j_doubles_singles(**builders_input,)
    print('Ready.')
    print(f'Size = {humanify(doubles_singles.nbytes)}.')  # 78.5 MiB
    print(f'Shape = {doubles_singles.shape}')  # (54150, 190)
    assert doubles_singles.shape == (4, 4, 10, 10, 4, 10,)
    print()
    print('Doubles-Doubles:')
    doubles_doubles = get_cc_j_doubles_doubles(**builders_input,)
    print('Ready.')
    print(f'Size = {humanify(doubles_doubles.nbytes)}.')  # ??? 23 GiB
    print(f'Shape = {doubles_doubles.shape}')  # (54150, 54150)
    assert doubles_doubles.shape == (4, 4, 10, 10, 4, 4, 10, 10,)
    print()


if __name__ == "__main__":
    test_cc_jacobian_build()
