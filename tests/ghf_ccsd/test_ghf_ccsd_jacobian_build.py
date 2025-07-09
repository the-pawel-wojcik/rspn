from contextlib import contextmanager
import pickle
import pytest
from time import perf_counter

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from rspn.ghf_ccsd.ghf_ccsd_lr import build_cc_jacobian
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
@pytest.mark.skip
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


@pytest.mark.skip
def test_cc_jacobian_spectrum():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    builders_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )

    cc_jacobian = build_cc_jacobian(builders_input)
    eigenvalues = np.linalg.eigvals(cc_jacobian)
    for eval in sorted(eigenvalues, key=lambda z: abs(z)):
        print(f'{eval.real:+12.6f}', end='')
        if abs(eval.imag) > 1e-6:
            print(f'{eval.real:+12.6f}i')
        else:
            print('')


@contextmanager
def timeit(header: str=''):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f'[{end-start:.3f} s] {header} time.')


def test_cc_jacobian_to_NDArray_translation():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    builders_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )

    with timeit('singles-singles build'):
        singles_singles = get_cc_j_singles_singles(**builders_input,)
    with timeit('singles-doubles build'):
        singles_doubles = get_cc_j_singles_doubles(**builders_input,)
    with timeit('doubles-singles build'):
        doubles_singles = get_cc_j_doubles_singles(**builders_input,)
    with timeit('doubles-doubles build'):
        doubles_doubles = get_cc_j_doubles_doubles(**builders_input,)

    no = ccsd.ghf_data.no
    nv = ccsd.ghf_data.nv
    dim_s = nv * no
    dim_d = nv * nv * no * no

    with timeit('reshape'):
        jacobian = np.block([
            [
                singles_singles.reshape(dim_s, dim_s),
                singles_doubles.reshape(dim_s, dim_d),
            ],
            [
                doubles_singles.reshape(dim_d, dim_s),
                doubles_doubles.reshape(dim_d, dim_d),
            ],
        ])

    np.set_printoptions(precision=3, suppress=True)
    print(f'{singles_singles.shape=}')
    print('0,0 - singles')
    print(f'  {jacobian[0,0:10]}\n  {singles_singles[0,0,0,:]}')
    print(f'  {jacobian[0,10:20]}\n  {singles_singles[0,0,1,:]}')
    print(f'  {jacobian[0,20:30]}\n  {singles_singles[0,0,2,:]}')
    print(f'  {jacobian[0,30:40]}\n  {singles_singles[0,0,3,:]}')
    print('0,1 - singles')
    print(f'  {jacobian[1,0:10]}\n  {singles_singles[0,1,0,:]}')
    print(f'  {jacobian[1,10:20]}\n  {singles_singles[0,1,1,:]}')
    print(f'  {jacobian[1,20:30]}\n  {singles_singles[0,1,2,:]}')
    print(f'  {jacobian[1,30:40]}\n  {singles_singles[0,1,3,:]}')
    print('3,9 - singles')
    print(f'  {jacobian[39,0:10]}\n  {singles_singles[3,9,0,:]}')
    print(f'  {jacobian[39,10:20]}\n  {singles_singles[3,9,1,:]}')
    print(f'  {jacobian[39,20:30]}\n  {singles_singles[3,9,2,:]}')
    print(f'  {jacobian[39,30:40]}\n  {singles_singles[3,9,3,:]}')


if __name__ == "__main__":
    test_cc_jacobian_build()
