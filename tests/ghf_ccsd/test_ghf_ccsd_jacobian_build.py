from contextlib import contextmanager
from itertools import product
from time import perf_counter

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from numpy.typing import NDArray
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


ALL_CLOSE_THRESH = 1e-9


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


# this one takes too long for a regular test
def test_cc_jacobian_build(water_sto3g: GHF_CCSD) -> None:
    builders_input = GHF_Generators_Input(
        ghf_data=water_sto3g.ghf_data,
        ghf_ccsd_data=water_sto3g.data,
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


def test_cc_jacobian_spectrum(water_sto3g: GHF_CCSD) -> None:
    builders_input = GHF_Generators_Input(
        ghf_data=water_sto3g.ghf_data,
        ghf_ccsd_data=water_sto3g.data,
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


def singles_mbidx_to_idx(sv: int, so: int) -> int:
    return sv * 10 + so


def doubles_mbidx_to_idx(dvl: int, dvr: int, dol:int, dor: int) -> int:
    return dor + 10 * dol + 100 * dvr + 400 * dvl + 40


def compare_singles(
    jacobian: NDArray,
    singles_singles: NDArray,
    verbose: bool = False,
) -> None:
    assert singles_singles.shape == (4, 10, 4, 10)
    # lsv = left singles virtual
    for lsv in range(4):
        for lso in range(10):
            for rsv in range(4):
                if verbose is True:
                    print(f'singles({lsv},{lso}) - singles({rsv}, :)')
                    left = singles_mbidx_to_idx(sv=lsv, so=lso)
                    right_start = singles_mbidx_to_idx(sv=rsv, so=0)
                    jacobian_block = jacobian[left,right_start:right_start+10] 
                    ss_block = singles_singles[lsv, lso, rsv, :]
                    assert np.allclose(
                        jacobian_block,
                        ss_block,
                        ALL_CLOSE_THRESH
                    )
                    if verbose is True:
                        jb_norm = np.linalg.norm(jacobian_block)
                        ssb_norm = np.linalg.norm(ss_block)
                        assert np.isclose(jb_norm, ssb_norm, ALL_CLOSE_THRESH)
                        print(f'  {jacobian_block}')
                        print(f'  {ss_block}')


def compare_sd(
    jacobian: NDArray,
    singles_doubles: NDArray,
    sv: int = 0,
    doubles_vv: tuple[int, int] = (0, 0),
    verbose: bool = False,
) -> None:
    """ Compare that the blocks
    singles_doubles[(sv, :), (doubles_vv[0], doubles_vv[1], :, :)]
    are the same in both the singles_doubles and the cc_jacobian matrices.
    In other words it makes sure that the mappings between the many-body index
    works. """
    dvl = doubles_vv[0]  # doubles virtual left
    dvr = doubles_vv[1]  # doubles virtual right
    for so in range(10):
        if verbose is True:
            print(f'{sv},{so} - doubles ({dvl},{dvr},:,:)')

        lft = singles_mbidx_to_idx(sv, so)
        for dol in range(10):
            right_start = doubles_mbidx_to_idx(dvl, dvr, dol, 0)
            jac_block = jacobian[lft, right_start:right_start + 10]
            sd_block = singles_doubles[sv,so,dvl,dvr,dol,:]
            jack_norm = np.linalg.norm(jac_block)
            if verbose is True and jack_norm > 1e-3:
                print(f'  {jac_block}')
                print(f'  {sd_block}')
            assert np.allclose(jac_block, sd_block, atol=ALL_CLOSE_THRESH)


def compare_ds(
    jacobian: NDArray,
    doubles_singles: NDArray,
    doubles_vv: tuple[int, int] = (0, 0),
    sv: int = 0,
    verbose: bool = False,
) -> None:
    assert doubles_singles.shape == (4, 4, 10, 10, 4, 10)
    dvl = doubles_vv[0]  # doubles virtual left
    dvr = doubles_vv[1]  # doubles virtual right
    for so in range(10):
        if verbose is True:
            print(f'doubles({dvl},{dvr},:,:) - singles({sv},{so})')

        right = singles_mbidx_to_idx(sv, so)
        for dol in range(10):
            left_start = doubles_mbidx_to_idx(dvl, dvr, dol, 0)
            jacobian_block = jacobian[left_start:left_start + 10, right]
            ds_block = doubles_singles[dvl, dvr, dol, :, sv, so]
            jacobian_block_norm = np.linalg.norm(jacobian_block)
            if verbose is True and jacobian_block_norm > 1e-3:
                print(f'  {jacobian_block}')
                print(f'  {ds_block}')
            assert np.allclose(jacobian_block, ds_block, atol=ALL_CLOSE_THRESH)


def compare_doubles(
    jacobian: NDArray,
    doubles_doubles: NDArray,
    verbose: bool = False,
) -> None:
    assert doubles_doubles.shape == (4, 4, 10, 10, 4, 4, 10, 10)
    # lvl = left virtual left
    # lor = left occupied right
    for lvl, lvr, lol, lor, rvl, rvr, rol in product(
        range(4), range(4), range(10), range(10), range(4), range(4), range(10)
    ):
        if verbose is True:
            print(f'doubles({lvl}, {lvr}, {lol}, {lor}) -', end='')
            print(f' doubles({rvl}, {rvr}, {rol}, :)')
        left = doubles_mbidx_to_idx(dvl=lvl, dvr=lvr, dol=lol, dor=lor)
        right_start = doubles_mbidx_to_idx(dvl=rvl, dvr=rvr, dol=rol, dor=0)
        jacobian_block = jacobian[left,right_start:right_start+10] 
        ss_block = doubles_doubles[lvl, lvr, lol, lor, rvl, rvr, rol, :]
        assert np.allclose(jacobian_block, ss_block, ALL_CLOSE_THRESH)
        if verbose is True:
            jb_norm = np.linalg.norm(jacobian_block)
            ssb_norm = np.linalg.norm(ss_block)
            assert np.isclose(jb_norm, ssb_norm, ALL_CLOSE_THRESH)
            print(f'  {jacobian_block}')
            print(f'  {ss_block}')


def test_cc_jacobian_to_NDArray_translation(water_sto3g: GHF_CCSD) -> None:
    builders_input = GHF_Generators_Input(
        ghf_data=water_sto3g.ghf_data,
        ghf_ccsd_data=water_sto3g.data,
    )

    with timeit('singles-singles build'):
        singles_singles = get_cc_j_singles_singles(**builders_input,)
    with timeit('singles-doubles build'):
        singles_doubles = get_cc_j_singles_doubles(**builders_input,)
    with timeit('doubles-singles build'):
        doubles_singles = get_cc_j_doubles_singles(**builders_input,)
    with timeit('doubles-doubles build'):
        doubles_doubles = get_cc_j_doubles_doubles(**builders_input,)

    ghf_data = water_sto3g.ghf_data
    no = ghf_data.no
    nv = ghf_data.nv
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
    with timeit('Checking singles-singles'):
        compare_singles(jacobian, singles_singles)
    for sv in range(4):
        for dvl in range(4):
            for dvr in range(4):
                compare_sd(jacobian, singles_doubles,
                           sv=sv, doubles_vv=(dvl, dvr))

    for sv in range(4):
        for dvl in range(4):
            for dvr in range(4):
                compare_ds(jacobian, doubles_singles,
                           sv=sv, doubles_vv=(dvl, dvr))

    with timeit('Checking doubles-doubles'):
        compare_doubles(jacobian, doubles_doubles)
