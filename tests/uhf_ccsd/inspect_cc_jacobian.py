from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from numpy.typing import NDArray
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
import numpy as np
import pytest


@pytest.fixture(scope='session')
def uhf_ccsd_jacobian_water_sto3g(uhf_ccsd_water_sto3g: UHF_CCSD) -> NDArray:
    kwargs = GeneratorsInput(
        uhf_scf_data=uhf_ccsd_water_sto3g.scf_data,
        uhf_ccsd_data=uhf_ccsd_water_sto3g.data,
    )
    lr_config = UHF_CCSD_LR_config(store_jacobian=True)
    lr = UHF_CCSD_LR(
        uhf_ccsd_water_sto3g.data,
        uhf_ccsd_water_sto3g.scf_data,
        lr_config,
    )
    cc_jacobian = build_cc_jacobian(kwargs=kwargs, dims=lr.assign_dims())
    return cc_jacobian


@pytest.mark.skip
def test_view_the_jacobian(uhf_ccsd_jacobian_water_sto3g: NDArray) -> None:
    cc_jacobian = uhf_ccsd_jacobian_water_sto3g
    # for id, eval in enumerate(np.sort(cc_jacobian.diagonal())):
    #     print(f'{id:>3}: {eval:.4f}')

    # off_diag = cc_jacobian - np.diag(cc_jacobian.diagonal())
    # for id, eval in enumerate(np.sort(off_diag.flatten())):
    #     print(f'{id:>3}: {eval:.4f}')

    print("Fifth column of the CC Jacobian:")
    single_column = cc_jacobian[:, 4]
    for id, eval in enumerate(single_column):
        val = f'{eval:-8.3f}' if abs(eval) > 1e-3 else ' ' * 8
        print(f'{id:>3}: {val}')


@pytest.mark.skip
def test_look_at_the_eigensystem(
    uhf_ccsd_jacobian_water_sto3g: NDArray,
) -> None:
    cc_jacobian = uhf_ccsd_jacobian_water_sto3g
    evals, evecs = np.linalg.eig(cc_jacobian)

    sorting_indices = np.argsort(evals)
    evals = evals[sorting_indices]
    evecs = evecs[:, sorting_indices]

    index = 0
    vector = evecs[:, index]
    print(f'{evals[index]=}')
    before = np.linalg.norm(vector)
    print(f'{before=}')
    after = np.linalg.norm(cc_jacobian @ vector)
    print(f'{after=}')
    print(f'{vector.T @ cc_jacobian @ vector=}')


@pytest.mark.skip
def test_show_part_of_jacobian(uhf_ccsd_jacobian_water_sto3g: NDArray) -> None:
    jacobian = uhf_ccsd_jacobian_water_sto3g

    with np.printoptions(precision=3, suppress=True):
        print("aa - aa")
        aa_aa = jacobian[0:10, 0:10]
        print(aa_aa)

        print("aa - bb")
        aa_bb = jacobian[0:10, 10:20]
        print(aa_bb)

        for idx in range(10):
            print(f"aa - aaaa part {idx}")
            print(jacobian[0:10, 20 + 10 * idx: 20 + 10 * (idx + 1)])

        for idx in range(10):
            print(f"aa - aaaa part {idx}")
            print(jacobian[0:10, 20 + 10 * idx: 20 + 10 * (idx + 1)])
