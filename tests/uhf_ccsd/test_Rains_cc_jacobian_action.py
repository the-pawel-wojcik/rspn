from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.hf.util import turn_UHF_Data_to_UHF_ov_data
from chem.meta.spin_mbe import Spin_MBE
import numpy as np
from numpy.typing import NDArray
import pytest
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config


def get_cc_jacobian(ccsd: UHF_CCSD) -> NDArray:
    energy_uhf_ccsd =  ccsd.get_energy()
    print(f'{energy_uhf_ccsd=}')
    lr_config = UHF_CCSD_LR_config(store_jacobian=True)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    dims = lr.assign_dims()
    cc_jacobian = build_cc_jacobian(
        kwargs=kwargs,
        dims=dims,
    )
    return cc_jacobian


def get_cc_jacobian_action(ccsd: UHF_CCSD) -> Minus_UHF_CCSD_Jacobian_action:
    minus_cc_Jacobian_action = Minus_UHF_CCSD_Jacobian_action(
        uhf_hf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    return minus_cc_Jacobian_action


@pytest.mark.skip
def test_Jacobian_build_vs_Jacobian_action(
    uhf_ccsd_water_sto3g: UHF_CCSD,
) -> None :
    ccsd = uhf_ccsd_water_sto3g
    cc_jacobian = get_cc_jacobian(ccsd)
    minus_cc_Jacobian_action = get_cc_jacobian_action(ccsd)

    # Generate a set of random test vectors
    TEST_VECTORS_COUNT = 10
    today = 20250623
    rng = np.random.default_rng(seed=today)
    dim = cc_jacobian.shape[0]
    test_vectors = (rng.random(size=(dim)) for _ in range(TEST_VECTORS_COUNT))

    for test_vector in test_vectors:
        # trim out the part that is not touched by the action
        uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(ccsd.scf_data)
        test_vector_mbe = Spin_MBE.from_NDArray(test_vector, uhf_ov_data)
        test_vector_mbe.pretty_print_mbe()
        # test_vector_mbe.doubles[E2_spin.baab] *= 0.0
        # test_vector_mbe.doubles[E2_spin.abba] *= 0.0
        # test_vector = test_vector_mbe.flatten()

        out_vector = - cc_jacobian @ test_vector
        sigma_brute_force = Spin_MBE.from_NDArray(out_vector, uhf_ov_data)
        sigma_brute_force.pretty_print_mbe()

        sigma_elegant = minus_cc_Jacobian_action @ test_vector
        sigma_elegant_mbe = Spin_MBE.from_NDArray(sigma_elegant, uhf_ov_data)
        sigma_elegant_mbe.pretty_print_mbe()
        break


def test_Jacobian_action_object(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    minus_cc_Jacobian_action = get_cc_jacobian_action(uhf_ccsd_water_sto3g)
    
    assert hasattr(minus_cc_Jacobian_action, 'dtype')
    assert hasattr(minus_cc_Jacobian_action, 'shape')
    assert len(minus_cc_Jacobian_action.shape) == 2
