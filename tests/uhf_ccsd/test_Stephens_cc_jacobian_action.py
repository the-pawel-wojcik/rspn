import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.containers import Spin_MBE
from numpy.typing import NDArray
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from rspn.uhf_ccsd._jacobian_action_Stephen import (
    Minus_UHF_CCSD_Jacobian_action
)
import numpy as np


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
    energy_uhf_ccsd = ccsd.get_energy()

    minus_cc_Jacobian_action = Minus_UHF_CCSD_Jacobian_action(
        uhf_hf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
        cc_energy=energy_uhf_ccsd,
    )

    return minus_cc_Jacobian_action


def test_Jacobian_build_vs_Jacobian_action():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

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
        test_vector_mbe = Spin_MBE.from_flattened_NDArray(
            vector=test_vector, uhf_scf_data=ccsd.scf_data,
        )
        test_vector_mbe.pretty_print_mbe()
        # test_vector_mbe.doubles[E2_spin.baab] *= 0.0
        # test_vector_mbe.doubles[E2_spin.abba] *= 0.0
        # test_vector = test_vector_mbe.flatten()

        out_vector = - cc_jacobian @ test_vector
        sigma_brute_force = Spin_MBE.from_flattened_NDArray(
            vector=out_vector, uhf_scf_data=ccsd.scf_data,
        )
        sigma_brute_force.pretty_print_mbe()

        sigma_elegant = Spin_MBE.from_flattened_NDArray(
            vector=minus_cc_Jacobian_action.matvec(test_vector),
            uhf_scf_data=ccsd.scf_data,
        )
        sigma_elegant.pretty_print_mbe()
        break


def dont_test_Jacobian_action():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    m_cc_jacobian = Minus_UHF_CCSD_Jacobian_action(
        ccsd.scf_data, ccsd.data, ccsd.get_energy()
    )
    dims, _, _ = Spin_MBE.find_dims_slices_shapes(ccsd.scf_data)
    dim = Spin_MBE.get_vector_dim(dims)
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = m_cc_jacobian.matvec(test_vector)
        mbe_out = Spin_MBE.from_flattened_NDArray(out_vector, ccsd.scf_data)
        # mbe_out.pretty_print_mbe()
        # sigma_elegant = build_elegant_sigma(test_vector, ccsd, **kwargs)
        # sigma_elegant.pretty_print_mbe()

        # assert -sigma_elegant == mbe_out


if __name__ == "__main__":
    test_Jacobian_build_vs_Jacobian_action()
    # test_Jacobian_action()
