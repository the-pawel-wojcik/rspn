import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE
from numpy.typing import NDArray
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from rspn.uhf_ccsd.equations.cc_jacobian_contract_Rain.singles import (
    get_cc_j_w_singles_aa,
    get_cc_j_w_singles_bb,
)
from rspn.uhf_ccsd.equations.cc_jacobian_contract_Rain.doubles import (
    get_cc_j_w_doubles_aaaa,
    get_cc_j_w_doubles_abab,
    get_cc_j_w_doubles_abba,
    get_cc_j_w_doubles_baab,
    get_cc_j_w_doubles_baba,
    get_cc_j_w_doubles_bbbb,
)
import numpy as np


def build_elegant_sigma(
    test_vector: NDArray,
    ccsd: UHF_CCSD,
    **kwargs,
) -> Spin_MBE:
    mock_rhs = Spin_MBE.from_flattened_NDArray(test_vector, ccsd.scf_data)
    sigma_elegant = Spin_MBE()
    sigma_elegant.singles[E1_spin.aa] = get_cc_j_w_singles_aa(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.singles[E1_spin.bb] = get_cc_j_w_singles_bb(
        **kwargs,
        vector=mock_rhs,
    )

    sigma_elegant.doubles[E2_spin.aaaa] = get_cc_j_w_doubles_aaaa(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.aaaa] = get_cc_j_w_doubles_aaaa(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.abab] = get_cc_j_w_doubles_abab(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.abba] = get_cc_j_w_doubles_abba(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.baab] = get_cc_j_w_doubles_baab(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.baba] = get_cc_j_w_doubles_baba(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_elegant.doubles[E2_spin.bbbb] = get_cc_j_w_doubles_bbbb(
        **kwargs,
        vector=mock_rhs,
    )

    return sigma_elegant


def test_Jacobian_build_vs_Jacobian_action():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    dims = lr.assign_dims()
    cc_jacobian = build_cc_jacobian(
        kwargs=kwargs,
        dims=dims,
    )

    # Generate ten test vectors
    dim = cc_jacobian.shape[0]
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = cc_jacobian @ test_vector
        sigma_brute_force = Spin_MBE.from_flattened_NDArray(
            vector=out_vector, uhf_scf_data=ccsd.scf_data,
        )
        # sigma_brute_force.pretty_print_mbe()

        sigma_elegant = build_elegant_sigma(test_vector, ccsd, **kwargs)
        # sigma_elegant.pretty_print_mbe()

        # The Rain's approach does not work!
        assert not sigma_elegant == sigma_brute_force


def test_Jacobian_action():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    m_cc_jacobian = Minus_UHF_CCSD_Jacobian_action(ccsd.scf_data, ccsd.data)
    dims, _, _ = Spin_MBE.find_dims_slices_shapes(ccsd.scf_data)
    dim = Spin_MBE.get_vector_dim(dims)
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = m_cc_jacobian.matvec(test_vector)
        mbe_out = Spin_MBE.from_flattened_NDArray(out_vector, ccsd.scf_data)
        # mbe_out.pretty_print_mbe()
        sigma_elegant = build_elegant_sigma(test_vector, ccsd, **kwargs)
        # sigma_elegant.pretty_print_mbe()

        assert -sigma_elegant == mbe_out


if __name__ == "__main__":
    test_Jacobian_build_vs_Jacobian_action()
    test_Jacobian_action()
