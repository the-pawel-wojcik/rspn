from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.spin_mbe import E1_spin, E2_spin, Spin_MBE
from chem.hf.util import turn_UHF_Data_to_UHF_ov_data
import numpy as np
from numpy.typing import NDArray
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
from chem.ccsd.uhf_ccsd import UHF_CCSD
import pytest
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
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


def lean_sigma_build(
    test_vector: NDArray,
    ccsd: UHF_CCSD,
    **kwargs,
) -> Spin_MBE:
    """
    Lean means that the CC Jacobian is not stored in the memory. Instead, the
    action of the CC Jacobian is calculated on the fly.
    """
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(ccsd.scf_data)
    mock_rhs = Spin_MBE.from_NDArray(test_vector, uhf_ov_data)
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


@pytest.fixture
def ingredients(
    uhf_ccsd_water_sto3g: UHF_CCSD
) -> tuple[NDArray, UHF_CCSD, GeneratorsInput]:
    ccsd = uhf_ccsd_water_sto3g

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
    return cc_jacobian, ccsd, kwargs


@pytest.mark.skip
def test_Jacobian_build_vs_Jacobian_action_on_random_vectors(
    ingredients: tuple[NDArray, UHF_CCSD, GeneratorsInput]
) -> None:
    cc_jacobian, ccsd, kwargs = ingredients
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(ccsd.scf_data)

    # Generate random test vectors
    TEST_VECTORS_COUNT = 10
    dim = cc_jacobian.shape[0]
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(TEST_VECTORS_COUNT))

    for test_vector in test_vectors:
        fast_sigma_ndarray = cc_jacobian @ test_vector
        fast_sigma = Spin_MBE.from_NDArray(fast_sigma_ndarray, uhf_ov_data)
        fast_sigma.pretty_print_mbe()

        lean_sigma = lean_sigma_build(test_vector, ccsd, **kwargs)
        lean_sigma.pretty_print_mbe()

        assert lean_sigma == fast_sigma


def test_Jacobian_build_vs_Jacobian_action_on_versors(
    ingredients: tuple[NDArray, UHF_CCSD, GeneratorsInput]
) -> None:
    cc_jacobian, ccsd, kwargs = ingredients
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(ccsd.scf_data)

    TEST_VERSORS_COUNT = 10
    dim = cc_jacobian.shape[0]
    test_versors = [
        np.zeros(shape=(dim)) for _ in range(TEST_VERSORS_COUNT)
    ]
    for idx, versor in enumerate(test_versors):
        versor[idx] = 1.0

    for versor in test_versors:
        fast_sigma_ndarray = cc_jacobian @ versor
        fast_sigma = Spin_MBE.from_NDArray(fast_sigma_ndarray, uhf_ov_data)
        for _ in E2_spin:
            fast_sigma.pretty_print_mbe()

            # with open(f'fast_{compared_block}.txt', 'w') as txt:
            #     print("Fast sigma", file=txt)
            #     fast_sigma.pretty_print_doubles_block(compared_block, file=txt)

            lean_sigma = lean_sigma_build(versor, ccsd, **kwargs)
            lean_sigma.pretty_print_mbe()

            # with open(f'lean_{compared_block}.txt', 'w') as txt:
            #     print("Lean sigma", file=txt)
            #     lean_sigma.pretty_print_doubles_block(compared_block, file=txt)

        # assert fast_sigma == lean_sigma
        break


@pytest.mark.skip
def test_Jacobian_action(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    ccsd = uhf_ccsd_water_sto3g
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(ccsd.scf_data)
    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    m_cc_jacobian = Minus_UHF_CCSD_Jacobian_action(ccsd.scf_data, ccsd.data)
    dim = uhf_ov_data.get_vector_dim()
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = m_cc_jacobian.matvec(test_vector)
        mbe_out = Spin_MBE.from_NDArray(out_vector, uhf_ov_data)
        # mbe_out.pretty_print_mbe()
        sigma_elegant = lean_sigma_build(test_vector, ccsd, **kwargs)
        # sigma_elegant.pretty_print_mbe()

        assert -sigma_elegant == mbe_out
