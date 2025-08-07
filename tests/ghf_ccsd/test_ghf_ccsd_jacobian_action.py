from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE, GHF_ov_data
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._jacobian_action import Minus_GHF_CCSD_Jacobian_action
from chem.ccsd.ghf_ccsd import GHF_CCSD
import pytest
from rspn.ghf_ccsd.equations.cc_jacobian_action.singles import (
    get_cc_j_w_singles,
)
from rspn.ghf_ccsd.equations.cc_jacobian_action.doubles import (
    get_cc_j_w_doubles,
)


def lean_sigma_build(
    test_vector: NDArray,
    ccsd: GHF_CCSD,
    **kwargs,
) -> GHF_CCSD_MBE:
    """ Lean means that the CC Jacobian is not stored in the memory. Instead,
    the action of the CC Jacobian is calculated on the fly. """
    nmo = ccsd.ghf_data.nmo
    no = ccsd.ghf_data.no
    nv = ccsd.ghf_data.nv
    ghf_ov_data = GHF_ov_data(nmo=nmo, no=no, nv=nv)
    mock_rhs = GHF_CCSD_MBE.from_NDArray(test_vector, ghf_ov_data)
    sigma_singles = get_cc_j_w_singles(
        **kwargs,
        vector=mock_rhs,
    )
    sigma_doubles = get_cc_j_w_doubles(
        **kwargs,
        vector=mock_rhs,
    )
    action = GHF_CCSD_MBE(
        singles=sigma_singles,
        doubles=sigma_doubles,
    )
    return action


@pytest.fixture
def ingredients(
    ghf_ccsd_water_sto3g: GHF_CCSD,
) -> tuple[NDArray, GHF_CCSD, GHF_Generators_Input]:
    ccsd = ghf_ccsd_water_sto3g
    kwargs = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(
        kwargs=kwargs,
    )
    return cc_jacobian, ccsd, kwargs


@pytest.mark.skip
def test_Jacobian_build_vs_Jacobian_action_on_random_vectors(
    ingredients: tuple[NDArray, GHF_CCSD, GHF_Generators_Input],
) -> None:
    cc_jacobian, ccsd, kwargs = ingredients
    nmo = ccsd.ghf_data.nmo
    no = ccsd.ghf_data.no
    nv = ccsd.ghf_data.nv
    ghf_ov_data = GHF_ov_data(nmo=nmo, no=no, nv=nv)

    # Generate random test vectors
    TEST_VECTORS_COUNT = 10
    dim = cc_jacobian.shape[0]
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(TEST_VECTORS_COUNT))

    for test_vector in test_vectors:
        fast_sigma_ndarray = cc_jacobian @ test_vector
        fast_sigma = GHF_CCSD_MBE.from_NDArray(fast_sigma_ndarray, ghf_ov_data)
        fast_sigma.pretty_print_mbe()

        lean_sigma = lean_sigma_build(test_vector, ccsd, **kwargs)
        lean_sigma.pretty_print_mbe()

        assert lean_sigma == fast_sigma


@pytest.mark.skip
def test_Jacobian_build_vs_Jacobian_action_on_versors(
    ingredients: tuple[NDArray, GHF_CCSD, GHF_Generators_Input],
) -> None:
    cc_jacobian, ccsd, kwargs = ingredients
    nmo = ccsd.ghf_data.nmo
    no = ccsd.ghf_data.no
    nv = ccsd.ghf_data.nv
    ghf_ov_data = GHF_ov_data(nmo, no, nv)

    # Generate random test vectors
    TEST_VERSORS_COUNT = 10
    dim = cc_jacobian.shape[0]
    test_versors = [
        np.zeros(shape=(dim)) for _ in range(TEST_VERSORS_COUNT)
    ]
    for idx, versor in enumerate(test_versors):
        versor[idx] = 1.0

    for test_vector in test_versors:
        fast_sigma_ndarray = cc_jacobian @ test_vector
        fast_sigma = GHF_CCSD_MBE.from_NDArray(fast_sigma_ndarray, ghf_ov_data)
        fast_sigma.pretty_print_mbe()

        lean_sigma = lean_sigma_build(test_vector, ccsd, **kwargs)
        lean_sigma.pretty_print_mbe()

        assert lean_sigma == fast_sigma


@pytest.mark.skip
def test_Jacobian_action(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    nmo = ccsd.ghf_data.nmo
    no = ccsd.ghf_data.no
    nv = ccsd.ghf_data.nv
    ghf_ov_data = GHF_ov_data(nmo, no, nv)
    kwargs = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    m_cc_jacobian = Minus_GHF_CCSD_Jacobian_action(ccsd.ghf_data, ccsd.data)
    dim = m_cc_jacobian.shape[0]
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = m_cc_jacobian.matvec(test_vector)
        mbe_out = GHF_CCSD_MBE.from_NDArray(out_vector, ghf_ov_data)
        # mbe_out.pretty_print_mbe()
        sigma_elegant = lean_sigma_build(test_vector, ccsd, **kwargs)
        # sigma_elegant.pretty_print_mbe()

        assert -sigma_elegant == mbe_out
