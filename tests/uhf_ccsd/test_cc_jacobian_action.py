import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE
from numpy.typing import NDArray
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from rspn.uhf_ccsd.equations.cc_jacobian_contract_Rain.singles import (
    get_cc_j_w_singles_aa,
    get_cc_j_w_singles_bb,
)
import numpy as np


BLOCKS = [ 'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb', ]


def get_singles_dim(dims: dict[str, int]) -> int:
    return sum(dims[block] for block in E1_spin)


def get_doubles_dim(dims: dict[str, int]) -> int:
    return sum(dims[block] for block in E2_spin)


def get_vector_dim(dims: dict[str, int]) -> int:
    singles_dim = get_singles_dim(dims)
    doubles_dim = get_doubles_dim(dims)
    return singles_dim + doubles_dim


def vector_to_mbe(vector: NDArray, dims: dict[str, int]) -> Spin_MBE:

    assert len(vector.shape) == 1
    assert vector.shape[0] == get_vector_dim(dims)

    mbe = Spin_MBE()
    dim_sum = 0
    for block in E1_spin:
        block_dim = dims[block]
        mbe.singles[block] = vector[dim_sum: dim_sum + block_dim]
        dim_sum += block_dim

    for block in E2_spin:
        block_dim = dims[block]
        mbe.doubles[block] = vector[dim_sum: dim_sum + block_dim]
        dim_sum += block_dim

    return mbe


def test_CC_Jacobian_action():
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
        dims=lr.assign_dims(),
    )

    # Generate ten test vectors
    dim = cc_jacobian.shape[0]
    today = 20250623
    rng = np.random.default_rng(seed=today)
    test_vectors = (rng.random(size=(dim)) for _ in range(10))

    for test_vector in test_vectors:
        out_vector = cc_jacobian @ test_vector
        sigma_brute_force = vector_to_mbe(out_vector, dims)
        sigma_brute_force.pretty_print_mbe()
        mock_rhs = vector_to_mbe(test_vector, dims)
        sigma_elegant = Spin_MBE()
        sigma_elegant.singles[E1_spin.aa] = get_cc_j_w_singles_aa(
            **kwargs,
            vector=mock_rhs,
        )
        sigma_elegant.singles[E1_spin.bb] = get_cc_j_w_singles_bb(
            **kwargs,
            vector=mock_rhs,
        )
        break


if __name__ == "__main__":
    test_CC_Jacobian_action()
