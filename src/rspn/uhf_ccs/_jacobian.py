from chem.ccs.equations.util import UHF_CCS_InputPair
import numpy as np
from numpy.typing import NDArray
from rspn.uhf_ccs.equations.cc_jacobian.singles_singles import (
    get_cc_j_singles_singles_aaaa,
    get_cc_j_singles_singles_aabb,
    get_cc_j_singles_singles_bbaa,
    get_cc_j_singles_singles_bbbb,
)


def build_cc_jacobian(
    kwargs: UHF_CCS_InputPair,
    dims: dict[str, int],
) -> NDArray:
    singles_singles = cc_jacobian_singles_singles(kwargs=kwargs, dims=dims)
    return singles_singles


def cc_jacobian_singles_singles(
    kwargs: UHF_CCS_InputPair,
    dims: dict[str, int],
) -> NDArray:
    dim_aa = dims['aa']
    dim_bb = dims['bb']
    aa_aa = get_cc_j_singles_singles_aaaa(**kwargs).reshape(dim_aa, dim_aa)
    aa_bb = get_cc_j_singles_singles_aabb(**kwargs).reshape(dim_aa, dim_bb)
    bb_aa = get_cc_j_singles_singles_bbaa(**kwargs).reshape(dim_bb, dim_aa)
    bb_bb = get_cc_j_singles_singles_bbbb(**kwargs).reshape(dim_bb, dim_bb)

    jacobian_singles_singles = np.block([
        [aa_aa, aa_bb,],
        [bb_aa, bb_bb,],
    ])

    return jacobian_singles_singles
