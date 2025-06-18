from chem.ccs.equations.util import UHF_CCS_InputTriple
from chem.meta.coordinates import Descartes
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from rspn.uhf_ccs.equations.lHeecc.e1e1 import (
    get_lhe1e1cc_aaaa,
    get_lhe1e1cc_aabb,
    get_lhe1e1cc_bbaa,
    get_lhe1e1cc_bbbb,
)


def build_pol_xA_F_xB(
    kwargs: UHF_CCS_InputTriple,
    t_res_B: dict[Descartes, dict[str, NDArray]],
    t_res_A: dict[Descartes, dict[str, NDArray]],
) -> Polarizability:
    f_aa_aa = get_lhe1e1cc_aaaa(**kwargs)
    f_aa_bb = get_lhe1e1cc_aabb(**kwargs)
    f_bb_aa = get_lhe1e1cc_bbaa(**kwargs)
    f_bb_bb = get_lhe1e1cc_bbbb(**kwargs)

    return Polarizability.from_builder(
        builder=lambda first, second: (
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['aa'],
                f_aa_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['aa'],
                f_aa_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['bb'],
                f_bb_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['bb'],
                f_bb_bb,
                t_res_B[second]['bb'],
            )
        )
    )
