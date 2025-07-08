from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.meta.coordinates import Descartes
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd.equations.lHeecc.e1e1 import get_lhe1e1cc
from rspn.ghf_ccsd.equations.lHeecc.e1e2 import get_lhe1e2cc
from rspn.ghf_ccsd.equations.lHeecc.e2e1 import get_lhe2e1cc
from rspn.ghf_ccsd.equations.lHeecc.e2e2 import get_lhe2e2cc


def build_pol_xA_F_xB(
    kwargs: GHF_Generators_Input,
    t_res_B: dict[Descartes, dict[str, NDArray]],
    t_res_A: dict[Descartes, dict[str, NDArray]],
) -> Polarizability:
    f_e1_e1 = get_lhe1e1cc(**kwargs)
    f_e1_e2 = get_lhe1e2cc(**kwargs)
    f_e2_e1 = get_lhe2e1cc(**kwargs)
    f_e2_e2 = get_lhe2e2cc(**kwargs)

    return Polarizability.from_builder(
        builder=lambda first, second: (
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['singles'],
                f_e1_e1,
                t_res_B[second]['singles'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['singles'],
                f_e1_e2,
                t_res_B[second]['doubles'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['doubles'],
                f_e2_e1,
                t_res_B[second]['singles'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['doubles'],
                f_e2_e2,
                t_res_B[second]['doubles'],
            )
        )
    )
