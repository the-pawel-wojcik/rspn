from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.meta.coordinates import Descartes
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd.equations.lHeecc.e1e1 import get_lhe1e1cc
from rspn.ghf_ccsd.equations.lHeecc.e1e2 import get_lhe1e2cc
from rspn.ghf_ccsd.equations.lHeecc.e2e1 import get_lhe2e1cc
from rspn.ghf_ccsd.equations.lHeecc.e2e2 import get_lhe2e2cc
# HINT: double counting
# from rspn.ghf_ccsd._jacobian import (
#     single_count_sd_and_ds,
#     single_count_doubles,
# )


def build_pol_xA_F_xB(
    kwargs: GHF_Generators_Input,
    t_res_B: dict[Descartes, NDArray],
    t_res_A: dict[Descartes, NDArray],
) -> Polarizability:
    f_e1_e1_raw = get_lhe1e1cc(**kwargs)
    f_e1_e2_raw = get_lhe1e2cc(**kwargs)
    f_e2_e1_raw = get_lhe2e1cc(**kwargs)
    f_e2_e2_raw = get_lhe2e2cc(**kwargs)

    nv = kwargs['ghf_data'].nv
    no = kwargs['ghf_data'].no
    f_e1_e1 = f_e1_e1_raw.reshape(nv*no, nv*no)

    # HINT: double counting
    # f_e1_e2, f_e2_e1 = single_count_sd_and_ds(
    #     raw_sd=f_e1_e2_raw,
    #     raw_ds=f_e2_e1_raw,
    #     nv=nv, no=no,
    # )
    # f_e2_e2 = single_count_doubles(f_e2_e2_raw, nv=nv, no=no)

    f_e1_e2 = f_e1_e2_raw.reshape((nv*no, nv**2*no**2))
    f_e2_e1 = f_e2_e1_raw.reshape((nv**2*no**2, nv*no))
    f_e2_e2 = f_e2_e2_raw.reshape((nv**2*no**2, nv**2*no**2))

    f_matrix = np.block([
        [f_e1_e1, f_e1_e2],
        [f_e2_e1, f_e2_e2],
    ])

    return Polarizability.from_builder(
        builder=lambda first, second: float(
            t_res_A[first].T @ f_matrix @ t_res_B[second]
        )
    )
