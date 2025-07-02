from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE
from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import CARTESIAN, Descartes
from chem.meta.polarizability import Polarizability
from numpy.typing import NDArray
from rspn.uhf_ccsd.equations.lHtwcc.singles import (
    get_lHtauwCC_singles_aa,
    get_lHtauwCC_singles_bb,
)
from rspn.uhf_ccsd.equations.lHtwcc.doubles import (
    get_lHtauwCC_doubles_aaaa,
    get_lHtauwCC_doubles_abab,
    get_lHtauwCC_doubles_abba,
    get_lHtauwCC_doubles_baab,
    get_lHtauwCC_doubles_baba,
    get_lHtauwCC_doubles_bbbb,
)


def build_pol_xA_F_xB(
    kwargs: GeneratorsInput,
    t_res_B: dict[Descartes, dict[str, NDArray]],
    t_res_A: dict[Descartes, dict[str, NDArray]],
) -> Polarizability:

    fx = dict()
    for direction in CARTESIAN:
        fx[direction] = Spin_MBE()
        response = t_res_B[direction]
        for spin_block in E1_spin:
            fx[direction].singles[spin_block] = get_lHtauwCC_doubles_aaaa(
                **kwargs, response,
            )
        for spin_block in E2_spin:
            fx[direction].doubles[spin_block] = get_lHtauwCC_doubles_aaaa(
                **kwargs, response,
            )

    return Polarizability.from_builder(
        builder=lambda first, second: t_res_A[first] @ fx[second]
    )
