from chem.meta.spin_mbe import E1_spin, E2_spin, Spin_MBE
from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import CARTESIAN, Descartes
from chem.meta.polarizability import Polarizability
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
    t_res_B: dict[Descartes, Spin_MBE],
    t_res_A: dict[Descartes, Spin_MBE],
) -> Polarizability:

    fx = dict()
    for direction in CARTESIAN:
        fx[direction] = Spin_MBE()
        response_A = t_res_B[direction]
        fx[direction].singles[E1_spin.aa] = get_lHtauwCC_singles_aa(
            **kwargs, vector=response_A,
        )
        fx[direction].singles[E1_spin.bb] = get_lHtauwCC_singles_bb(
            **kwargs, vector=response_A,
        )

        fx[direction].doubles[E2_spin.aaaa] = get_lHtauwCC_doubles_aaaa(
            **kwargs, vector=response_A,
        )
        fx[direction].doubles[E2_spin.abab] = get_lHtauwCC_doubles_abab(
            **kwargs, vector=response_A,
        )
        fx[direction].doubles[E2_spin.abba] = get_lHtauwCC_doubles_abba(
            **kwargs, vector=response_A,
        )
        fx[direction].doubles[E2_spin.baab] = get_lHtauwCC_doubles_baab(
            **kwargs, vector=response_A,
        )
        fx[direction].doubles[E2_spin.baba] = get_lHtauwCC_doubles_baba(
            **kwargs, vector=response_A,
        )
        fx[direction].doubles[E2_spin.bbbb] = get_lHtauwCC_doubles_bbbb(
            **kwargs, vector=response_A,
        )

    return Polarizability.from_builder(
        builder=lambda first, second: t_res_A[first] @ fx[second]
    )
