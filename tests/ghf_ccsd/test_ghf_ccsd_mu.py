from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.meta.coordinates import Descartes
import numpy as np
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_cc_mu(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    input = GHF_Generators_Input(
        ghf_data=ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    assert set(cced_interation_op) == {coord for coord in Descartes}
    for direction, val in cced_interation_op.items():
        assert set(val) == {'singles', 'doubles',}

        assert val['singles'].shape == (4, 10)
        assert val['doubles'].shape == (4, 4, 10, 10)
        np.set_printoptions(precision=6, suppress=True)
        print()
        print(direction)
        # ghf_data = ghf_ccsd_water_sto3g.ghf_data
        # o = ghf_data.o
        # v = ghf_data.v
        # mu = ghf_data.mu[direction]
        # print('mu[o, v]')
        # print(mu[o, v])
