from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_cc_mu(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    input = GHF_Generators_Input(
        ghf_data=ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    assert set(cced_interation_op) == {coord for coord in CARTESIAN}
    for _, val in cced_interation_op.items():
        assert set(val) == {'singles', 'doubles',}

        assert val['singles'].shape == (4, 10)
        assert val['doubles'].shape == (4, 4, 10, 10)
