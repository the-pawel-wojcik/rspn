from chem.ccs.equations.util import UHF_CCS_InputPair
from chem.ccs.uhf_ccs import UHF_CCS
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccs._nuOpCC import build_nu_bar_V_cc


def test_cc_mu(uhf_ccs_water_ccpVDZ: UHF_CCS) -> None:
    input = UHF_CCS_InputPair(
        uhf_data=uhf_ccs_water_ccpVDZ.scf_data,
        uhf_ccs_data=uhf_ccs_water_ccpVDZ.data,
    )
    cc_interation_operator = build_nu_bar_V_cc(input=input)
    # TODO: Test the values
    assert set(cc_interation_operator) == {coord for coord in CARTESIAN}
    for _, val in cc_interation_operator.items():
        assert set(val) == {'aa', 'bb'}
        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)
