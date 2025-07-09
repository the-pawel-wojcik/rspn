import pickle

from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_cc_mu():
    # TODO: Test the values
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    assert set(cced_interation_op) == {coord for coord in CARTESIAN}
    for _, val in cced_interation_op.items():
        assert set(val) == {'ref', 'singles', 'doubles',}

        assert val['ref'].ndim == 0
        assert val['singles'].shape == (4, 10)
        assert val['doubles'].shape == (4, 4, 10, 10)


if __name__ == "__main__":
    test_cc_mu()
