import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_cc_mu():
    with open('pickles/water_ccpVDZ.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr_config = UHF_CCSD_LR_config(store_jacobian=True)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    input = GeneratorsInput(
        uhf_scf_data=lr.uhf_scf_data,
        uhf_ccsd_data=lr.uhf_ccsd_data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    # TODO: Test the values
    assert set(cced_interation_op) == {coord for coord in CARTESIAN}
    for _, val in cced_interation_op.items():
        assert set(val) == {
            'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
        }

        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)
        assert val['aaaa'].shape == (19, 19, 5, 5)
        assert val['abab'].shape == (19, 19, 5, 5)
        assert val['abba'].shape == (19, 19, 5, 5)
        assert val['baab'].shape == (19, 19, 5, 5)
        assert val['baba'].shape == (19, 19, 5, 5)
        assert val['bbbb'].shape == (19, 19, 5, 5)


if __name__ == "__main__":
    test_cc_mu()
