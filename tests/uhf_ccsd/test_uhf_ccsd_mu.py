import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_cc_mu():
    with open('pickles/uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    eta_mu = lr.build_cc_electric_dipole_singles()
    # TODO: Test the mu values
    assert set(eta_mu) == {coord for coord in CARTESIAN}
    for _, val in eta_mu.items():
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
