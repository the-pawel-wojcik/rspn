"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
import pytest


def test_eta_works():
    with open('pickles/water_ccpVDZ.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr_config = UHF_CCSD_LR_config(BUILD_JACOBIAN=True)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    eta_mu = lr._find_eta_mu()
    assert set(eta_mu) == {coord for coord in CARTESIAN}
    for key, val in eta_mu.items():
        assert set(val) == {
            'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
        }
        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)
        assert val['aaaa'].shape == (19, 19, 5, 5)


if __name__ == "__main__":
    test_eta_works()
