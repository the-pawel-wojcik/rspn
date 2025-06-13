"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccs.uhf_ccs import UHF_CCS
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccs.uhf_ccs_lr import UHF_CCS_LR


def test_eta_works():
    with open('pickles/water_uhf_ccs_lambda_ccpVDZ.pkl','rb') as bak_file:
        ccs: UHF_CCS = pickle.load(bak_file)
    lr = UHF_CCS_LR(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
        uhf_ccs_lambda_data=ccs.cc_lambda_data,
    )
            
    eta_mu = lr._find_eta_mu()
    assert set(eta_mu) == {coord for coord in CARTESIAN}
    for key, val in eta_mu.items():
        assert set(val) == {'aa', 'bb'}
        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)


if __name__ == "__main__":
    test_eta_works()
