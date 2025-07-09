"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import Descartes
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config


def test_eta_works():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
    eta_mu = lr._find_eta_mu()
    assert set(eta_mu) == {coord for coord in Descartes}
    for _, val in eta_mu.items():
        assert set(val) == {'ref', 'singles', 'doubles'}
        assert val['ref'].ndim == 0
        assert val['singles'].shape == (4, 10)
        assert val['doubles'].shape == (4, 4, 10, 10)
