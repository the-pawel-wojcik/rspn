""" Test the construction of the eta for the electric dipole moment. """
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import Descartes
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config


def test_eta_works(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
    eta_mu = lr._find_eta_mu()
    assert set(eta_mu) == {coord for coord in Descartes}
    for _, val in eta_mu.items():
        assert set(val) == {'singles', 'doubles'}
        assert val['singles'].shape == (4, 10)
        assert val['doubles'].shape == (4, 4, 10, 10)
