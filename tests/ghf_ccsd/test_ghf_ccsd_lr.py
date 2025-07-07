import pickle

from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config
from chem.ccsd.ghf_ccsd import GHF_CCSD


def test_GHF_CCSD_LR():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
