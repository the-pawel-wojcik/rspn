import pickle

from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config
from chem.ccsd.ghf_ccsd import GHF_CCSD


def test_constructor():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)


def test_polarizabilities():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
    polarizability = lr.find_polarizabilities()
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(polarizability)
