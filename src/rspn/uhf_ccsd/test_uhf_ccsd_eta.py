"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_eta_works():
    with open('data/uhf_ccsd_lambda.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    eta_mu = lr._find_eta_mu()
    for key, val in eta_mu.items():
        print(f'{key:=^50}')
        print(f'{val.keys()=}')
        print(f'{type(val['aa'])=}')
        print(f'{val['aa'].shape=}')
        print(f'{val['bb'].shape=}')


def test_eta_missing_lambda():
    """ Solving lambdas is necessary for builidng the response vectors. Produces
    an error if the lambdas are missing. TODO: solve the lambda equations first
    instead. """
    with open('uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    eta_mu = lr._find_eta_mu()
    for key, val in eta_mu.items():
        print(f'{key:=^50}')
        print(f'{val.keys()=}')
        print(f'{type(val['aa'])=}')
        print(f'{val['aa'].shape=}')
        print(f'{val['bb'].shape=}')


if __name__ == "__main__":
    test_eta_missing_lambda()
    test_eta_missing_lambda()
