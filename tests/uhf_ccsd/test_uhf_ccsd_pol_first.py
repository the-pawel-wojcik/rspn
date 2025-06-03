"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_polarizabilities():
    with open('pickles/uhf_ccsd_lambda.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    pol = lr.find_polarizabilities()
    print(pol)


if __name__ == "__main__":
    test_polarizabilities()
