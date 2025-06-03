"""
Test the construction of the eta for the electric dipole moment.
"""
import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import CARTESIAN, Descartes
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def pprint_pol(pol: dict[Descartes, dict[Descartes, float]]) -> None:
    for first in CARTESIAN:
        for second in CARTESIAN:
            print(f'{first}{second}: {pol[first][second]:7.4f}')


def test_polarizabilities():
    with open('pickles/uhf_ccsd_lambda.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    pol = lr.find_polarizabilities()
    pprint_pol(pol)


if __name__ == "__main__":
    test_polarizabilities()
