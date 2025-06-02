import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
import numpy as np
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_cc_mu():
    with open('uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    eta_mu = lr.build_cc_electric_dipole_singles()
    np.set_printoptions(precision=2)
    for key, val in eta_mu.items():
        print(f'{key:=^50}')
        print(f'{val.keys()=}')
        # max = np.max(np.abs(val['aa']))
        # mask = np.abs(val['aa']) < max/50
        # tmp = val['aa'].copy()
        # tmp[mask] = 0.0
        # print(f'{tmp=}')
        print(f'{val['aa'].shape=}')
        print(f'{val['bb'].shape=}')
        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)


if __name__ == "__main__":
    test_cc_mu()
