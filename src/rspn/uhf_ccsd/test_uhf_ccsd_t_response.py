import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
import numpy as np
from scipy.sparse.linalg import gmres


def t_response_test():
    with open('uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    cc_mu = {}
    for coord in CARTESIAN:
        cc_mu[coord] = lr._build_the_cc_dipole_helper(coord)

    for coord in CARTESIAN:
        sol, exit_code = gmres(cc_jacobian, cc_mu[coord], rtol=1e-7, atol=1e-7)
        if exit_code != 0:
            raise RuntimeError('GMRES didn\'t find the response vector!')
        print(f'Response vector for {coord} found!')
        print(f'{np.allclose(cc_jacobian @ sol, cc_mu[coord], rtol=1e-6, atol=1e-6)=}')


if __name__ == "__main__":
    t_response_test()
