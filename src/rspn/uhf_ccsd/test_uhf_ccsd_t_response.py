import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import Descartes
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
import numpy as np


def t_response_test():
    with open('uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    cc_mu = lr.build_cc_electric_dipole_singles()
    t_response = lr.find_t_response(
        cc_jacobian=cc_jacobian,
        cc_mu=cc_mu,
    )
    print(f'{t_response.keys()=}')
    assert set(t_response.keys()) == {Descartes.x, Descartes.y, Descartes.z}
    mux = t_response[Descartes.x]
    print(f'{mux.keys()=}')
    assert set(mux.keys()) == {'aa', 'bb'}
    aa = mux['aa']
    with np.printoptions(precision=3):
        print(aa)

if __name__ == "__main__":
    t_response_test()
