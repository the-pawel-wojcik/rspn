import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_polarizabilities():
    with open('pickles/uhf_ccsd_lambda.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    cc_mu = lr.build_cc_electric_dipole_singles()
    t_response = lr.find_t_response(cc_jacobian, cc_mu)
    eta_mu = lr._find_eta_mu()
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)
    print(f'{pol_etaA_xB=!s}')
    pol_xA_F_xB = lr._build_pol_xA_F_xB(t_res_A=t_response, t_res_B=t_response)
    print(f'{pol_xA_F_xB=!s}')


if __name__ == "__main__":
    test_polarizabilities()
