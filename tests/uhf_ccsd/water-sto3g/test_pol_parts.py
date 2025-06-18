import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.uhf_ccsd._jacobian import (
    cc_jacobian_singles_singles,
    cc_jacobian_singles_doubles,
    cc_jacobian_doubles_singles,
    cc_jacobian_doubles_doubles,
    build_cc_jacobian,
)


def test_pol_parts():
    with open('pickles/water_sto3g.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(uhf_scf_data=ccsd.scf_data, uhf_ccsd_data=ccsd.data,)
    builders_input = GeneratorsInput(
        uhf_scf_data=lr.uhf_scf_data,
        uhf_ccsd_data=lr.uhf_ccsd_data,
    )
    cc_jacobian = build_cc_jacobian(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    cc_electric_dipole = build_nu_bar_V_cc(input=builders_input)
    t_response = lr.find_t_response(cc_jacobian, cc_electric_dipole)
    eta_mu = lr._find_eta_mu()
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)

    pol_xA_F_xB = build_pol_xA_F_xB(
        builders_input, t_res_A=t_response, t_res_B=t_response,
    )
    pol_etaB_xA = pol_etaA_xB
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(pol_etaA_xB)
    print(f'{'η^A X^B + η^B X^A':{fmt}}')
    print(pol_etaA_xB + pol_etaB_xA)
    print(f'{'X^B F X^B':{fmt}}')
    print(pol_xA_F_xB)
    print(f'{' η^A X^B + X^B F X^B + η^B X^A ':{fmt}}')
    print(pol_etaA_xB + pol_xA_F_xB + pol_etaB_xA)


if __name__ == "__main__":
    test_pol_parts()
