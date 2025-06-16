import pickle

from chem.ccs.equations.util import UHF_CCS_InputTriple
from chem.ccs.uhf_ccs import UHF_CCS, UHF_CCS_InputPair
from rspn.uhf_ccs.uhf_ccs_lr import UHF_CCS_LR
from rspn.uhf_ccs._jacobian import build_cc_jacobian
from rspn.uhf_ccs._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccs._nuOpCC import build_nu_bar_V_cc


def test_polarizabilities():
    with open('../pickles/water_uhf_ccs_lambda_sto3g.pkl', 'rb') as bak_file:
        ccs: UHF_CCS = pickle.load(bak_file)
    input_triple = UHF_CCS_InputTriple(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
        uhf_ccs_lambda_data=ccs.cc_lambda_data,
    )
    input_pair = UHF_CCS_InputPair(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
    )
    lr = UHF_CCS_LR(**input_triple)
    cc_jacobian = build_cc_jacobian(
        kwargs=input_pair,
        dims=lr.assign_dims(),
    )
    cced_interaction_op = build_nu_bar_V_cc(input=input_pair)
    t_response = lr.find_t_response(cc_jacobian, cced_interaction_op)
    eta_mu = lr._find_eta_mu()
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(pol_etaA_xB)
    print(f'{'2 * η^A X^B':{fmt}}')
    print(pol_etaA_xB + pol_etaA_xB)
    pol_xA_F_xB = build_pol_xA_F_xB(
        kwargs=input_triple,
        t_res_A=t_response,
        t_res_B=t_response,
    )
    print(f'{'X^B F X^B':{fmt}}')
    print(pol_xA_F_xB)
    print(f'{' Polarizability ':{fmt}}')
    print(pol_etaA_xB + pol_xA_F_xB + pol_etaA_xB)


if __name__ == "__main__":
    test_polarizabilities()
