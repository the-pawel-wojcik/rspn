import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
from rspn.uhf_ccsd._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_polarizabilities():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr_config = UHF_CCSD_LR_config(store_jacobian=False)
    builder_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    lr = UHF_CCSD_LR(CONFIG=lr_config, **builder_input)
    minus_cc_Jacobian_action = Minus_UHF_CCSD_Jacobian_action(
        uhf_hf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )

    cced_interaction_op = build_nu_bar_V_cc(input=builder_input)
    t_response = lr.find_t_response(
        minus_cc_jacobian=minus_cc_Jacobian_action,
        cc_mu=cced_interaction_op,
    )
    eta_mu = lr._find_eta_mu()
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(pol_etaA_xB)
    pol_xA_F_xB = build_pol_xA_F_xB(
        kwargs=builder_input,
        t_res_A=t_response,
        t_res_B=t_response,
    )
    print(f'{'X^B F X^B':{fmt}}')
    print(pol_xA_F_xB)
    print(f'{' Polarizability ':{fmt}}')
    print(pol_etaA_xB + pol_xA_F_xB + pol_etaA_xB)


if __name__ == "__main__":
    test_polarizabilities()
