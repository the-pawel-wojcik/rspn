from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import Descartes
import numpy as np
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.ghf_ccsd._lheecc import build_pol_xA_F_xB
from psi4_data import PSI4_MU_BAR_IjAb, PSI4_MU_BAR_IA, psi4_rhf_doubles_to_ghf


def test_polarizabilities(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
    builder_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(kwargs=builder_input)
    cced_interaction_op = build_nu_bar_V_cc(input=builder_input)
    # cced_interaction_op = {
    #     direction: {
    #         'singles': value['singles'],
    #         'doubles': value['doubles'] * 0.0,
    #     } for direction, value in cced_interaction_op.items()
    # }
    # cced_interaction_op_psi4 = {
    #     direction: {
    #         'singles': PSI4_MU_BAR_IA[direction].T,
    #         # 'singles': cced_interaction_op[direction]['singles'],
    #         'doubles': psi4_rhf_doubles_to_ghf(double_rhf, ccsd.ghf_data)
    #     }
    #     for direction, double_rhf in PSI4_MU_BAR_IjAb.items()
    # }
    # cced_interaction_op = cced_interaction_op_psi4
    # for direction in Descartes:
    #     assert np.allclose(
    #         cced_interaction_op_psi4[direction]['singles'],
    #         cced_interaction_op[direction]['singles'],
    #         atol=1e-8,
    #     )
    t_response = lr.find_t_response(-cc_jacobian, cced_interaction_op)
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
