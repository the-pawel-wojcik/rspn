from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import Descartes
from chem.meta.polarizability import Polarizability
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.ghf_ccsd._lheecc import build_pol_xA_F_xB


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
    t_response = lr.find_t_response(-cc_jacobian, cced_interaction_op)
    eta_mu = lr._find_eta_mu()
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)
    pol_etaA_xB_goal = Polarizability({
        Descartes.x: {
            Descartes.x: -0.0234,
            Descartes.y: 0.0,
            Descartes.z: 0.0,
        },
        Descartes.y: {
            Descartes.x: 0.0,
            Descartes.y: -2.7409,
            Descartes.z: 0.0,
        },
        Descartes.z: {
            Descartes.x: 0.0,
            Descartes.y: 0.0,
            Descartes.z: -1.4379,
        },
    })
    assert pol_etaA_xB.isclose(pol_etaA_xB_goal, 1e-4)
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(pol_etaA_xB)
    pol_xA_F_xB = build_pol_xA_F_xB(
        kwargs=builder_input,
        t_res_A=t_response,
        t_res_B=t_response,
    )
    pol_xA_F_xB_goal = Polarizability({
        Descartes.x: {
            Descartes.x: 0.0011,
            Descartes.y: 0.0,
            Descartes.z: 0.0,
        },
        Descartes.y: {
            Descartes.x: 0.0,
            Descartes.y: 0.3544,
            Descartes.z: 0.0,
        },
        Descartes.z: {
            Descartes.x: 0.0,
            Descartes.y: 0.0,
            Descartes.z: 0.3481,
        },
    })
    assert pol_xA_F_xB.isclose(pol_xA_F_xB_goal, atol=1e-4)
    print(f'{'X^B F X^B':{fmt}}')
    print(pol_xA_F_xB)

    pol = pol_etaA_xB + pol_xA_F_xB + pol_etaA_xB
    pol_goal = Polarizability({
        Descartes.x: {
            Descartes.x: -0.0458,
            Descartes.y: 0.0,
            Descartes.z: 0.0,
        },
        Descartes.y: {
            Descartes.x: 0.0,
            Descartes.y: -5.1273,
            Descartes.z: 0.0,
        },
        Descartes.z: {
            Descartes.x: 0.0,
            Descartes.y: 0.0,
            Descartes.z: -2.5278,
        },
    })
    assert pol.isclose(pol_goal, 1e-4)
    print(f'{' Polarizability ':{fmt}}')
    print(pol)
