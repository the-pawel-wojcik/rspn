from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.meta.coordinates import Descartes
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_t_response_structure(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data)
    builder_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(kwargs=builder_input)
    cced_interaction_op = build_nu_bar_V_cc(input=builder_input)
    t_mu_resp = lr.find_t_response(
        minus_cc_jacobian=-cc_jacobian,
        cc_mu=cced_interaction_op,
    )
    assert set(t_mu_resp.keys()) == {Descartes.x, Descartes.y, Descartes.z}
    t_mu_res_x = t_mu_resp[Descartes.x]
    # HINT: double counting
    # assert t_mu_res_x.shape == (310,)
    assert t_mu_res_x.shape == (1640,)
