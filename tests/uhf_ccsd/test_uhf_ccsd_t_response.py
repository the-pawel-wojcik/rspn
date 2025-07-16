from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import Descartes
import pytest
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_t_response(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    ccsd = uhf_ccsd_water_sto3g

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    builder_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(
        kwargs=builder_input,
        dims=lr.assign_dims(),
    )
    cced_interaction_op = build_nu_bar_V_cc(input=builder_input)
    t_mu_resp = lr.find_t_response(
        minus_cc_jacobian=-cc_jacobian,
        cc_mu=cced_interaction_op,
    )
    assert set(t_mu_resp.keys()) == {Descartes.x, Descartes.y, Descartes.z}
    t_mu_res_x = t_mu_resp[Descartes.x]
    assert set(t_mu_res_x.keys()) == {
        'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
    }
    aa = t_mu_res_x['aa']
    assert aa.shape == (2, 5)
    aaaa = t_mu_res_x['aaaa']
    assert aaaa.shape == (2, 2, 5, 5)


def test_t_repsone_with_no_jacobian_build(
    uhf_ccsd_water_sto3g: UHF_CCSD,
) -> None:
    ccsd = uhf_ccsd_water_sto3g

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    builder_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    minus_cc_Jacobian_action = Minus_UHF_CCSD_Jacobian_action(
        uhf_hf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    cced_interaction_op = build_nu_bar_V_cc(input=builder_input)
    t_mu_resp = lr.find_t_response(
        minus_cc_jacobian=minus_cc_Jacobian_action,
        cc_mu=cced_interaction_op,
    )
    assert set(t_mu_resp.keys()) == {Descartes.x, Descartes.y, Descartes.z}
    t_mu_res_x = t_mu_resp[Descartes.x]
    assert set(t_mu_res_x.keys()) == {
        'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
    }
    aa = t_mu_res_x['aa']
    assert aa.shape == (2, 5)
    aaaa = t_mu_res_x['aaaa']
    assert aaaa.shape == (2, 2, 5, 5)
