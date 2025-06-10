import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import Descartes
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc


def test_t_response():
    with open('pickles/water_sto3g.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
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
        cc_jacobian=cc_jacobian,
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


if __name__ == "__main__":
    test_t_response()
