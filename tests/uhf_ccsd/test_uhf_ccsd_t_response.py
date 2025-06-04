import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from chem.meta.coordinates import Descartes
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def test_t_response():
    with open('pickles/uhf_ccsd.pkl','rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    cc_mu = lr.build_cc_electric_dipole_singles()
    t_mu_resp = lr.find_t_response(
        cc_jacobian=cc_jacobian,
        cc_mu=cc_mu,
    )
    assert set(t_mu_resp.keys()) == {Descartes.x, Descartes.y, Descartes.z}
    t_mu_res_x = t_mu_resp[Descartes.x]
    assert set(t_mu_res_x.keys()) == {
        'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
    }
    aa = t_mu_res_x['aa']
    assert aa.shape == (19, 5)
    aaaa = t_mu_res_x['aaaa']
    assert aaaa.shape == (19, 19, 5, 5)
    # TODO: add tests of the values
    # with np.printoptions(precision=3):
    #     print(aa)

if __name__ == "__main__":
    test_t_response()
