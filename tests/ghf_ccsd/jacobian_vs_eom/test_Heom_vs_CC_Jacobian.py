import pickle 

from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_Generators_Input
from DrDePrince_Heom import build_eom_ccsd_H
import numpy as np
from rspn.ghf_ccsd.ghf_ccsd_lr import build_cc_jacobian


def prepare_Heom() -> None:
    with open('../pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    ghf_data = ccsd.ghf_data
    eom_ccsd_H = build_eom_ccsd_H(
        f=ghf_data.f,
        g=ghf_data.g,
        o=ghf_data.o,
        v=ghf_data.v,
        t1=ccsd.data.t1,
        t2=ccsd.data.t2,
        nsocc=ghf_data.no,
        nsvirt=ghf_data.nv,
        core_list=list(range(ghf_data.no)),
    ) 
    with open('pickles/water_sto3g_eom_ccsd_H.pkl', 'wb') as file:
        pickle.dump(eom_ccsd_H, file)


def prepare_jacobian() -> None:
    with open('../pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    builders_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(builders_input)
    with open('pickles/water_sto3g_cc_jacobian.pkl', 'wb') as file:
        pickle.dump(cc_jacobian, file)


def test_Heom_vs_Jacobian():
    # prepare_Heom()
    # prepare_jacobian()
    with open('pickles/water_sto3g_eom_ccsd_H.pkl', 'rb') as file:
        eom_ccsd_H = pickle.load(file)
    with open('pickles/water_sto3g_cc_jacobian.pkl', 'rb') as file:
        ghf_ccsd_jacobian = pickle.load(file)

    # cut the reference
    eom_ccsd_H = eom_ccsd_H[1:, 1:]
    # cut diagonal
    eom_ccsd_H = eom_ccsd_H - np.diag(eom_ccsd_H.diagonal())
    ghf_ccsd_jacobian = ghf_ccsd_jacobian - np.diag(ghf_ccsd_jacobian.diagonal())

    assert np.allclose(ghf_ccsd_jacobian, eom_ccsd_H)
