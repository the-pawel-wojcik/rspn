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


def test_spectrum_of_Heom():
    # prepare_Heom()
    with open('pickles/water_sto3g_eom_ccsd_H.pkl', 'rb') as file:
        eom_ccsd_H = pickle.load(file)

    eom_ccsd_H_evals = np.linalg.eigvals(eom_ccsd_H)
    print()
    sorted_spectrum = sorted(eom_ccsd_H_evals, key=lambda z: z)
    print(f"Lowest eigenvalue: {sorted_spectrum[0]}")
    excitation_spectrum = sorted_spectrum - sorted_spectrum[0]
    print(f"Excitation spectrum")
    for eval in excitation_spectrum:
        print(f'{eval.real:+12.6f}', end='')
        if abs(eval.imag) > 1e-6:
            print(f'{eval.real:+12.6f}i')
        else:
            print('')
