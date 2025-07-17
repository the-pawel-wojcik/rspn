from pathlib import Path
import pickle

from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_Generators_Input
from numpy.typing import NDArray
from DrDePrince_Heom import build_eom_ccsd_H
import numpy as np
import pytest
from pytest import Config
from rspn.ghf_ccsd.ghf_ccsd_lr import build_cc_jacobian


@pytest.fixture(scope='session')
def eom_ccsd_H(
    ghf_ccsd_water_sto3g: GHF_CCSD,
    pytestconfig: Config,
) -> NDArray:
    pickles_dir = pytestconfig.rootpath / Path('tests/pickles')
    if not pickles_dir.is_dir():
        pickles_dir.mkdir()

    pickle_path = pickles_dir / Path("ghf_eom_ccsd_H_water_sto3g.pickle")
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            return pickle.load(f)

    ghf_data = ghf_ccsd_water_sto3g.ghf_data
    eom_ccsd_H: NDArray = build_eom_ccsd_H(
        f=ghf_data.f,
        g=ghf_data.g,
        o=ghf_data.o,
        v=ghf_data.v,
        t1=ghf_ccsd_water_sto3g.data.t1,
        t2=ghf_ccsd_water_sto3g.data.t2,
        nsocc=ghf_data.no,
        nsvirt=ghf_data.nv,
        core_list=list(range(ghf_data.no)),
    ) 
    with pickle_path.open('wb') as file:
        pickle.dump(eom_ccsd_H, file)

    return eom_ccsd_H


@pytest.fixture(scope='session')
def ghf_ccsd_Jacobian_water_sto3g(
    ghf_ccsd_water_sto3g: GHF_CCSD,
    pytestconfig: Config,
) -> NDArray:
    pickles_dir = pytestconfig.rootpath / Path('tests/pickles')
    if not pickles_dir.is_dir():
        pickles_dir.mkdir()

    pickle_path = pickles_dir / Path("ghf_ccsd_jacobian_water_sto3g.pickle")
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            return pickle.load(f)

    builders_input = GHF_Generators_Input(
        ghf_data=ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )
    cc_jacobian = build_cc_jacobian(builders_input)
    with pickle_path.open('wb') as file:
        pickle.dump(cc_jacobian, file)

    return cc_jacobian


def test_Heom_vs_Jacobian(
    eom_ccsd_H: NDArray,
    ghf_ccsd_Jacobian_water_sto3g: NDArray,
) -> None:
    jacobian = ghf_ccsd_Jacobian_water_sto3g
    # cut the reference
    eom_ccsd_H = eom_ccsd_H[1:, 1:]
    # cut diagonal
    eom_ccsd_H = eom_ccsd_H - np.diag(eom_ccsd_H.diagonal())
    ghf_ccsd_jacobian = jacobian - np.diag(jacobian.diagonal())

    assert np.allclose(ghf_ccsd_jacobian, eom_ccsd_H)


def test_spectrum_of_Heom(eom_ccsd_H: NDArray) -> None:
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
