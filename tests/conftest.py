from pathlib import Path
import pickle

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data
import pytest
from pytest import Config


def solve_ghf_ccsd_for_water_sto3g_at_HF() -> GHF_CCSD:
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O  0.0  0.0000000  0.1271610
    H  0.0  0.7580820 -0.5086420
    H  0.0 -0.7580820 -0.5086420
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    ccsd = GHF_CCSD(ghf_data)
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    return ccsd


def solve_ghf_ccsd_for_water_ccpVDZ_at_HF() -> GHF_CCSD:
    """ Geometry from CCCBDB: HF/cc-pVDZ """
    geometry = """
    0 1
    O  0.0  0.0000000  0.1157190
    H  0.0  0.7487850 -0.4628770
    H  0.0 -0.7487850 -0.4628770
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='cc-pVDZ')
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    ccsd = GHF_CCSD(ghf_data)
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    return ccsd


@pytest.fixture(scope='session')
def ghf_ccsd_water_sto3g(pytestconfig: Config) -> GHF_CCSD:
    pickles_dir = pytestconfig.rootpath / Path('pickles')
    if not pickles_dir.is_dir():
        pickles_dir.mkdir()

    pickle_path = pickles_dir / Path("ghf_ccsd_water_sto3g@HF.pickle")
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            return pickle.load(f)
    
    ccsd = solve_ghf_ccsd_for_water_sto3g_at_HF()
    with pickle_path.open("wb") as f:
        pickle.dump(ccsd, f)
    return ccsd


@pytest.fixture(scope='session')
def ghf_ccsd_water_ccpVDZ(pytestconfig: Config) -> GHF_CCSD:
    pickles_dir = pytestconfig.rootpath / Path('pickles')
    if not pickles_dir.is_dir():
        pickles_dir.mkdir()

    pickle_path = pickles_dir / Path("ghf_ccsd_water_ccpVDZ@HF.pickle")
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            return pickle.load(f)
    
    ccsd = solve_ghf_ccsd_for_water_ccpVDZ_at_HF()
    with pickle_path.open("wb") as f:
        pickle.dump(ccsd, f)
    return ccsd
