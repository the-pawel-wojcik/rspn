from pathlib import Path
import pytest
import pickle

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data


def solve_and_save_water_sto3g_at_HF():
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    ccsd = GHF_CCSD(ghf_data)
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    with open('pickles/water_sto3g@HF.pkl', 'wb') as bak_file:
        pickle.dump(ccsd, bak_file)


pickles = Path('pickles/')
if not pickles.is_dir():
    print('Creating the pickles/ directory')
    pickles.mkdir()


water_sto3g_pkl_path = Path('pickles/water_sto3g@HF.pkl')
if not water_sto3g_pkl_path.is_file():
    solve_and_save_water_sto3g_at_HF()


@pytest.fixture(scope="module")
def water_sto3g() -> GHF_CCSD:
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)
    return ccsd
