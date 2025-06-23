import pickle

from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


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
    intermediates = extract_intermediates(hf_result.wfn)
    ccsd = UHF_CCSD(intermediates)
    ccsd.verbose = 1
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    with open('pickles/water_sto3g@HF.pkl', 'wb') as bak_file:
        pickle.dump(ccsd, bak_file)


if __name__ == "__main__":
    solve_and_save_water_sto3g_at_HF()
