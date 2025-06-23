import pickle

from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


def prepare_water_ccpVDZ():
    """ Solve and save to a pickle the CCSD and CCSD Lambda equations for
    water. 

    The geometry of water comes from CCCBDB. The geometry was calculated using
    CCSD/cc-pVDZ.
    """
    water_geometry = """
    0 1
    O  0.0   0.0000000   0.1210960
    H  0.0   0.7505720  -0.4843850
    H  0.0  -0.7505720  -0.4843850
    symmetry c1
    """
    hf_data = hf(geometry=water_geometry, basis='cc-pvdz',)
    intermediates = extract_intermediates(hf_data.wfn)
    ccsd = UHF_CCSD(intermediates)
    ccsd.verbose = 1
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    # changed from
    # with open('pickles/uhf_ccsd_lambda.pkl', 'wb') as bak_file:
    with open('pickles/water_ccpVDZ.pkl', 'wb') as bak_file:
        pickle.dump(ccsd, bak_file)


if __name__ == "__main__":
    prepare_water_ccpVDZ()
