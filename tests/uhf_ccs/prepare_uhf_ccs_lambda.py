import pickle

from chem.hf.electronic_structure import scf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccs.uhf_ccs import UHF_CCS


def prepare_pickle():
    _, _, wfn = scf()
    intermediates = extract_intermediates(wfn)
    ccs = UHF_CCS(intermediates)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    ccs.solve_lambda_equations()
    with open('pickles/water_uhf_ccs_lambda_ccpVDZ.pkl','wb') as bak_file:
        pickle.dump(ccs, bak_file)


if __name__ == "__main__":
    prepare_pickle()
