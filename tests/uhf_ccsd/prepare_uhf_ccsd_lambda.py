import pickle

from chem.hf.electronic_structure import scf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


def basic_test():
    _, _, wfn = scf()
    intermediates = extract_intermediates(wfn)
    ccsd = UHF_CCSD(intermediates)
    ccsd.verbose = 1
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    with open('pickles/uhf_ccsd_lambda.pkl','wb') as bak_file:
        pickle.dump(ccsd, bak_file)


if __name__ == "__main__":
    basic_test()
