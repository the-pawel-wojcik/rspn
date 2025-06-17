import pickle

import psi4
from psi4.core import Molecule, Wavefunction
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


def scf() -> tuple[Molecule, float, Wavefunction]:
    """ Geometry from CCCBDB: HF/STO-3G """
    mol: Molecule = psi4.geometry("""
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """)

    psi4.set_options({'basis': 'sto-3g',
                      'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12})

    psi4.core.be_quiet()

    # compute the Hartree-Fock energy and wavefunction
    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)

    return mol, energy, wfn


def solve_and_save():
    _, _, wfn = scf()
    intermediates = extract_intermediates(wfn)
    ccsd = UHF_CCSD(intermediates)
    ccsd.verbose = 1
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    with open('pickles/water_sto3g.pkl','wb') as bak_file:
        pickle.dump(ccsd, bak_file)


if __name__ == "__main__":
    solve_and_save()
