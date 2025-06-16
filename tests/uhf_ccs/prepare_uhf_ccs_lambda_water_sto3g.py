import pickle

import psi4
from psi4.core import Molecule, Wavefunction
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccs.uhf_ccs import UHF_CCS


def scf(
    geometry: str,
    basis: str,
) -> tuple[Molecule, float, Wavefunction]:

    mol: Molecule = psi4.geometry(geometry)

    psi4.set_options({
        'basis': basis,
        'scf_type': 'pk',
        'e_convergence': 1e-12,
        'd_convergence': 1e-12,
    })

    psi4.core.be_quiet()

    # compute the Hartree-Fock energy and wavefunction
    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)

    return mol, energy, wfn


def prepare_pickle():
    water_uhf_sto3g_geometry = """
    0 1
    O1  0.0000000   0.0000000   0.1271610
    H2  0.0000000   0.7580820  -0.5086420
    H3  0.0000000  -0.7580820  -0.5086420
    symmetry c1
    """
    _, _, wfn = scf(water_uhf_sto3g_geometry, 'sto-3g')
    intermediates = extract_intermediates(wfn)
    ccs = UHF_CCS(intermediates)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    ccs.solve_lambda_equations()
    with open('pickles/water_uhf_ccs_lambda_sto3g.pkl', 'wb') as bak_file:
        pickle.dump(ccs, bak_file)


if __name__ == "__main__":
    prepare_pickle()
