from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config


def main():
    basis = "sto-3g"
    # CCCBDB geometry for HF water/STO-3G 
    geometry = """
    0 1
    O1  0.0000000   0.0000000   0.1271610
    H2  0.0000000   0.7580820  -0.5086420
    H3  0.0000000  -0.7580820  -0.5086420
    symmetry c1
    """
    hf_out: ResultHF = hf(geometry=geometry, basis=basis)
    print(f'{hf_out.hf_energy=} Eh')
    ghf_data = wfn_to_GHF_Data(hf_out.wfn)
    ccsd = GHF_CCSD(
        ghf_data=ghf_data,
        config=GHF_CCSD_Config(
            verbose=2,
            use_diis=False,
            max_iterations=100,
            energy_convergence=1e-10,
            residuals_convergence=1e-10,
            shift_1e=0.0,
            shift_2e=0.0,
            t_amp_print_threshold=1e-1,
        )
    )
    ccsd.solve_cc_equations()
    ccsd.solve_lambda_equations()
    ccsd_energy = (
        ccsd.get_energy()
        +
        hf_out.molecule.nuclear_repulsion_energy()
    )
    print(f'{ccsd_energy=} Eh')
    lr = GHF_CCSD_LR(
        ghf_data=ghf_data,
        ghf_ccsd_data=ccsd.data,
        CONFIG=GHF_CCSD_LR_config(
            gmres_threshold=1e-6,
            gmres_maxiter=100,
            gmres_preconditioner=None,
            gmres_guess=None,
            gmres_verbose=True,
            store_jacobian=True,
            store_lHeecc=True,
            verbose=2,
        )
    )
    pol = lr.find_polarizabilities()
    fmt = '=^50'
    print(f'{' Polarizability (atomic units)':{fmt}}')
    print(pol)


if __name__ == "__main__":
    main()
