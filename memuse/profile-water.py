import pickle
import sys
import tracemalloc

from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._lheecc import build_pol_xA_F_xB
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config

STREAM = sys.stderr

def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def tmp(info: str) -> None:
    current, peak = tracemalloc.get_traced_memory()
    print('\nMalloc trace', file=STREAM)
    print(f' info    = {info}', file=STREAM)
    print(f' current = {humanify(current)}', file=STREAM)
    print(f' peak    = {humanify(peak)}\n', file=STREAM, flush=True)


def prepare_pickle() -> None:
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
    tmp("after HF")
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
    tmp("after CCSD")
    lr = GHF_CCSD_LR(
        ghf_data=ghf_data,
        ghf_ccsd_data=ccsd.data,
        CONFIG=GHF_CCSD_LR_config(
            gmres_threshold=1e-6,
            gmres_maxiter=100,
            gmres_preconditioner=None,
            gmres_guess=None,
            gmres_verbose=False,
            store_jacobian=True,
            store_lHeecc=True,
            verbose=0,
        )
    )
    builders_input = GHF_Generators_Input(
        ghf_data=lr.ghf_data,
        ghf_ccsd_data=lr.ghf_ccsd_data,
    )
    with open('input.pickle', 'wb') as save_file:
        pickle.dump((lr, builders_input), save_file)


def profile_polarizabilities() -> None:
    tracemalloc.start()
    tmp("start")
    with open('input.pickle', 'rb') as save_file:
        saved_files = pickle.load(save_file)
    lr: GHF_CCSD_LR = saved_files[0]
    builders_input: GHF_Generators_Input = saved_files[1]

    cc_electric_dipole = build_nu_bar_V_cc(input=builders_input)
    tmp("after mu bar construction")
    print("Mu bar components:", file=STREAM)
    for direction, op in cc_electric_dipole.items():
        for key, matrix in op.items():
            print(
                f'{direction} {key} {humanify(matrix.nbytes)} {matrix.shape}',
                file=STREAM,
            )

    cc_jacobian = build_cc_jacobian(
        kwargs=builders_input,
    )
    tmp("after Jacobian build")
    print(
        f'GHF-CCSD Jacobian {humanify(cc_jacobian.nbytes)} {cc_jacobian.shape}',
        file=STREAM,
    )
    t_response = lr.find_t_response(
        minus_cc_jacobian=-cc_jacobian,
        cc_mu=cc_electric_dipole,
    )

    tmp("after response vectors build.")
    print("t response components:", file=STREAM)
    for direction, matrix in t_response.items():
        print(
            f'{direction} {humanify(matrix.nbytes)} {matrix.shape}',
            file=STREAM,
        )

    eta_mu = lr._find_eta_mu()
    tmp("after eta build.")
    print("eta components:", file=STREAM)
    for direction, op in eta_mu.items():
        for key, matrix in op.items():
            print(
                f'{direction} {key} {humanify(matrix.nbytes)} {matrix.shape}',
                file=STREAM,
            )

    # TODO: generalize to operators other than electric dipole
    pol_etaA_xB = lr._build_pol_eta_X(eta_mu, t_response)
    # when there is only one operator this term is the same as the first
    # one pol_etaB_xA = self._build_pol_eta_X(eta_mu, t_response)
    pol_etaB_xA = pol_etaA_xB

    tmp("after build of the eta-part of the polarizability.")

    pol_xA_F_xB = build_pol_xA_F_xB(
        builders_input, t_res_A=t_response, t_res_B=t_response,
    )

    tmp("after build of the F-part of polarizability.")

    pol = pol_etaA_xB + pol_xA_F_xB + pol_etaB_xA
    fmt = '=^50'
    print(f'{' Polarizability (atomic units)':{fmt}}')
    print(pol, flush=True)
    tmp("after polarizabilities")


def main():
    prepare_pickle()
    profile_polarizabilities()


if __name__ == "__main__":
    main()
