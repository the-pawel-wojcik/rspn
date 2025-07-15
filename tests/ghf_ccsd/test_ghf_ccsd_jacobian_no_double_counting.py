from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd.ghf_ccsd_lr import build_cc_jacobian


def split_roots_into_singles_and_degenerate(
    eigenvalues: NDArray,
) -> tuple[list[float], list[dict[str, int | float]]]:
    singlets: list[float] = list()
    degenerates: list[dict[str, int | float]] = list()

    # split the eigenvalues into degenerte ones and non-degenerate ones
    evals_iter = iter(sorted(eigenvalues, key=lambda z: abs(z)))
    current_triple = [next(evals_iter) for _ in range(3)]
    degenate_bunch = list()

    # handle the starting edge case explicitly
    if np.isclose(current_triple[0], current_triple[1], 1e-12):
        degenate_bunch.append(current_triple[0])
    else:
        singlets.append(current_triple[0])

    for eval in evals_iter:
        if np.isclose(current_triple[0], current_triple[1], 1e-12):
            degenate_bunch.append(current_triple[1])
        else:
            if len(degenate_bunch) != 0:
                degenerates.append({
                    'value': degenate_bunch[0],
                    'degeneracy': len(degenate_bunch),
                })
            if np.isclose(current_triple[1], current_triple[2], 1e-12):
                degenate_bunch = [current_triple[1]] # start a new series
            else:
                singlets.append(current_triple[1])
        current_triple = current_triple[1:] + [eval]
    else:
        if np.isclose(eval, current_triple[2], 1e-12):
            degenerates.append({
                'value': eval,
                'degeneracy': len(degenate_bunch) + 1,
            })
        else:
            if len(degenate_bunch) == 0:
                singlets.append(eval)
            else:
                degenerates.append({
                    'value': degenate_bunch[0],
                    'degeneracy': len(degenate_bunch),
                })
    return singlets, degenerates


def test_cc_jacobian_spectrum(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    builders_input = GHF_Generators_Input(
        ghf_data=ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )

    cc_jacobian = build_cc_jacobian(builders_input)
    eigenvalues = np.linalg.eigvals(cc_jacobian)
    singlets, degenerates = split_roots_into_singles_and_degenerate(eigenvalues)

    print("Singlets energies:")
    for eval in singlets:
        print(f'{eval.real:+12.6f}', end='')
        if abs(eval.imag) > 1e-6:
            print(f'{eval.real:+12.6f}i')
        else:
            print('')

    print("Degenerate energies:")
    for deg in degenerates:
        eval = deg['value']
        degeneracy = deg['degeneracy']
        print(f'{eval.real:+12.6f}', end='')
        if abs(eval.imag) > 1e-6:
            print(f'{eval.real:+12.6f}i')
        else:
            print(f' x {degeneracy}')
