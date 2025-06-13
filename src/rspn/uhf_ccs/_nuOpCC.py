r"""
Builds the matrices of the perturbation operator as given by Eq. (60) from Ref.
[1]. E.g., the singles part has the following look

    matrix[x][a, i] = <HF| e^{-T} a*(i) a(a) \hat{mu} _x e^{T} |HF>

the flipped order of `a` and `i` appears because the pair (a, i) is an index of
a single excitation operator $\tau _mu ^\dagger$. The theory from Ref. [1] 
works on the excitation indices.

[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
from chem.ccs.equations.util import UHF_CCS_InputPair
from chem.meta.coordinates import Descartes, CARTESIAN
from numpy.typing import NDArray
from rspn.uhf_ccs.equations.nu_Op_cc.singles import (
    get_mux_aa,
    get_mux_bb,
    get_muy_aa,
    get_muy_bb,
    get_muz_aa,
    get_muz_bb,
)


def build_nu_bar_V_cc(
    input: UHF_CCS_InputPair,
) -> dict[Descartes, dict[str, NDArray]]:
    """It is the same as the input operator matrix elements as for T=0 all e^T
    componets vanish."""
    xi_mu = {}
    for coord in CARTESIAN:
        xi_mu[coord] = _build_helper(coord, input)
    return xi_mu


def _build_helper(
    coord: Descartes,
    input: UHF_CCS_InputPair,
) -> dict[str, NDArray]:

    if coord == Descartes.x:
        return {
            'aa': get_mux_aa(**input),
            'bb': get_mux_bb(**input),
        }

    elif coord == Descartes.y:
        return {
            'aa': get_muy_aa(**input),
            'bb': get_muy_bb(**input),
        }

    elif coord == Descartes.z:
        return {
            'aa': get_muz_aa(**input),
            'bb': get_muz_bb(**input),
        }

    else:
        raise ValueError(f"Unknown cartesian coordinate: {coord}.")
