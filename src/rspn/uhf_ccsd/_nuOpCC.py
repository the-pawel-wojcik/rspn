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
from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import Descartes, CARTESIAN
from numpy.typing import NDArray
from rspn.uhf_ccsd.equations.dipole.singles import (
    get_mux_aa,
    get_mux_bb,
    get_muy_aa,
    get_muy_bb,
    get_muz_aa,
    get_muz_bb,
)
from rspn.uhf_ccsd.equations.dipole.doubles import (
    get_mux_aaaa,
    get_mux_abab,
    get_mux_abba,
    get_mux_baab,
    get_mux_baba,
    get_mux_bbbb,
    get_muy_aaaa,
    get_muy_abab,
    get_muy_abba,
    get_muy_baab,
    get_muy_baba,
    get_muy_bbbb,
    get_muz_aaaa,
    get_muz_abab,
    get_muz_abba,
    get_muz_baab,
    get_muz_baba,
    get_muz_bbbb,
)


def build_nu_bar_V_cc(
    input: GeneratorsInput,
) -> dict[Descartes, dict[str, NDArray]]:
    xi_mu = {}
    for coord in CARTESIAN:
        xi_mu[coord] = _build_helper(coord, input)
    return xi_mu


def _build_helper(
    coord: Descartes,
    input: GeneratorsInput,
) -> dict[str, NDArray]:

    if coord == Descartes.x:
        return {
            'aa': get_mux_aa(**input),
            'bb': get_mux_bb(**input),
            'aaaa': get_mux_aaaa(**input),
            'abab': get_mux_abab(**input),
            'abba': get_mux_abba(**input),
            'baab': get_mux_baab(**input),
            'baba': get_mux_baba(**input),
            'bbbb': get_mux_bbbb(**input),
        }

    elif coord == Descartes.y:
        return {
            'aa': get_muy_aa(**input),
            'bb': get_muy_bb(**input),
            'aaaa': get_muy_aaaa(**input),
            'abab': get_muy_abab(**input),
            'abba': get_muy_abba(**input),
            'baab': get_muy_baab(**input),
            'baba': get_muy_baba(**input),
            'bbbb': get_muy_bbbb(**input),
        }

    elif coord == Descartes.z:
        return {
            'aa': get_muz_aa(**input),
            'bb': get_muz_bb(**input),
            'aaaa': get_muz_aaaa(**input),
            'abab': get_muz_abab(**input),
            'abba': get_muz_abba(**input),
            'baab': get_muz_baab(**input),
            'baba': get_muz_baba(**input),
            'bbbb': get_muz_bbbb(**input),
        }

    else:
        raise ValueError(f"Unknown cartesian coordinate: {coord}.")
