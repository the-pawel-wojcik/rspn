r"""
Builds the matrices of the perturbation operator as given by Eq. (60) from Ref.
[1]. E.g., the singles part has the following look

    matrix[x][a, i] = <HF| e^{-T} a*(i) a(a) \hat{mu} _x e^{T} |HF>

the flipped order of `a` and `i` appears because the pair (a, i) is an index of
a single excitation operator $\tau _mu ^\dagger$. The theory from Ref. [1]
works on the excitation indices.

Only the dipole moment operator is implemented.

[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal
of Chemical Physics 93, 3333 (1990).
"""
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.meta.coordinates import Descartes, CARTESIAN
from numpy.typing import NDArray
from rspn.ghf_ccsd.equations.op.ref import (
    get_mux_ref,
    get_muy_ref,
    get_muz_ref,
)
from rspn.ghf_ccsd.equations.op.singles import (
    get_mux_singles,
    get_muy_singles,
    get_muz_singles,
)
from rspn.ghf_ccsd.equations.op.doubles import (
    get_mux_doubles,
    get_muy_doubles,
    get_muz_doubles,
)


def build_nu_bar_V_cc(
    input: GHF_Generators_Input,
) -> dict[Descartes, dict[str, NDArray]]:
    """ V stands for the electric dipole operator. """
    xi_mu = {}
    for coord in CARTESIAN:
        xi_mu[coord] = _build_helper(coord, input)
    return xi_mu


def _build_helper(
    coord: Descartes,
    input: GHF_Generators_Input,
) -> dict[str, NDArray]:

    if coord == Descartes.x:
        return {
            'ref': get_mux_ref(**input),
            'singles': get_mux_singles(**input),
            'doubles': get_mux_doubles(**input),
        }

    elif coord == Descartes.y:
        return {
            'ref': get_muy_ref(**input),
            'singles': get_muy_singles(**input),
            'doubles': get_muy_doubles(**input),
        }

    elif coord == Descartes.z:
        return {
            'ref': get_muz_ref(**input),
            'singles': get_muz_singles(**input),
            'doubles': get_muz_doubles(**input),
        }

    else:
        raise ValueError(f"Unknown cartesian coordinate: {coord}.")
