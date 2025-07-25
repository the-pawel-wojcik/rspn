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
from chem.meta.coordinates import Descartes
from numpy.typing import NDArray
from rspn.ghf_ccsd.equations.op.singles import get_pert_op_bar_singles
from rspn.ghf_ccsd.equations.op.doubles import get_pert_op_bar_doubles


def build_nu_bar_V_cc(
    input: GHF_Generators_Input,
) -> dict[Descartes, dict[str, NDArray]]:
    """ V stands for the electric dipole operator. """
    ghf_data = input['ghf_data']
    xi_mu = {
        dir: {
            'singles': get_pert_op_bar_singles(**input, h=ghf_data.mu[dir]),
            'doubles': get_pert_op_bar_doubles(**input, h=ghf_data.mu[dir]),
        } for dir in Descartes
    }
    return xi_mu
