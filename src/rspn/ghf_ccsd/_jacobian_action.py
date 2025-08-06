from chem.ccsd.containers import GHF_CCSD_Data
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE, GHF_ov_data
from numpy.typing import NDArray
import numpy as np
from rspn.ghf_ccsd.equations.cc_jacobian_action.singles import (
    get_cc_j_w_singles,
)
from rspn.ghf_ccsd.equations.cc_jacobian_action.doubles import (
    get_cc_j_w_doubles,
)
from scipy.sparse.linalg import LinearOperator


class Minus_GHF_CCSD_Jacobian_action(LinearOperator):
    """ gmres helper.
    Takes a vector v and returns the vector w = -A @ v
    where A is the matrix of the CC Jacobian. 

    Clever in the way that is does not store the matrix A. """


    def __init__(
        self,
        ghf_data: GHF_Data,
        ghf_ccsd_data: GHF_CCSD_Data,
    ) -> None:
        self.ghf_data = ghf_data
        self.kwargs = GHF_Generators_Input(
            ghf_data=ghf_data,
            ghf_ccsd_data=ghf_ccsd_data,
        )
        nv = ghf_data.nv
        no = ghf_data.no
        jacobian_dimension = no * nv + (no * nv)**2
        # the ones below are required by LinearOperator
        shape = (jacobian_dimension, jacobian_dimension)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x: NDArray) -> NDArray:
        nmo = self.ghf_data.nmo
        nv = self.ghf_data.nv
        no = self.ghf_data.no
        ghf_ov_data = GHF_ov_data(nmo=nmo, no=no, nv=nv)
        inp_vec = GHF_CCSD_MBE.from_NDArray(x, ghf_ov_data)
        sigma_singles = get_cc_j_w_singles(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_doubles = get_cc_j_w_doubles(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma = GHF_CCSD_MBE(
            singles=sigma_singles,
            doubles=sigma_doubles,
        )

        return -sigma.flatten()
