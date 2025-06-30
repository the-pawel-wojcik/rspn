from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE, UHF_CCSD_Data
from chem.ccsd.equations.util import GeneratorsInput
from chem.hf.intermediates_builders import Intermediates
from numpy.typing import NDArray
import numpy as np
from rspn.uhf_ccsd.equations.cc_jacobian_contract_Rain.singles import (
    get_cc_j_w_singles_aa,
    get_cc_j_w_singles_bb,
)
from rspn.uhf_ccsd.equations.cc_jacobian_contract_Rain.doubles import (
    get_cc_j_w_doubles_aaaa,
    get_cc_j_w_doubles_abab,
    get_cc_j_w_doubles_abba,
    get_cc_j_w_doubles_baab,
    get_cc_j_w_doubles_baba,
    get_cc_j_w_doubles_bbbb,
)
from scipy.sparse.linalg import LinearOperator


class Minus_UHF_CCSD_Jacobian_action(LinearOperator):
    """ gmres helper.
    Takes a vector v and returns the vector w = -A @ v
    where A is the matrix of the CC Jacobian. 

    Clever in the way that is does not store the matrix A. """


    def __init__(
        self,
        uhf_hf_data: Intermediates,
        uhf_ccsd_data: UHF_CCSD_Data,
    ) -> None:
        self.uhf_hf_data = uhf_hf_data
        self.kwargs = GeneratorsInput(
            uhf_scf_data=uhf_hf_data,
            uhf_ccsd_data=uhf_ccsd_data,
        )
        dims, slices, shapes = Spin_MBE.find_dims_slices_shapes(uhf_hf_data)
        self.dims = dims
        self.slices = slices
        self.shapes = shapes

        jacobian_dimension = Spin_MBE.get_vector_dim(dims)

        # the ones below are required by LinearOperator
        shape = (jacobian_dimension, jacobian_dimension)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x: NDArray) -> NDArray:

        inp_vec = Spin_MBE.from_flattened_NDArray(x, self.uhf_hf_data)
        sigma_elegant = Spin_MBE()
        sigma_elegant.singles[E1_spin.aa] = get_cc_j_w_singles_aa(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.singles[E1_spin.bb] = get_cc_j_w_singles_bb(
            **self.kwargs,
            vector=inp_vec,
        )

        sigma_elegant.doubles[E2_spin.aaaa] = get_cc_j_w_doubles_aaaa(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.doubles[E2_spin.abab] = get_cc_j_w_doubles_abab(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.doubles[E2_spin.abba] = get_cc_j_w_doubles_abba(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.doubles[E2_spin.baab] = get_cc_j_w_doubles_baab(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.doubles[E2_spin.baba] = get_cc_j_w_doubles_baba(
            **self.kwargs,
            vector=inp_vec,
        )
        sigma_elegant.doubles[E2_spin.bbbb] = get_cc_j_w_doubles_bbbb(
            **self.kwargs,
            vector=inp_vec,
        )

        return -sigma_elegant.flatten()
