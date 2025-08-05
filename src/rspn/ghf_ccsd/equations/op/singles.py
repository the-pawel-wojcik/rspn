from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_pert_op_bar_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
    h: NDArray,  # ghf_data.mu[Descartes.x|y|z],
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2

    pert_op_bar_singles =  1.00 * einsum('ai->ai', h[v, o])
    pert_op_bar_singles += -1.00 * einsum('ji,aj->ai', h[o, o], t1)
    pert_op_bar_singles +=  1.00 * einsum('ab,bi->ai', h[v, v], t1)
    pert_op_bar_singles += -1.00 * einsum('jb,baij->ai', h[o, v], t2)
    pert_op_bar_singles += -1.00 * einsum('jb,aj,bi->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    return pert_op_bar_singles
