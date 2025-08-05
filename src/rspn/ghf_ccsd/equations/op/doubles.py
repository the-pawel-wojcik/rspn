from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_pert_op_bar_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
    h: NDArray,  # ghf_data.mu[Descartes.x|y|z],
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'i', 'j') """
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2

    contracted_intermediate = -1.00 * einsum('kj,abik->abij', h[o, o], t2)
    pert_op_bar_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abij', h[v, v], t2)
    pert_op_bar_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abij', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    pert_op_bar_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abij', h[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    pert_op_bar_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    return pert_op_bar_doubles
