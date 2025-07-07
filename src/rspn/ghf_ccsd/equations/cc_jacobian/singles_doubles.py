from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_singles_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
    f = ghf_data.f
    g = ghf_data.g
    kd =  ghf_data.identity_singles
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd[v, v], g[o, o, v, o])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd[o, o], g[o, v, v, v])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    return cc_j_singles_doubles
