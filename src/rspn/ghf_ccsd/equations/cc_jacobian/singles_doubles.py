from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_singles_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'j', 'k') """
    f = ghf_data.f
    g = ghf_data.g
    kd =  ghf_data.identity_singles
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibcjk', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate)  + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibcjk', kd[v, v], g[o, o, v, o])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibcjk', kd[o, o], g[o, v, v, v])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibcjk', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibcjk', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibcjk', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate)  + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->aicbkj', contracted_intermediate) 
    return cc_j_singles_doubles
