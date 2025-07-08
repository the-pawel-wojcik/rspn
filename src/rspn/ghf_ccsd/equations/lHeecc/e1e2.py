from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe1e2cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibckj', g[o, o, v, v], l1)
    lhe1e2cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ic->aibckj', g[o, o, v, v], l1)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,ja->aibckj', g[o, o, v, v], l1)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->aibckj', g[o, o, v, o], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc +=  1.00 * einsum('jkal,ilbc->aibckj', g[o, o, v, o], l2)
    contracted_intermediate =  1.00 * einsum('kicl,jlab->aibckj', g[o, o, v, o], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idab,jkdc->aibckj', g[o, v, v, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdab,kidc->aibckj', g[o, v, v, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc +=  1.00 * einsum('idbc,jkda->aibckj', g[o, v, v, v], l2)
    contracted_intermediate = -1.00 * einsum('liab,dl,jkcd->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljab,dl,kicd->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc += -1.00 * einsum('jkad,dl,libc->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc += -1.00 * einsum('libc,dl,jkad->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('libd,dl,jkac->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicd,dl,ljab->aibckj', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->aibckj', f[o, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ib,jkac->aibckj', f[o, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc
