from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe1e2cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'j', 'k') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibcjk', g[o, o, v, v], l1)
    lhe1e2cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate)  + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ic->aibcjk', g[o, o, v, v], l1)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,ja->aibcjk', g[o, o, v, v], l1)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->aibcjk', g[o, o, v, o], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate)  + -1.00000 * einsum('aibcjk->biacjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->biackj', contracted_intermediate) 
    lhe1e2cc +=  1.00 * einsum('jkal,ilbc->aibcjk', g[o, o, v, o], l2)
    contracted_intermediate =  1.00 * einsum('kicl,jlab->aibcjk', g[o, o, v, o], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idab,jkdc->aibcjk', g[o, v, v, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->akbcji', contracted_intermediate)  + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->akcbji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdab,kidc->aibcjk', g[o, v, v, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    lhe1e2cc +=  1.00 * einsum('idbc,jkda->aibcjk', g[o, v, v, v], l2)
    contracted_intermediate = -1.00 * einsum('liab,dl,jkcd->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->akbcji', contracted_intermediate)  + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->akcbji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate)  + -1.00000 * einsum('aibcjk->biacjk', contracted_intermediate)  +  1.00000 * einsum('aibcjk->biackj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljab,dl,kicd->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    lhe1e2cc += -1.00 * einsum('jkad,dl,libc->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    lhe1e2cc += -1.00 * einsum('libc,dl,jkad->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('libd,dl,jkac->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicd,dl,ljab->aibcjk', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->aibcjk', f[o, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aibckj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ib,jkac->aibcjk', f[o, v], l2)
    lhe1e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibcjk->aicbjk', contracted_intermediate) 
    return lhe1e2cc
