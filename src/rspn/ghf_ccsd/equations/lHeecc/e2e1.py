from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe2e1cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate = -1.00 * einsum('kjab,ic->abjick', g[o, o, v, v], l1)
    lhe2e1cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g[o, o, v, v], l1)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g[o, o, v, v], l1)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjick', g[o, o, v, o], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjick', g[o, o, v, o], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc +=  1.00 * einsum('kicl,jlab->abjick', g[o, o, v, o], l2)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjick', g[o, v, v, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjick', g[o, v, v, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc +=  1.00 * einsum('kdbc,ijda->abjick', g[o, v, v, v], l2)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc += -1.00 * einsum('lkbc,dl,ijad->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc += -1.00 * einsum('kicd,dl,ljab->abjick', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f[o, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f[o, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc
