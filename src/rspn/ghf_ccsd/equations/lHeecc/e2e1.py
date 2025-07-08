from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe2e1cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'i', 'j', 'c', 'k') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate = -1.00 * einsum('kjab,ic->abijck', g[o, o, v, v], l1)
    lhe2e1cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abijck', g[o, o, v, v], l1)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abkjci', contracted_intermediate)  + -1.00000 * einsum('abijck->baijck', contracted_intermediate)  +  1.00000 * einsum('abijck->bakjci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abijck', g[o, o, v, v], l1)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abijck', g[o, o, v, o], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate)  + -1.00000 * einsum('abijck->baijck', contracted_intermediate)  +  1.00000 * einsum('abijck->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abijck', g[o, o, v, o], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abkjci', contracted_intermediate) 
    lhe2e1cc +=  1.00 * einsum('kicl,jlab->abijck', g[o, o, v, o], l2)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abijck', g[o, v, v, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->acijbk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abijck', g[o, v, v, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate)  + -1.00000 * einsum('abijck->baijck', contracted_intermediate)  +  1.00000 * einsum('abijck->bajick', contracted_intermediate) 
    lhe2e1cc +=  1.00 * einsum('kdbc,ijda->abijck', g[o, v, v, v], l2)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->acijbk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate)  + -1.00000 * einsum('abijck->baijck', contracted_intermediate)  +  1.00000 * einsum('abijck->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate)  + -1.00000 * einsum('abijck->baijck', contracted_intermediate)  +  1.00000 * einsum('abijck->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->baijck', contracted_intermediate) 
    lhe2e1cc += -1.00 * einsum('lkbc,dl,ijad->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abkjci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate) 
    lhe2e1cc += -1.00 * einsum('kicd,dl,ljab->abijck', g[o, o, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abijck', f[o, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abijck', f[o, v], l2)
    lhe2e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijck->abjick', contracted_intermediate) 
    return lhe2e1cc
