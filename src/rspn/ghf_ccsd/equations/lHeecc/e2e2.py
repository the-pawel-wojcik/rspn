from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe2e2cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc +=  1.00 * einsum('klab,ijcd->abjicdlk', g[o, o, v, v], l2)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc +=  1.00 * einsum('licd,kjab->abjicdlk', g[o, o, v, v], l2)
    return lhe2e2cc
