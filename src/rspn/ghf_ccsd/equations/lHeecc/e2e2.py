from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe2e2cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'i', 'j', 'c', 'd', 'k', 'l') """
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc =  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abijcdlk', contracted_intermediate) 
    lhe2e2cc +=  1.00 * einsum('klab,ijcd->abijcdkl', g[o, o, v, v], l2)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abljcdki', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abljdcki', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abkjcdil', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abkjdcil', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abljcdki', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abljdcki', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abkjcdil', contracted_intermediate)  + -1.00000 * einsum('abijcdkl->abijdckl', contracted_intermediate)  +  1.00000 * einsum('abijcdkl->abkjdcil', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abljcdki', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abijcdkl', g[o, o, v, v], l2)
    lhe2e2cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abijcdkl->abjicdkl', contracted_intermediate) 
    lhe2e2cc +=  1.00 * einsum('licd,kjab->abijcdkl', g[o, o, v, v], l2)
    return lhe2e2cc
