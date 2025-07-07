from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_eta(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
    operator: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h = operator
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    contracted_intermediate = -1.00 * einsum('ja,ib->abji', h[o, v], l1)
    eta =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,ikab->abji', h[o, o], l2)
    eta +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ca,ijcb->abji', h[v, v], l2)
    eta +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ck,ijbc->abji', h[o, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    eta +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,ck,kiab->abji', h[o, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    eta +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    return eta
