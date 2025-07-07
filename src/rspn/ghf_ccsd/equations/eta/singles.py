from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_eta(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
    operator: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h = operator
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    eta =  1.00 * einsum('ia->ai', h[o, v])
    eta += -1.00 * einsum('ij,ja->ai', h[o, o], l1)
    eta +=  1.00 * einsum('ba,ib->ai', h[v, v], l1)
    eta += -1.00 * einsum('ja,bj,ib->ai', h[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    eta += -1.00 * einsum('ib,bj,ja->ai', h[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    eta += -0.50 * einsum('ka,cbjk,jicb->ai', h[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    eta += -0.50 * einsum('ic,cbjk,jkab->ai', h[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    return eta
