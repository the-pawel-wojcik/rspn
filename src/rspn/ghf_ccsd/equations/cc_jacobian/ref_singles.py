from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_ref_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    
    cc_j_ref_singles =  1.00 * einsum('ia->ai', f[o, v])
    cc_j_ref_singles += -1.00 * einsum('jiab,bj->ai', g[o, o, v, v], t1)
    return cc_j_ref_singles
