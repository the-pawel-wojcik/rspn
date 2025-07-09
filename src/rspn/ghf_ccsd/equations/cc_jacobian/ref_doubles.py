from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_ref_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'i', 'j') """
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    
    cc_j_ref_doubles =  1.00 * einsum('ijab->abij', g[o, o, v, v])
    return cc_j_ref_doubles
