from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.coordinates import Descartes


def get_mux_ref(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.x]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    
    mux_ref =  1.00 * einsum('ii', h[o, o])
    mux_ref +=  1.00 * einsum('ia,ai', h[o, v], t1)
    return mux_ref


def get_muy_ref(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.y]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    
    muy_ref =  1.00 * einsum('ii', h[o, o])
    muy_ref +=  1.00 * einsum('ia,ai', h[o, v], t1)
    return muy_ref


def get_muz_ref(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.z]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    
    muz_ref =  1.00 * einsum('ii', h[o, o])
    muz_ref +=  1.00 * einsum('ia,ai', h[o, v], t1)
    return muz_ref
