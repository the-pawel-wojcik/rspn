from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.coordinates import Descartes


def get_mux_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h = ghf_data.mu[Descartes.x]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    mux_singles =  1.00 * einsum('ai->ai', h[v, o])
    mux_singles +=  1.00 * einsum('jj,ai->ai', h[o, o], t1)
    mux_singles += -1.00 * einsum('ji,aj->ai', h[o, o], t1)
    mux_singles +=  1.00 * einsum('ab,bi->ai', h[v, v], t1)
    mux_singles += -1.00 * einsum('jb,baij->ai', h[o, v], t2)
    mux_singles +=  1.00 * einsum('jb,ai,bj->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_singles += -1.00 * einsum('jb,aj,bi->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    return mux_singles


def get_muy_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h = ghf_data.mu[Descartes.y]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    muy_singles =  1.00 * einsum('ai->ai', h[v, o])
    muy_singles +=  1.00 * einsum('jj,ai->ai', h[o, o], t1)
    muy_singles += -1.00 * einsum('ji,aj->ai', h[o, o], t1)
    muy_singles +=  1.00 * einsum('ab,bi->ai', h[v, v], t1)
    muy_singles += -1.00 * einsum('jb,baij->ai', h[o, v], t2)
    muy_singles +=  1.00 * einsum('jb,ai,bj->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_singles += -1.00 * einsum('jb,aj,bi->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    return muy_singles


def get_muz_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h = ghf_data.mu[Descartes.z]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    muz_singles =  1.00 * einsum('ai->ai', h[v, o])
    muz_singles +=  1.00 * einsum('jj,ai->ai', h[o, o], t1)
    muz_singles += -1.00 * einsum('ji,aj->ai', h[o, o], t1)
    muz_singles +=  1.00 * einsum('ab,bi->ai', h[v, v], t1)
    muz_singles += -1.00 * einsum('jb,baij->ai', h[o, v], t2)
    muz_singles +=  1.00 * einsum('jb,ai,bj->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_singles += -1.00 * einsum('jb,aj,bi->ai', h[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    return muz_singles
