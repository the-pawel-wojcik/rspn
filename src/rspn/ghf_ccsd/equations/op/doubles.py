from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.coordinates import Descartes


def get_mux_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h = ghf_data.mu[Descartes.x]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h[v, o], t1)
    mux_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    mux_doubles +=  1.00 * einsum('kk,abij->abji', h[o, o], t2)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h[o, o], t2)
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h[v, v], t2)
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    mux_doubles +=  1.00 * einsum('kc,abij,ck->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h[v, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    mux_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return mux_doubles


def get_muy_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h = ghf_data.mu[Descartes.y]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h[v, o], t1)
    muy_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muy_doubles +=  1.00 * einsum('kk,abij->abji', h[o, o], t2)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h[o, o], t2)
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h[v, v], t2)
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muy_doubles +=  1.00 * einsum('kc,abij,ck->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h[v, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muy_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muy_doubles


def get_muz_doubles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h = ghf_data.mu[Descartes.z]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h[v, o], t1)
    muz_doubles =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muz_doubles +=  1.00 * einsum('kk,abij->abji', h[o, o], t2)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h[o, o], t2)
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h[v, v], t2)
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muz_doubles +=  1.00 * einsum('kc,abij,ck->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h[o, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h[v, v], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h[o, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muz_doubles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muz_doubles
