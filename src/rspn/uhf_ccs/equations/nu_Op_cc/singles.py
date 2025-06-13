from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.uhf_ccs import UHF_CCS_Data


def get_mux_aa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_x
    h_bb = uhf_data.mub_x
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    mux_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    mux_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    mux_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    mux_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    mux_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    mux_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return mux_aa


def get_mux_bb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_x
    h_bb = uhf_data.mub_x
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    mux_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    mux_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    mux_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    mux_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    mux_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    mux_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return mux_bb


def get_muy_aa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_y
    h_bb = uhf_data.mub_y
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    muy_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    muy_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    muy_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    muy_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    muy_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    muy_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return muy_aa


def get_muy_bb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_y
    h_bb = uhf_data.mub_y
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    muy_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    muy_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    muy_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    muy_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    muy_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    muy_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return muy_bb


def get_muz_aa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_z
    h_bb = uhf_data.mub_z
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    muz_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    muz_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    muz_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    muz_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    muz_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    muz_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return muz_aa


def get_muz_bb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = uhf_data.mua_z
    h_bb = uhf_data.mub_z
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    muz_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    muz_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    muz_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    muz_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    muz_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    muz_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return muz_bb
