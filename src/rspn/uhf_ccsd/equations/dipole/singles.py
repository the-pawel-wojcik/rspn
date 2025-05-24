from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_mux_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    dipole_singles_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    dipole_singles_aa += -1.00 * einsum('jb,baij->ai', h_aa[oa, va], t2_aaaa)
    dipole_singles_aa +=  1.00 * einsum('jb,abij->ai', h_bb[ob, vb], t2_abab)
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return dipole_singles_aa


def get_mux_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jb,baji->ai', h_aa[oa, va], t2_abab)
    dipole_singles_bb += -1.00 * einsum('jb,baij->ai', h_bb[ob, vb], t2_bbbb)
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return dipole_singles_bb


def get_muz_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    dipole_singles_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    dipole_singles_aa += -1.00 * einsum('jb,baij->ai', h_aa[oa, va], t2_aaaa)
    dipole_singles_aa +=  1.00 * einsum('jb,abij->ai', h_bb[ob, vb], t2_abab)
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return dipole_singles_aa


def get_muz_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jb,baji->ai', h_aa[oa, va], t2_abab)
    dipole_singles_bb += -1.00 * einsum('jb,baij->ai', h_bb[ob, vb], t2_bbbb)
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return dipole_singles_bb


def get_muy_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_aa =  1.00 * einsum('ai->ai', h_aa[va, oa])
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_aa)
    dipole_singles_aa += -1.00 * einsum('ji,aj->ai', h_aa[oa, oa], t1_aa)
    dipole_singles_aa +=  1.00 * einsum('ab,bi->ai', h_aa[va, va], t1_aa)
    dipole_singles_aa += -1.00 * einsum('jb,baij->ai', h_aa[oa, va], t2_aaaa)
    dipole_singles_aa +=  1.00 * einsum('jb,abij->ai', h_bb[ob, vb], t2_abab)
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_aa += -1.00 * einsum('jb,aj,bi->ai', h_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return dipole_singles_aa


def get_muy_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    kd_aa =  uhf_scf_data.identity_aa
    kd_bb =  uhf_scf_data.identity_bb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    dipole_singles_bb =  1.00 * einsum('ai->ai', h_bb[vb, ob])
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_aa[oa, oa], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jj,ai->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb += -1.00 * einsum('ji,aj->ai', h_bb[ob, ob], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('ab,bi->ai', h_bb[vb, vb], t1_bb)
    dipole_singles_bb +=  1.00 * einsum('jb,baji->ai', h_aa[oa, va], t2_abab)
    dipole_singles_bb += -1.00 * einsum('jb,baij->ai', h_bb[ob, vb], t2_bbbb)
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_aa[oa, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    dipole_singles_bb += -1.00 * einsum('jb,aj,bi->ai', h_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return dipole_singles_bb
