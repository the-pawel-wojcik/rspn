from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_mux_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_aa)
    mux_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    mux_aaaa +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_aaaa)
    mux_aaaa +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_aaaa)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_aa[oa, oa], t2_aaaa)
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_aaaa)
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    mux_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    mux_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return mux_aaaa


def get_mux_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    mux_abab = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_bb)
    mux_abab += -1.00 * einsum('bi,aj->abji', h_bb[vb, ob], t1_aa)
    mux_abab += -1.00 * einsum('kk,abji->abji', h_aa[oa, oa], t2_abab)
    mux_abab += -1.00 * einsum('kk,abji->abji', h_bb[ob, ob], t2_abab)
    mux_abab +=  1.00 * einsum('kj,abki->abji', h_aa[oa, oa], t2_abab)
    mux_abab +=  1.00 * einsum('ki,abjk->abji', h_bb[ob, ob], t2_abab)
    mux_abab += -1.00 * einsum('ac,cbji->abji', h_aa[va, va], t2_abab)
    mux_abab += -1.00 * einsum('bc,acji->abji', h_bb[vb, vb], t2_abab)
    mux_abab += -1.00 * einsum('kc,abji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab += -1.00 * einsum('kc,abji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,abki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,abjk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,ak,cbji->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,acji,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab += -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,cajk,bi->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab += -1.00 * einsum('kc,acjk,bi->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab +=  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abab +=  1.00 * einsum('ki,aj,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab += -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab += -1.00 * einsum('bc,aj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    mux_abab +=  1.00 * einsum('kc,aj,bk,ci->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    return mux_abab


def get_mux_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    mux_abba =  1.00 * einsum('bj,ai->abji', h_bb[vb, ob], t1_aa)
    mux_abba +=  1.00 * einsum('ai,bj->abji', h_aa[va, oa], t1_bb)
    mux_abba +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_abab)
    mux_abba +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_abab)
    mux_abba += -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_abab)
    mux_abba += -1.00 * einsum('ki,abkj->abji', h_aa[oa, oa], t2_abab)
    mux_abba +=  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_abab)
    mux_abba +=  1.00 * einsum('bc,acij->abji', h_bb[vb, vb], t2_abab)
    mux_abba +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba += -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba += -1.00 * einsum('kc,abkj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba += -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba += -1.00 * einsum('kc,acij,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba += -1.00 * einsum('kc,caik,bj->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba +=  1.00 * einsum('kc,acik,bj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba +=  1.00 * einsum('kc,ai,cbkj->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba += -1.00 * einsum('kc,ai,cbjk->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba += -1.00 * einsum('kj,ai,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba += -1.00 * einsum('ki,ak,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_abba +=  1.00 * einsum('bc,ai,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba +=  1.00 * einsum('ac,bj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_abba += -1.00 * einsum('kc,ai,bk,cj->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    mux_abba += -1.00 * einsum('kc,ak,bj,ci->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    return mux_abba


def get_mux_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    mux_baab =  1.00 * einsum('bj,ai->abji', h_aa[va, oa], t1_bb)
    mux_baab +=  1.00 * einsum('ai,bj->abji', h_bb[vb, ob], t1_aa)
    mux_baab +=  1.00 * einsum('kk,baji->abji', h_aa[oa, oa], t2_abab)
    mux_baab +=  1.00 * einsum('kk,baji->abji', h_bb[ob, ob], t2_abab)
    mux_baab += -1.00 * einsum('kj,baki->abji', h_aa[oa, oa], t2_abab)
    mux_baab += -1.00 * einsum('ki,bajk->abji', h_bb[ob, ob], t2_abab)
    mux_baab +=  1.00 * einsum('ac,bcji->abji', h_bb[vb, vb], t2_abab)
    mux_baab +=  1.00 * einsum('bc,caji->abji', h_aa[va, va], t2_abab)
    mux_baab +=  1.00 * einsum('kc,baji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kc,baji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab += -1.00 * einsum('kc,baki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab += -1.00 * einsum('kc,bajk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab += -1.00 * einsum('kc,ak,bcji->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab += -1.00 * einsum('kc,caji,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kc,caki,bj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab += -1.00 * einsum('kc,caik,bj->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab += -1.00 * einsum('kc,ai,cbjk->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kc,ai,bcjk->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab += -1.00 * einsum('kj,ai,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab += -1.00 * einsum('ki,ak,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baab +=  1.00 * einsum('bc,ai,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('ac,bj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_baab += -1.00 * einsum('kc,ai,bk,cj->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    mux_baab += -1.00 * einsum('kc,ak,bj,ci->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    return mux_baab


def get_mux_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    mux_baba = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_aa)
    mux_baba += -1.00 * einsum('bi,aj->abji', h_aa[va, oa], t1_bb)
    mux_baba += -1.00 * einsum('kk,baij->abji', h_aa[oa, oa], t2_abab)
    mux_baba += -1.00 * einsum('kk,baij->abji', h_bb[ob, ob], t2_abab)
    mux_baba +=  1.00 * einsum('kj,baik->abji', h_bb[ob, ob], t2_abab)
    mux_baba +=  1.00 * einsum('ki,bakj->abji', h_aa[oa, oa], t2_abab)
    mux_baba += -1.00 * einsum('ac,bcij->abji', h_bb[vb, vb], t2_abab)
    mux_baba += -1.00 * einsum('bc,caij->abji', h_aa[va, va], t2_abab)
    mux_baba += -1.00 * einsum('kc,baij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('kc,baij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,baik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,bakj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,ak,bcij->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,caij,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('kc,cakj,bi->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,cajk,bi->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba +=  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_baba +=  1.00 * einsum('ki,aj,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('bc,aj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_baba +=  1.00 * einsum('kc,aj,bk,ci->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    return mux_baba


def get_mux_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_x
    h_bb = uhf_scf_data.mub_x
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_bb)
    mux_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    mux_bbbb +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_bbbb)
    mux_bbbb +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_bbbb)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_bbbb)
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_bb[vb, vb], t2_bbbb)
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    mux_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    mux_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return mux_bbbb


def get_muy_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_aa)
    muy_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muy_aaaa +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_aaaa)
    muy_aaaa +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_aaaa)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_aa[oa, oa], t2_aaaa)
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_aaaa)
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muy_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muy_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muy_aaaa


def get_muy_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muy_abab = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_bb)
    muy_abab += -1.00 * einsum('bi,aj->abji', h_bb[vb, ob], t1_aa)
    muy_abab += -1.00 * einsum('kk,abji->abji', h_aa[oa, oa], t2_abab)
    muy_abab += -1.00 * einsum('kk,abji->abji', h_bb[ob, ob], t2_abab)
    muy_abab +=  1.00 * einsum('kj,abki->abji', h_aa[oa, oa], t2_abab)
    muy_abab +=  1.00 * einsum('ki,abjk->abji', h_bb[ob, ob], t2_abab)
    muy_abab += -1.00 * einsum('ac,cbji->abji', h_aa[va, va], t2_abab)
    muy_abab += -1.00 * einsum('bc,acji->abji', h_bb[vb, vb], t2_abab)
    muy_abab += -1.00 * einsum('kc,abji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab += -1.00 * einsum('kc,abji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,abki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,abjk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,ak,cbji->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,acji,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab += -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,cajk,bi->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab += -1.00 * einsum('kc,acjk,bi->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab +=  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abab +=  1.00 * einsum('ki,aj,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab += -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab += -1.00 * einsum('bc,aj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muy_abab +=  1.00 * einsum('kc,aj,bk,ci->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    return muy_abab


def get_muy_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muy_abba =  1.00 * einsum('bj,ai->abji', h_bb[vb, ob], t1_aa)
    muy_abba +=  1.00 * einsum('ai,bj->abji', h_aa[va, oa], t1_bb)
    muy_abba +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_abab)
    muy_abba +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_abab)
    muy_abba += -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_abab)
    muy_abba += -1.00 * einsum('ki,abkj->abji', h_aa[oa, oa], t2_abab)
    muy_abba +=  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_abab)
    muy_abba +=  1.00 * einsum('bc,acij->abji', h_bb[vb, vb], t2_abab)
    muy_abba +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba += -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba += -1.00 * einsum('kc,abkj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba += -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba += -1.00 * einsum('kc,acij,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba += -1.00 * einsum('kc,caik,bj->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba +=  1.00 * einsum('kc,acik,bj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba +=  1.00 * einsum('kc,ai,cbkj->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba += -1.00 * einsum('kc,ai,cbjk->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba += -1.00 * einsum('kj,ai,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba += -1.00 * einsum('ki,ak,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_abba +=  1.00 * einsum('bc,ai,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba +=  1.00 * einsum('ac,bj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_abba += -1.00 * einsum('kc,ai,bk,cj->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    muy_abba += -1.00 * einsum('kc,ak,bj,ci->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    return muy_abba


def get_muy_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muy_baab =  1.00 * einsum('bj,ai->abji', h_aa[va, oa], t1_bb)
    muy_baab +=  1.00 * einsum('ai,bj->abji', h_bb[vb, ob], t1_aa)
    muy_baab +=  1.00 * einsum('kk,baji->abji', h_aa[oa, oa], t2_abab)
    muy_baab +=  1.00 * einsum('kk,baji->abji', h_bb[ob, ob], t2_abab)
    muy_baab += -1.00 * einsum('kj,baki->abji', h_aa[oa, oa], t2_abab)
    muy_baab += -1.00 * einsum('ki,bajk->abji', h_bb[ob, ob], t2_abab)
    muy_baab +=  1.00 * einsum('ac,bcji->abji', h_bb[vb, vb], t2_abab)
    muy_baab +=  1.00 * einsum('bc,caji->abji', h_aa[va, va], t2_abab)
    muy_baab +=  1.00 * einsum('kc,baji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kc,baji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab += -1.00 * einsum('kc,baki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab += -1.00 * einsum('kc,bajk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab += -1.00 * einsum('kc,ak,bcji->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab += -1.00 * einsum('kc,caji,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kc,caki,bj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab += -1.00 * einsum('kc,caik,bj->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab += -1.00 * einsum('kc,ai,cbjk->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kc,ai,bcjk->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab += -1.00 * einsum('kj,ai,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab += -1.00 * einsum('ki,ak,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baab +=  1.00 * einsum('bc,ai,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('ac,bj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_baab += -1.00 * einsum('kc,ai,bk,cj->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    muy_baab += -1.00 * einsum('kc,ak,bj,ci->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    return muy_baab


def get_muy_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muy_baba = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_aa)
    muy_baba += -1.00 * einsum('bi,aj->abji', h_aa[va, oa], t1_bb)
    muy_baba += -1.00 * einsum('kk,baij->abji', h_aa[oa, oa], t2_abab)
    muy_baba += -1.00 * einsum('kk,baij->abji', h_bb[ob, ob], t2_abab)
    muy_baba +=  1.00 * einsum('kj,baik->abji', h_bb[ob, ob], t2_abab)
    muy_baba +=  1.00 * einsum('ki,bakj->abji', h_aa[oa, oa], t2_abab)
    muy_baba += -1.00 * einsum('ac,bcij->abji', h_bb[vb, vb], t2_abab)
    muy_baba += -1.00 * einsum('bc,caij->abji', h_aa[va, va], t2_abab)
    muy_baba += -1.00 * einsum('kc,baij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('kc,baij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,baik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,bakj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,ak,bcij->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,caij,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('kc,cakj,bi->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,cajk,bi->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba +=  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_baba +=  1.00 * einsum('ki,aj,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('bc,aj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_baba +=  1.00 * einsum('kc,aj,bk,ci->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    return muy_baba


def get_muy_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_y
    h_bb = uhf_scf_data.mub_y
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_bb)
    muy_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muy_bbbb +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_bbbb)
    muy_bbbb +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_bbbb)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_bbbb)
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_bb[vb, vb], t2_bbbb)
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muy_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muy_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muy_bbbb


def get_muz_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_aa)
    muz_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muz_aaaa +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_aaaa)
    muz_aaaa +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_aaaa)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_aa[oa, oa], t2_aaaa)
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_aaaa)
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muz_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aaaa +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muz_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muz_aaaa


def get_muz_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muz_abab = -1.00 * einsum('aj,bi->abji', h_aa[va, oa], t1_bb)
    muz_abab += -1.00 * einsum('bi,aj->abji', h_bb[vb, ob], t1_aa)
    muz_abab += -1.00 * einsum('kk,abji->abji', h_aa[oa, oa], t2_abab)
    muz_abab += -1.00 * einsum('kk,abji->abji', h_bb[ob, ob], t2_abab)
    muz_abab +=  1.00 * einsum('kj,abki->abji', h_aa[oa, oa], t2_abab)
    muz_abab +=  1.00 * einsum('ki,abjk->abji', h_bb[ob, ob], t2_abab)
    muz_abab += -1.00 * einsum('ac,cbji->abji', h_aa[va, va], t2_abab)
    muz_abab += -1.00 * einsum('bc,acji->abji', h_bb[vb, vb], t2_abab)
    muz_abab += -1.00 * einsum('kc,abji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab += -1.00 * einsum('kc,abji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,abki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,abjk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,ak,cbji->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,acji,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab += -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,cajk,bi->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab += -1.00 * einsum('kc,acjk,bi->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab +=  1.00 * einsum('kj,ak,bi->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abab +=  1.00 * einsum('ki,aj,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab += -1.00 * einsum('ac,bi,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab += -1.00 * einsum('bc,aj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_abab += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,ak,bi,cj->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    muz_abab +=  1.00 * einsum('kc,aj,bk,ci->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    return muz_abab


def get_muz_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muz_abba =  1.00 * einsum('bj,ai->abji', h_bb[vb, ob], t1_aa)
    muz_abba +=  1.00 * einsum('ai,bj->abji', h_aa[va, oa], t1_bb)
    muz_abba +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_abab)
    muz_abba +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_abab)
    muz_abba += -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_abab)
    muz_abba += -1.00 * einsum('ki,abkj->abji', h_aa[oa, oa], t2_abab)
    muz_abba +=  1.00 * einsum('ac,cbij->abji', h_aa[va, va], t2_abab)
    muz_abba +=  1.00 * einsum('bc,acij->abji', h_bb[vb, vb], t2_abab)
    muz_abba +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba += -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba += -1.00 * einsum('kc,abkj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba += -1.00 * einsum('kc,ak,cbij->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba += -1.00 * einsum('kc,acij,bk->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba += -1.00 * einsum('kc,caik,bj->abji', h_aa[oa, va], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba +=  1.00 * einsum('kc,acik,bj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba +=  1.00 * einsum('kc,ai,cbkj->abji', h_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba += -1.00 * einsum('kc,ai,cbjk->abji', h_bb[ob, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba += -1.00 * einsum('kj,ai,bk->abji', h_bb[ob, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba += -1.00 * einsum('ki,ak,bj->abji', h_aa[oa, oa], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_abba +=  1.00 * einsum('bc,ai,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba +=  1.00 * einsum('ac,bj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_abba +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_abba += -1.00 * einsum('kc,ai,bk,cj->abji', h_bb[ob, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    muz_abba += -1.00 * einsum('kc,ak,bj,ci->abji', h_aa[oa, va], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    return muz_abba


def get_muz_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muz_baab =  1.00 * einsum('bj,ai->abji', h_aa[va, oa], t1_bb)
    muz_baab +=  1.00 * einsum('ai,bj->abji', h_bb[vb, ob], t1_aa)
    muz_baab +=  1.00 * einsum('kk,baji->abji', h_aa[oa, oa], t2_abab)
    muz_baab +=  1.00 * einsum('kk,baji->abji', h_bb[ob, ob], t2_abab)
    muz_baab += -1.00 * einsum('kj,baki->abji', h_aa[oa, oa], t2_abab)
    muz_baab += -1.00 * einsum('ki,bajk->abji', h_bb[ob, ob], t2_abab)
    muz_baab +=  1.00 * einsum('ac,bcji->abji', h_bb[vb, vb], t2_abab)
    muz_baab +=  1.00 * einsum('bc,caji->abji', h_aa[va, va], t2_abab)
    muz_baab +=  1.00 * einsum('kc,baji,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kc,baji,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab += -1.00 * einsum('kc,baki,cj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab += -1.00 * einsum('kc,bajk,ci->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab += -1.00 * einsum('kc,ak,bcji->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab += -1.00 * einsum('kc,caji,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kc,caki,bj->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab += -1.00 * einsum('kc,caik,bj->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab += -1.00 * einsum('kc,ai,cbjk->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kc,ai,bcjk->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kk,ai,bj->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab +=  1.00 * einsum('kk,ai,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab += -1.00 * einsum('kj,ai,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab += -1.00 * einsum('ki,ak,bj->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baab +=  1.00 * einsum('bc,ai,cj->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('ac,bj,ci->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_baab +=  1.00 * einsum('kc,ai,bj,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_baab += -1.00 * einsum('kc,ai,bk,cj->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    muz_baab += -1.00 * einsum('kc,ak,bj,ci->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    return muz_baab


def get_muz_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    muz_baba = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_aa)
    muz_baba += -1.00 * einsum('bi,aj->abji', h_aa[va, oa], t1_bb)
    muz_baba += -1.00 * einsum('kk,baij->abji', h_aa[oa, oa], t2_abab)
    muz_baba += -1.00 * einsum('kk,baij->abji', h_bb[ob, ob], t2_abab)
    muz_baba +=  1.00 * einsum('kj,baik->abji', h_bb[ob, ob], t2_abab)
    muz_baba +=  1.00 * einsum('ki,bakj->abji', h_aa[oa, oa], t2_abab)
    muz_baba += -1.00 * einsum('ac,bcij->abji', h_bb[vb, vb], t2_abab)
    muz_baba += -1.00 * einsum('bc,caij->abji', h_aa[va, va], t2_abab)
    muz_baba += -1.00 * einsum('kc,baij,ck->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('kc,baij,ck->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,baik,cj->abji', h_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,bakj,ci->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,ak,bcij->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,caij,bk->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,aj,cbik->abji', h_aa[oa, va], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('kc,aj,bcik->abji', h_bb[ob, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('kc,cakj,bi->abji', h_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,cajk,bi->abji', h_bb[ob, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba += -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba += -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba +=  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_baba +=  1.00 * einsum('ki,aj,bk->abji', h_aa[oa, oa], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('bc,aj,ci->abji', h_aa[va, va], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_baba += -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_baba +=  1.00 * einsum('kc,aj,bk,ci->abji', h_aa[oa, va], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    return muz_baba


def get_muz_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    h_aa = uhf_scf_data.mua_z
    h_bb = uhf_scf_data.mub_z
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    contracted_intermediate = -1.00 * einsum('aj,bi->abji', h_bb[vb, ob], t1_bb)
    muz_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    muz_bbbb +=  1.00 * einsum('kk,abij->abji', h_aa[oa, oa], t2_bbbb)
    muz_bbbb +=  1.00 * einsum('kk,abij->abji', h_bb[ob, ob], t2_bbbb)
    contracted_intermediate = -1.00 * einsum('kj,abik->abji', h_bb[ob, ob], t2_bbbb)
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abji', h_bb[vb, vb], t2_bbbb)
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    muz_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_aa[oa, va], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00 * einsum('kc,abij,ck->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abji', h_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,cbki->abji', h_aa[oa, va], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,aj,cbik->abji', h_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_aa[oa, oa], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kk,aj,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kj,ak,bi->abji', h_bb[ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abji', h_bb[vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_aa[oa, va], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,aj,bi,ck->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kc,ak,bi,cj->abji', h_bb[ob, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    muz_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    return muz_bbbb
