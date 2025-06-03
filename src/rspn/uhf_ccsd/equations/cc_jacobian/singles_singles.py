from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_singles_singles_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_aaaa = -1.00 * einsum('ab,ji->aibj', kd_aa[va, va], f_aa[oa, oa])
    singles_singles_aaaa +=  1.00 * einsum('ij,ab->aibj', kd_aa[oa, oa], f_aa[va, va])
    singles_singles_aaaa += -1.00 * einsum('ij,kb,ak->aibj', kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ab,jc,ci->aibj', kd_aa[va, va], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  1.00 * einsum('jabi->aibj', g_aaaa[oa, va, va, oa])
    singles_singles_aaaa +=  1.00 * einsum('kjbi,ak->aibj', g_aaaa[oa, oa, va, oa], t1_aa)
    singles_singles_aaaa += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ab,jkic,ck->aibj', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  1.00 * einsum('jabc,ci->aibj', g_aaaa[oa, va, va, va], t1_aa)
    singles_singles_aaaa += -1.00 * einsum('ij,kabc,ck->aibj', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  1.00 * einsum('ij,akbc,ck->aibj', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  1.00 * einsum('kjbc,caik->aibj', g_aaaa[oa, oa, va, va], t2_aaaa)
    singles_singles_aaaa +=  1.00 * einsum('jkbc,acik->aibj', g_abab[oa, ob, va, vb], t2_abab)
    singles_singles_aaaa +=  0.50 * einsum('ij,lkbc,calk->aibj', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -0.50 * einsum('ij,lkbc,aclk->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -0.50 * einsum('ij,klbc,ackl->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  0.50 * einsum('ab,kjcd,cdik->aibj', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -0.50 * einsum('ab,jkcd,cdik->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa += -0.50 * einsum('ab,jkdc,dcik->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaa +=  1.00 * einsum('kjbc,ak,ci->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    singles_singles_aaaa += -1.00 * einsum('ab,jkdc,ck,di->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return singles_singles_aaaa


def get_singles_singles_aabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_aabb =  1.00 * einsum('ajib->aibj', g_abab[va, ob, oa, vb])
    singles_singles_aabb += -1.00 * einsum('kjib,ak->aibj', g_abab[oa, ob, oa, vb], t1_aa)
    singles_singles_aabb +=  1.00 * einsum('ajcb,ci->aibj', g_abab[va, ob, va, vb], t1_aa)
    singles_singles_aabb += -1.00 * einsum('kjcb,caik->aibj', g_abab[oa, ob, va, vb], t2_aaaa)
    singles_singles_aabb += -1.00 * einsum('kjbc,acik->aibj', g_bbbb[ob, ob, vb, vb], t2_abab)
    singles_singles_aabb += -1.00 * einsum('kjcb,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return singles_singles_aabb


def get_singles_singles_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_abab = -1.00 * einsum('ab,ji->aibj', kd_aa[va, va], f_bb[ob, ob])
    singles_singles_abab +=  1.00 * einsum('ij,ab->aibj', kd_bb[ob, ob], f_aa[va, va])
    singles_singles_abab += -1.00 * einsum('ij,kb,ak->aibj', kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ab,jc,ci->aibj', kd_aa[va, va], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ajbi->aibj', g_abab[va, ob, va, ob])
    singles_singles_abab +=  1.00 * einsum('kjbi,ak->aibj', g_abab[oa, ob, va, ob], t1_aa)
    singles_singles_abab += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ajbc,ci->aibj', g_abab[va, ob, va, vb], t1_bb)
    singles_singles_abab += -1.00 * einsum('ij,kabc,ck->aibj', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab +=  1.00 * einsum('ij,akbc,ck->aibj', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab +=  1.00 * einsum('kjbc,acki->aibj', g_abab[oa, ob, va, vb], t2_abab)
    singles_singles_abab +=  0.50 * einsum('ij,lkbc,calk->aibj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -0.50 * einsum('ij,lkbc,aclk->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -0.50 * einsum('ij,klbc,ackl->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -0.50 * einsum('ab,kjcd,cdki->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab += -0.50 * einsum('ab,kjdc,dcki->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab +=  0.50 * einsum('ab,kjcd,cdik->aibj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abab +=  1.00 * einsum('kjbc,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    singles_singles_abab += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return singles_singles_abab


def get_singles_singles_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_baba = -1.00 * einsum('ab,ji->aibj', kd_bb[vb, vb], f_aa[oa, oa])
    singles_singles_baba +=  1.00 * einsum('ij,ab->aibj', kd_aa[oa, oa], f_bb[vb, vb])
    singles_singles_baba += -1.00 * einsum('ij,kb,ak->aibj', kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ab,jc,ci->aibj', kd_bb[vb, vb], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('jaib->aibj', g_abab[oa, vb, oa, vb])
    singles_singles_baba +=  1.00 * einsum('jkib,ak->aibj', g_abab[oa, ob, oa, vb], t1_bb)
    singles_singles_baba += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ab,jkic,ck->aibj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('jacb,ci->aibj', g_abab[oa, vb, va, vb], t1_aa)
    singles_singles_baba +=  1.00 * einsum('ij,kacb,ck->aibj', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ij,kabc,ck->aibj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba +=  1.00 * einsum('jkcb,caik->aibj', g_abab[oa, ob, va, vb], t2_abab)
    singles_singles_baba += -0.50 * einsum('ij,lkcb,calk->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -0.50 * einsum('ij,klcb,cakl->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba +=  0.50 * einsum('ij,lkbc,calk->aibj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba +=  0.50 * einsum('ab,kjcd,cdik->aibj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -0.50 * einsum('ab,jkcd,cdik->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba += -0.50 * einsum('ab,jkdc,dcik->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_baba +=  1.00 * einsum('jkcb,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ij,klcb,al,ck->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    singles_singles_baba += -1.00 * einsum('ab,jkdc,ck,di->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return singles_singles_baba


def get_singles_singles_bbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_bbaa =  1.00 * einsum('jabi->aibj', g_abab[oa, vb, va, ob])
    singles_singles_bbaa += -1.00 * einsum('jkbi,ak->aibj', g_abab[oa, ob, va, ob], t1_bb)
    singles_singles_bbaa +=  1.00 * einsum('jabc,ci->aibj', g_abab[oa, vb, va, vb], t1_bb)
    singles_singles_bbaa += -1.00 * einsum('kjbc,caki->aibj', g_aaaa[oa, oa, va, va], t2_abab)
    singles_singles_bbaa += -1.00 * einsum('jkbc,caik->aibj', g_abab[oa, ob, va, vb], t2_bbbb)
    singles_singles_bbaa += -1.00 * einsum('jkbc,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return singles_singles_bbaa


def get_singles_singles_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    singles_singles_bbbb = -1.00 * einsum('ab,ji->aibj', kd_bb[vb, vb], f_bb[ob, ob])
    singles_singles_bbbb +=  1.00 * einsum('ij,ab->aibj', kd_bb[ob, ob], f_bb[vb, vb])
    singles_singles_bbbb += -1.00 * einsum('ij,kb,ak->aibj', kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ab,jc,ci->aibj', kd_bb[vb, vb], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  1.00 * einsum('jabi->aibj', g_bbbb[ob, vb, vb, ob])
    singles_singles_bbbb +=  1.00 * einsum('kjbi,ak->aibj', g_bbbb[ob, ob, vb, ob], t1_bb)
    singles_singles_bbbb += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  1.00 * einsum('jabc,ci->aibj', g_bbbb[ob, vb, vb, vb], t1_bb)
    singles_singles_bbbb +=  1.00 * einsum('ij,kacb,ck->aibj', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ij,kabc,ck->aibj', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  1.00 * einsum('kjcb,caki->aibj', g_abab[oa, ob, va, vb], t2_abab)
    singles_singles_bbbb +=  1.00 * einsum('kjbc,caik->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb)
    singles_singles_bbbb += -0.50 * einsum('ij,lkcb,calk->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -0.50 * einsum('ij,klcb,cakl->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  0.50 * einsum('ij,lkbc,calk->aibj', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -0.50 * einsum('ab,kjcd,cdki->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb += -0.50 * einsum('ab,kjdc,dcki->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  0.50 * einsum('ab,kjcd,cdik->aibj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbb +=  1.00 * einsum('kjbc,ak,ci->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ij,klcb,al,ck->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    singles_singles_bbbb += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return singles_singles_bbbb
