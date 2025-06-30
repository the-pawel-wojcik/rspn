from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.uhf_ccs import UHF_CCS_Data


def get_cc_j_singles_singles_aaaa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    cc_j_singles_singles_aaaa = -1.00 * einsum('ab,ji->aibj', kd_aa[va, va], f_aa[oa, oa])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('ij,ab->aibj', kd_aa[oa, oa], f_aa[va, va])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ij,kb,ak->aibj', kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ab,jc,ci->aibj', kd_aa[va, va], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('jabi->aibj', g_aaaa[oa, va, va, oa])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('kjbi,ak->aibj', g_aaaa[oa, oa, va, oa], t1_aa)
    cc_j_singles_singles_aaaa += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ab,jkic,ck->aibj', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('jabc,ci->aibj', g_aaaa[oa, va, va, va], t1_aa)
    cc_j_singles_singles_aaaa += -1.00 * einsum('ij,kabc,ck->aibj', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('ij,akbc,ck->aibj', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa +=  1.00 * einsum('kjbc,ak,ci->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_singles_singles_aaaa += -1.00 * einsum('ab,jkdc,ck,di->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return cc_j_singles_singles_aaaa


def get_cc_j_singles_singles_aabb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    cc_j_singles_singles_aabb =  1.00 * einsum('ajib->aibj', g_abab[va, ob, oa, vb])
    cc_j_singles_singles_aabb += -1.00 * einsum('kjib,ak->aibj', g_abab[oa, ob, oa, vb], t1_aa)
    cc_j_singles_singles_aabb +=  1.00 * einsum('ajcb,ci->aibj', g_abab[va, ob, va, vb], t1_aa)
    cc_j_singles_singles_aabb += -1.00 * einsum('kjcb,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    return cc_j_singles_singles_aabb


def get_cc_j_singles_singles_abab(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    cc_j_singles_singles_abab = -1.00 * einsum('ab,ji->aibj', kd_aa[va, va], f_bb[ob, ob])
    cc_j_singles_singles_abab +=  1.00 * einsum('ij,ab->aibj', kd_bb[ob, ob], f_aa[va, va])
    cc_j_singles_singles_abab += -1.00 * einsum('ij,kb,ak->aibj', kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ab,jc,ci->aibj', kd_aa[va, va], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ajbi->aibj', g_abab[va, ob, va, ob])
    cc_j_singles_singles_abab +=  1.00 * einsum('kjbi,ak->aibj', g_abab[oa, ob, va, ob], t1_aa)
    cc_j_singles_singles_abab += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ab,kjci,ck->aibj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ajbc,ci->aibj', g_abab[va, ob, va, vb], t1_bb)
    cc_j_singles_singles_abab += -1.00 * einsum('ij,kabc,ck->aibj', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab +=  1.00 * einsum('ij,akbc,ck->aibj', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_abab +=  1.00 * einsum('kjbc,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_singles_singles_abab += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return cc_j_singles_singles_abab


def get_cc_j_singles_singles_baba(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    cc_j_singles_singles_baba = -1.00 * einsum('ab,ji->aibj', kd_bb[vb, vb], f_aa[oa, oa])
    cc_j_singles_singles_baba +=  1.00 * einsum('ij,ab->aibj', kd_aa[oa, oa], f_bb[vb, vb])
    cc_j_singles_singles_baba += -1.00 * einsum('ij,kb,ak->aibj', kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ab,jc,ci->aibj', kd_bb[vb, vb], f_aa[oa, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('jaib->aibj', g_abab[oa, vb, oa, vb])
    cc_j_singles_singles_baba +=  1.00 * einsum('jkib,ak->aibj', g_abab[oa, ob, oa, vb], t1_bb)
    cc_j_singles_singles_baba += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ab,jkic,ck->aibj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('jacb,ci->aibj', g_abab[oa, vb, va, vb], t1_aa)
    cc_j_singles_singles_baba +=  1.00 * einsum('ij,kacb,ck->aibj', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ij,kabc,ck->aibj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_baba +=  1.00 * einsum('jkcb,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ij,klcb,al,ck->aibj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_singles_singles_baba += -1.00 * einsum('ab,jkdc,ck,di->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return cc_j_singles_singles_baba


def get_cc_j_singles_singles_bbaa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    cc_j_singles_singles_bbaa =  1.00 * einsum('jabi->aibj', g_abab[oa, vb, va, ob])
    cc_j_singles_singles_bbaa += -1.00 * einsum('jkbi,ak->aibj', g_abab[oa, ob, va, ob], t1_bb)
    cc_j_singles_singles_bbaa +=  1.00 * einsum('jabc,ci->aibj', g_abab[oa, vb, va, vb], t1_bb)
    cc_j_singles_singles_bbaa += -1.00 * einsum('jkbc,ak,ci->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    return cc_j_singles_singles_bbaa


def get_cc_j_singles_singles_bbbb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb

    cc_j_singles_singles_bbbb = -1.00 * einsum('ab,ji->aibj', kd_bb[vb, vb], f_bb[ob, ob])
    cc_j_singles_singles_bbbb +=  1.00 * einsum('ij,ab->aibj', kd_bb[ob, ob], f_bb[vb, vb])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ij,kb,ak->aibj', kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ab,jc,ci->aibj', kd_bb[vb, vb], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb +=  1.00 * einsum('jabi->aibj', g_bbbb[ob, vb, vb, ob])
    cc_j_singles_singles_bbbb +=  1.00 * einsum('kjbi,ak->aibj', g_bbbb[ob, ob, vb, ob], t1_bb)
    cc_j_singles_singles_bbbb += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ab,kjci,ck->aibj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb +=  1.00 * einsum('jabc,ci->aibj', g_bbbb[ob, vb, vb, vb], t1_bb)
    cc_j_singles_singles_bbbb +=  1.00 * einsum('ij,kacb,ck->aibj', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ij,kabc,ck->aibj', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb +=  1.00 * einsum('kjbc,ak,ci->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ij,klcb,al,ck->aibj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_singles_singles_bbbb += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return cc_j_singles_singles_bbbb
