from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_cc_j_singles_doubles_aaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_aaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_aaaa[oa, oa, va, oa])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_aaaa[oa, va, va, va])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    return cc_j_singles_doubles_aaaaaa


def get_cc_j_singles_doubles_aaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_aaabab = -1.00 * einsum('ab,ik,jc->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_aaabab +=  1.00 * einsum('ab,kjic->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_aaabab += -1.00 * einsum('ik,ajbc->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_aaabab +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaabab +=  1.00 * einsum('ab,kjdc,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaabab += -1.00 * einsum('ab,ik,ljdc,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aaabab +=  1.00 * einsum('ab,ik,ljcd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return cc_j_singles_doubles_aaabab


def get_cc_j_singles_doubles_aaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_aaabba =  1.00 * einsum('ab,ij,kc->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_aaabba += -1.00 * einsum('ab,jkic->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_aaabba +=  1.00 * einsum('ij,akbc->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_aaabba += -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaabba += -1.00 * einsum('ab,jkdc,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aaabba +=  1.00 * einsum('ab,ij,lkdc,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aaabba += -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return cc_j_singles_doubles_aaabba


def get_cc_j_singles_doubles_aabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_aabaab =  1.00 * einsum('ac,ik,jb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_aabaab += -1.00 * einsum('ac,kjib->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_aabaab +=  1.00 * einsum('ik,ajcb->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_aabaab += -1.00 * einsum('ik,ljcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aabaab += -1.00 * einsum('ac,kjdb,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aabaab +=  1.00 * einsum('ac,ik,ljdb,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aabaab += -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return cc_j_singles_doubles_aabaab


def get_cc_j_singles_doubles_aababa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_aababa = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_aababa +=  1.00 * einsum('ac,jkib->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_aababa += -1.00 * einsum('ij,akcb->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_aababa +=  1.00 * einsum('ij,lkcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aababa +=  1.00 * einsum('ac,jkdb,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_aababa += -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_aababa +=  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return cc_j_singles_doubles_aababa


def get_cc_j_singles_doubles_abaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_abaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kjbi->aibckj', kd_aa[va, va], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    cc_j_singles_doubles_abaaab += -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_aaaa[oa, va, va, va])
    cc_j_singles_doubles_abaaab += -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kjbd,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return cc_j_singles_doubles_abaaab


def get_cc_j_singles_doubles_abaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate =  1.00 * einsum('ac,ik,jb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_abaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    cc_j_singles_doubles_abaaba +=  1.00 * einsum('ik,jabc->aibckj', kd_bb[ob, ob], g_aaaa[oa, va, va, va])
    cc_j_singles_doubles_abaaba +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jlbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return cc_j_singles_doubles_abaaba


def get_cc_j_singles_doubles_ababbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate =  1.00 * einsum('ab,ij,kc->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_ababbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_ababbb +=  1.00 * einsum('ab,jkci->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob])
    contracted_intermediate =  1.00 * einsum('ij,akbc->aibckj', kd_bb[ob, ob], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_ababbb +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ab,ij,lkdc,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return cc_j_singles_doubles_ababbb


def get_cc_j_singles_doubles_abbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_abbabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_abbabb += -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob])
    contracted_intermediate = -1.00 * einsum('ij,akcb->aibckj', kd_bb[ob, ob], g_abab[va, ob, va, vb])
    cc_j_singles_doubles_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ij,lkcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_abbabb += -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    cc_j_singles_doubles_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return cc_j_singles_doubles_abbabb


def get_cc_j_singles_doubles_baabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_baabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_baabaa += -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa])
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ij,klbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_baabaa += -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return cc_j_singles_doubles_baabaa


def get_cc_j_singles_doubles_babaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate =  1.00 * einsum('ab,ij,kc->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_babaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_babaaa +=  1.00 * einsum('ab,jkci->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa])
    contracted_intermediate =  1.00 * einsum('ij,kacb->aibckj', kd_aa[oa, oa], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,klcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    cc_j_singles_doubles_babaaa +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ab,ij,klcd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return cc_j_singles_doubles_babaaa


def get_cc_j_singles_doubles_babbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate =  1.00 * einsum('ac,ik,jb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_babbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kjib->aibckj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    cc_j_singles_doubles_babbab +=  1.00 * einsum('ik,jabc->aibckj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb])
    cc_j_singles_doubles_babbab +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,kjdb,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,ljdb,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return cc_j_singles_doubles_babbab


def get_cc_j_singles_doubles_babbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_babbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jkib->aibckj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb])
    cc_j_singles_doubles_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    cc_j_singles_doubles_babbba += -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb])
    cc_j_singles_doubles_babbba += -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,jkdb,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    cc_j_singles_doubles_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return cc_j_singles_doubles_babbba


def get_cc_j_singles_doubles_bbabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_bbabab = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_bbabab +=  1.00 * einsum('ac,kjbi->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_bbabab += -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_bbabab +=  1.00 * einsum('ij,klbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabab +=  1.00 * einsum('ac,kjbd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabab +=  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabab += -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return cc_j_singles_doubles_bbabab


def get_cc_j_singles_doubles_bbabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_bbabba =  1.00 * einsum('ac,ik,jb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_bbabba += -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_bbabba +=  1.00 * einsum('ik,jabc->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_bbabba += -1.00 * einsum('ik,jlbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabba += -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabba += -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbabba +=  1.00 * einsum('ac,ik,jlbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return cc_j_singles_doubles_bbabba


def get_cc_j_singles_doubles_bbbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_bbbaab =  1.00 * einsum('ab,ij,kc->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_bbbaab += -1.00 * einsum('ab,kjci->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_bbbaab +=  1.00 * einsum('ij,kacb->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_bbbaab += -1.00 * einsum('ij,klcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaab += -1.00 * einsum('ab,kjcd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaab += -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaab +=  1.00 * einsum('ab,ij,klcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return cc_j_singles_doubles_bbbaab


def get_cc_j_singles_doubles_bbbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    cc_j_singles_doubles_bbbaba = -1.00 * einsum('ab,ik,jc->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_bbbaba +=  1.00 * einsum('ab,jkci->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    cc_j_singles_doubles_bbbaba += -1.00 * einsum('ik,jacb->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    cc_j_singles_doubles_bbbaba +=  1.00 * einsum('ik,jlcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaba +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaba +=  1.00 * einsum('ab,ik,ljcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbaba += -1.00 * einsum('ab,ik,jlcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return cc_j_singles_doubles_bbbaba


def get_cc_j_singles_doubles_bbbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'c', 'k', 'j') """
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    cc_j_singles_doubles_bbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    cc_j_singles_doubles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    return cc_j_singles_doubles_bbbbbb
