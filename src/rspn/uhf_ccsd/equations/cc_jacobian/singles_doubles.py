from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_singles_singles_aaaaaa(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_aaaa[oa, oa, va, oa])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_aaaa[oa, va, va, va])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    return singles_singles_aaaaaa


def get_singles_singles_aaabab(
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
    
    contracted_intermediate =  1.00 * einsum('ac,ik,jb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aaabab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_abab[oa, ob, va, ob])
    singles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    singles_singles_aaabab +=  1.00 * einsum('ik,jabc->aibckj', kd_bb[ob, ob], g_aaaa[oa, va, va, va])
    singles_singles_aaabab +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jlbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return singles_singles_aaabab


def get_singles_singles_aaabba(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aaabba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kjbi->aibckj', kd_aa[va, va], g_abab[oa, ob, va, ob])
    singles_singles_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    singles_singles_aaabba += -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_aaaa[oa, va, va, va])
    singles_singles_aaabba += -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kjbd,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return singles_singles_aaabba


def get_singles_singles_aabaab(
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
    
    singles_singles_aabaab =  1.00 * einsum('ab,ij,kc->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aabaab += -1.00 * einsum('ab,jkic->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    singles_singles_aabaab +=  1.00 * einsum('ij,akbc->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    singles_singles_aabaab += -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aabaab += -1.00 * einsum('ab,jkdc,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aabaab +=  1.00 * einsum('ab,ij,lkdc,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_aabaab += -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return singles_singles_aabaab


def get_singles_singles_aababa(
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
    
    singles_singles_aababa = -1.00 * einsum('ab,ik,jc->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aababa +=  1.00 * einsum('ab,kjic->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    singles_singles_aababa += -1.00 * einsum('ik,ajbc->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    singles_singles_aababa +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aababa +=  1.00 * einsum('ab,kjdc,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aababa += -1.00 * einsum('ab,ik,ljdc,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_aababa +=  1.00 * einsum('ab,ik,ljcd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return singles_singles_aababa


def get_singles_singles_aabbbb(
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
    
    contracted_intermediate =  1.00 * einsum('ab,ij,kc->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_aabbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_aabbbb +=  1.00 * einsum('ab,jkci->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob])
    contracted_intermediate =  1.00 * einsum('ij,akbc->aibckj', kd_bb[ob, ob], g_abab[va, ob, va, vb])
    singles_singles_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_aabbbb +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ab,ij,lkdc,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return singles_singles_aabbbb


def get_singles_singles_abaaab(
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
    
    singles_singles_abaaab = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_abaaab +=  1.00 * einsum('ac,jkib->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    singles_singles_abaaab += -1.00 * einsum('ij,akcb->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    singles_singles_abaaab +=  1.00 * einsum('ij,lkcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abaaab +=  1.00 * einsum('ac,jkdb,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abaaab += -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_abaaab +=  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return singles_singles_abaaab


def get_singles_singles_abaaba(
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
    
    singles_singles_abaaba =  1.00 * einsum('ac,ik,jb->aibckj', kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_abaaba += -1.00 * einsum('ac,kjib->aibckj', kd_aa[va, va], g_abab[oa, ob, oa, vb])
    singles_singles_abaaba +=  1.00 * einsum('ik,ajcb->aibckj', kd_aa[oa, oa], g_abab[va, ob, va, vb])
    singles_singles_abaaba += -1.00 * einsum('ik,ljcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abaaba += -1.00 * einsum('ac,kjdb,di->aibckj', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_abaaba +=  1.00 * einsum('ac,ik,ljdb,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_abaaba += -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    return singles_singles_abaaba


def get_singles_singles_ababbb(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_ababbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_ababbb += -1.00 * einsum('ac,jkbi->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, ob])
    contracted_intermediate = -1.00 * einsum('ij,akcb->aibckj', kd_bb[ob, ob], g_abab[va, ob, va, vb])
    singles_singles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ij,lkcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_ababbb += -1.00 * einsum('ac,jkbd,di->aibckj', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    singles_singles_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return singles_singles_ababbb


def get_singles_singles_babaaa(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_babaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_babaaa += -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa])
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_abab[oa, vb, va, vb])
    singles_singles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ij,klbc,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_babaaa += -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return singles_singles_babaaa


def get_singles_singles_babbab(
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
    
    singles_singles_babbab =  1.00 * einsum('ac,ik,jb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_babbab += -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    singles_singles_babbab +=  1.00 * einsum('ik,jabc->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    singles_singles_babbab += -1.00 * einsum('ik,jlbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_babbab += -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_babbab += -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_babbab +=  1.00 * einsum('ac,ik,jlbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return singles_singles_babbab


def get_singles_singles_babbba(
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
    
    singles_singles_babbba = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_babbba +=  1.00 * einsum('ac,kjbi->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    singles_singles_babbba += -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    singles_singles_babbba +=  1.00 * einsum('ij,klbc,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_babbba +=  1.00 * einsum('ac,kjbd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_babbba +=  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_babbba += -1.00 * einsum('ac,ij,klbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return singles_singles_babbba


def get_singles_singles_bbaaaa(
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
    
    contracted_intermediate =  1.00 * einsum('ab,ij,kc->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_bbaaaa +=  1.00 * einsum('ab,jkci->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa])
    contracted_intermediate =  1.00 * einsum('ij,kacb->aibckj', kd_aa[oa, oa], g_abab[oa, vb, va, vb])
    singles_singles_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,klcb,al->aibckj', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    singles_singles_bbaaaa +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ab,ij,klcd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return singles_singles_bbaaaa


def get_singles_singles_bbabab(
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
    
    singles_singles_bbabab = -1.00 * einsum('ab,ik,jc->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbabab +=  1.00 * einsum('ab,jkci->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    singles_singles_bbabab += -1.00 * einsum('ik,jacb->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    singles_singles_bbabab +=  1.00 * einsum('ik,jlcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbabab +=  1.00 * einsum('ab,jkcd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbabab +=  1.00 * einsum('ab,ik,ljcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_bbabab += -1.00 * einsum('ab,ik,jlcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return singles_singles_bbabab


def get_singles_singles_bbabba(
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
    
    singles_singles_bbabba =  1.00 * einsum('ab,ij,kc->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbabba += -1.00 * einsum('ab,kjci->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, ob])
    singles_singles_bbabba +=  1.00 * einsum('ij,kacb->aibckj', kd_bb[ob, ob], g_abab[oa, vb, va, vb])
    singles_singles_bbabba += -1.00 * einsum('ij,klcb,al->aibckj', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbabba += -1.00 * einsum('ab,kjcd,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbabba += -1.00 * einsum('ab,ij,lkcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_bbabba +=  1.00 * einsum('ab,ij,klcd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    return singles_singles_bbabba


def get_singles_singles_bbbaab(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbbaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jkib->aibckj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb])
    singles_singles_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    singles_singles_bbbaab += -1.00 * einsum('ij,kabc->aibckj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb])
    singles_singles_bbbaab += -1.00 * einsum('ij,lkbc,al->aibckj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,jkdb,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return singles_singles_bbbaab


def get_singles_singles_bbbaba(
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
    
    contracted_intermediate =  1.00 * einsum('ac,ik,jb->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbbaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kjib->aibckj', kd_bb[vb, vb], g_abab[oa, ob, oa, vb])
    singles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    singles_singles_bbbaba +=  1.00 * einsum('ik,jabc->aibckj', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb])
    singles_singles_bbbaba +=  1.00 * einsum('ik,ljbc,al->aibckj', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,kjdb,di->aibckj', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,ljdb,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,ljbd,dl->aibckj', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    singles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return singles_singles_bbbaba


def get_singles_singles_bbbbbb(
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
    
    contracted_intermediate = -1.00 * einsum('ac,ij,kb->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], optimize=['einsum_path', (0, 1, 2)])
    singles_singles_bbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbi->aibckj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,kabc->aibckj', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ij,lkbc,al->aibckj', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jkbd,di->aibckj', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ij,lkdb,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ij,lkbd,dl->aibckj', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    singles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    return singles_singles_bbbbbb
