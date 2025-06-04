from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_cc_j_doubles_singles_aaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('jk,lc,abil->abjick', kd_aa[oa, oa], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kd,dbij->abjick', kd_aa[va, va], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaij->abjick', kd_aa[va, va], g_aaaa[oa, va, oa, oa])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,abcj->abjick', kd_aa[oa, oa], g_aaaa[va, va, va, oa])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkij,bl->abjick', kd_aa[va, va], g_aaaa[oa, oa, oa, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd_aa[va, va], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,abcd,di->abjick', kd_aa[oa, oa], g_aaaa[va, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcj,abil->abjick', g_aaaa[oa, oa, va, oa], t2_aaaa)
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,mlcj,abml->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kljd,bdil->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kacd,dbij->abjick', g_aaaa[oa, va, va, va], t2_aaaa)
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,dbil->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,alcd,bdil->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,kade,deij->abjick', kd_aa[va, va], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,abil,dj->abjick', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,al,dbij->abjick', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,dbim->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmcd,al,bdim->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kled,ebij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebil,dj->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,beil,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kade,dj,ei->abjick', kd_aa[va, va], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_aaaaaa


def get_cc_j_doubles_singles_aaaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('lkjc,abil->abjick', g_abab[oa, ob, oa, vb], t2_aaaa)
    cc_j_doubles_singles_aaaabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('akdc,dbij->abjick', g_abab[va, ob, va, vb], t2_aaaa)
    cc_j_doubles_singles_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkdc,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkdc,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_aaaabb


def get_cc_j_doubles_singles_aaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_aaabab =  1.00 * einsum('ik,lc,abjl->abjick', kd_bb[ob, ob], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,kd,bdji->abjick', kd_aa[va, va], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,akji->abjick', kd_aa[va, va], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab +=  1.00 * einsum('ik,abcj->abjick', kd_bb[ob, ob], g_aaaa[va, va, va, oa])
    contracted_intermediate =  1.00 * einsum('ac,lkji,bl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,akjd,di->abjick', kd_aa[va, va], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,akdi,dj->abjick', kd_aa[va, va], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab +=  1.00 * einsum('ik,abcd,dj->abjick', kd_bb[ob, ob], g_aaaa[va, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab += -1.00 * einsum('lkci,abjl->abjick', g_abab[oa, ob, va, ob], t2_aaaa)
    cc_j_doubles_singles_aaabab +=  0.50 * einsum('ik,mlcj,abml->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lkjd,bdli->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkdi,dbjl->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdi,bdjl->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('akcd,bdji->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacd,dbjl->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,alcd,bdjl->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,akde,deji->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,aked,edji->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab += -1.00 * einsum('lkcd,abjl,di->abjick', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkcd,al,bdji->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab +=  1.00 * einsum('ik,mlcd,abjm,dl->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00 * einsum('ik,mlcd,abjm,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  0.50 * einsum('ik,mlcd,abml,dj->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,mlcd,al,dbjm->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,lmcd,al,bdjm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,beji,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,beji,dl->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,beli,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lked,ebjl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,bejl,di->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,lkde,bl,deji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,lked,bl,edji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab += -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lkjd,bl,di->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkdi,bl,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacd,bl,dj->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,akde,dj,ei->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aaabab += -1.00 * einsum('ik,mlcd,al,bm,dj->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_aaabab


def get_cc_j_doubles_singles_aabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_aabaab = -1.00 * einsum('jk,lc,abil->abjick', kd_bb[ob, ob], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kd,bdij->abjick', kd_aa[va, va], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,akij->abjick', kd_aa[va, va], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab += -1.00 * einsum('jk,abci->abjick', kd_bb[ob, ob], g_aaaa[va, va, va, oa])
    contracted_intermediate = -1.00 * einsum('ac,lkij,bl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,laci,bl->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,akdj,di->abjick', kd_aa[va, va], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,akid,dj->abjick', kd_aa[va, va], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab += -1.00 * einsum('jk,abcd,di->abjick', kd_bb[ob, ob], g_aaaa[va, va, va, va], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00 * einsum('lkcj,abil->abjick', g_abab[oa, ob, va, ob], t2_aaaa)
    cc_j_doubles_singles_aabaab += -0.50 * einsum('jk,mlci,abml->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkdj,bdil->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkid,bdlj->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('akcd,bdij->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,dbil->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,alcd,bdil->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,akde,deij->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,aked,edij->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab +=  1.00 * einsum('lkcd,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkcd,al,bdij->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab += -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab += -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab += -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,dbim->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmcd,al,bdim->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,beij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,beij,dl->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lked,ebil,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,beil,dj->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,belj,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,lked,bl,edij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab +=  1.00 * einsum('jk,mlci,al,bm->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkid,bl,dj->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,aked,dj,ei->abjick', kd_aa[va, va], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_aabaab +=  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,lked,bl,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_aabaab


def get_cc_j_doubles_singles_abaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('jk,lc,abil->abjick', kd_aa[oa, oa], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('bc,kd,daij->abjick', kd_bb[vb, vb], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('bc,kaij->abjick', kd_bb[vb, vb], g_aaaa[oa, va, oa, oa])
    contracted_intermediate = -1.00 * einsum('ik,abjc->abjick', kd_aa[oa, oa], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('bc,lkij,al->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, oa, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ik,aljc,bl->abjick', kd_aa[oa, oa], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lbjc,al->abjick', kd_aa[oa, oa], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,abdc,di->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kljc,abil->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,mljc,abml->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,lmjc,ablm->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,lkdj,dail->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kljd,adil->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('kbdc,daij->abjick', g_abab[oa, vb, va, vb], t2_aaaa)
    contracted_intermediate = -1.00 * einsum('jk,aldc,dbil->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lbdc,dail->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lbcd,adil->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  0.50 * einsum('bc,kade,deij->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kldc,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba += -1.00 * einsum('kldc,daij,bl->abjick', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jk,lmdc,abim,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jk,mldc,abml,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jk,lmdc,ablm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmdc,al,dbim->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mldc,daim,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,adim,bl->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('bc,lkde,eaij,dl->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00 * einsum('bc,kled,eaij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bc,lkde,eail,dj->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,klde,aeil,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba +=  0.50 * einsum('bc,lkde,al,deij->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,lmjc,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,lkdj,al,di->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,aldc,bl,di->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lbdc,al,di->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba += -1.00 * einsum('bc,kade,dj,ei->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,lmdc,al,bm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abaaba += -1.00 * einsum('bc,lkde,al,dj,ei->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_abaaba


def get_cc_j_doubles_singles_ababaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_ababaa =  1.00 * einsum('jk,lc,abli->abjick', kd_aa[oa, oa], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kd,dbji->abjick', kd_aa[va, va], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kbji->abjick', kd_aa[va, va], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,abci->abjick', kd_aa[oa, oa], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,klji,bl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,alci,bl->abjick', kd_aa[oa, oa], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,lbci,al->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kbjd,di->abjick', kd_aa[va, va], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kbdi,dj->abjick', kd_aa[va, va], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,abcd,di->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('lkcj,abli->abjick', g_aaaa[oa, oa, va, oa], t2_abab)
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('klci,abjl->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_ababaa += -0.50 * einsum('jk,mlci,abml->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -0.50 * einsum('jk,lmci,ablm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,lkdj,dbli->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,kljd,dbil->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,kldi,dbjl->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('kacd,dbji->abjick', g_aaaa[oa, va, va, va], t2_abab)
    cc_j_doubles_singles_ababaa += -1.00 * einsum('kbcd,adji->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,lacd,dbli->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,alcd,dbil->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,lbcd,adli->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  0.50 * einsum('ac,kbde,deji->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  0.50 * einsum('ac,kbed,edji->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('lkcd,abli,dj->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('klcd,abjl,di->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('lkcd,al,dbji->abjick', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('klcd,adji,bl->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,mlcd,abmi,dl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,mlcd,abmi,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -0.50 * einsum('jk,lmcd,ablm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,mlcd,al,dbmi->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,lmcd,al,dbim->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,mlcd,admi,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,lkde,ebji,dl->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kled,ebji,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,lkde,ebli,dj->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,klde,ebil,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,kled,ebjl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -0.50 * einsum('ac,klde,bl,deji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -0.50 * einsum('ac,kled,bl,edji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,lmci,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,kljd,bl,di->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,kldi,bl,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,alcd,bl,di->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('jk,lbcd,al,di->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa +=  1.00 * einsum('ac,kbde,dj,ei->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('jk,lmcd,al,bm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababaa += -1.00 * einsum('ac,klde,bl,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_ababaa


def get_cc_j_doubles_singles_ababbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_ababbb =  1.00 * einsum('ik,lc,abjl->abjick', kd_bb[ob, ob], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,kd,adji->abjick', kd_bb[vb, vb], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,akji->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,abjc->abjick', kd_bb[ob, ob], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkji,al->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,aljc,bl->abjick', kd_bb[ob, ob], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,lbjc,al->abjick', kd_bb[ob, ob], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,akjd,di->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,akdi,dj->abjick', kd_bb[vb, vb], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,abdc,dj->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('lkjc,abli->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_ababbb += -1.00 * einsum('lkci,abjl->abjick', g_bbbb[ob, ob, vb, ob], t2_abab)
    cc_j_doubles_singles_ababbb += -0.50 * einsum('ik,mljc,abml->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -0.50 * einsum('ik,lmjc,ablm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkjd,adli->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkdi,dajl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,lkdi,adjl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('akdc,dbji->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_ababbb += -1.00 * einsum('kbcd,adji->abjick', g_bbbb[ob, vb, vb, vb], t2_abab)
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,aldc,dbjl->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,lbdc,dajl->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,lbcd,adjl->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  0.50 * einsum('bc,akde,deji->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  0.50 * einsum('bc,aked,edji->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('lkdc,abli,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('lkcd,abjl,di->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('lkdc,al,dbji->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('lkcd,adji,bl->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,lmdc,abjm,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,mlcd,abjm,dl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -0.50 * einsum('ik,mldc,abml,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -0.50 * einsum('ik,lmdc,ablm,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,lmdc,al,dbjm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,mldc,dajm,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,mlcd,adjm,bl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,lkde,aeji,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,lkde,aeji,dl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkde,aeli,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lked,eajl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkde,aejl,di->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -0.50 * einsum('bc,lkde,al,deji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -0.50 * einsum('bc,lked,al,edji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,lmjc,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkjd,al,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkdi,al,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,aldc,bl,dj->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('ik,lbdc,al,dj->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb +=  1.00 * einsum('bc,akde,dj,ei->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('ik,lmdc,al,bm,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_ababbb += -1.00 * einsum('bc,lkde,al,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_ababbb


def get_cc_j_doubles_singles_abbaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_abbaaa = -1.00 * einsum('ik,lc,ablj->abjick', kd_aa[oa, oa], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kd,dbij->abjick', kd_aa[va, va], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kbij->abjick', kd_aa[va, va], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,abcj->abjick', kd_aa[oa, oa], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,klij,bl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,alcj,bl->abjick', kd_aa[oa, oa], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,lbcj,al->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kbdj,di->abjick', kd_aa[va, va], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kbid,dj->abjick', kd_aa[va, va], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,abcd,dj->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('klcj,abil->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('lkci,ablj->abjick', g_aaaa[oa, oa, va, oa], t2_abab)
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ik,mlcj,abml->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ik,lmcj,ablm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,kldj,dbil->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,lkdi,dblj->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,klid,dbjl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('kacd,dbij->abjick', g_aaaa[oa, va, va, va], t2_abab)
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('kbcd,adij->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,lacd,dblj->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,alcd,dbjl->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,lbcd,adlj->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -0.50 * einsum('ac,kbde,deij->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -0.50 * einsum('ac,kbed,edij->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('klcd,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('lkcd,ablj,di->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('lkcd,al,dbij->abjick', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('klcd,adij,bl->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,mlcd,abmj,dl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,mlcd,abmj,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ik,mlcd,abml,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ik,lmcd,ablm,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,mlcd,al,dbmj->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,lmcd,al,dbjm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,mlcd,admj,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kled,ebij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,kled,ebil,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,lkde,eblj,di->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,klde,ebjl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ac,klde,bl,deij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  0.50 * einsum('ac,kled,bl,edij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,lmcj,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,kldj,bl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,klid,bl,dj->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,alcd,bl,dj->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ik,lbcd,al,dj->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa += -1.00 * einsum('ac,kbed,dj,ei->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ik,lmcd,al,bm,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbaaa +=  1.00 * einsum('ac,kled,bl,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_abbaaa


def get_cc_j_doubles_singles_abbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_abbabb = -1.00 * einsum('jk,lc,abil->abjick', kd_bb[ob, ob], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,kd,adij->abjick', kd_bb[vb, vb], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,akij->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,abic->abjick', kd_bb[ob, ob], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkij,al->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,alic,bl->abjick', kd_bb[ob, ob], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,lbic,al->abjick', kd_bb[ob, ob], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,akdj,di->abjick', kd_bb[vb, vb], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,akid,dj->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,abdc,di->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('lkcj,abil->abjick', g_bbbb[ob, ob, vb, ob], t2_abab)
    cc_j_doubles_singles_abbabb += -1.00 * einsum('lkic,ablj->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('jk,mlic,abml->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('jk,lmic,ablm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkdj,dail->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,lkdj,adil->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkid,adlj->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('akdc,dbij->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('kbcd,adij->abjick', g_bbbb[ob, vb, vb, vb], t2_abab)
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,aldc,dbil->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,lbdc,dail->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,lbcd,adil->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -0.50 * einsum('bc,akde,deij->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -0.50 * einsum('bc,aked,edij->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('lkcd,abil,dj->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('lkdc,ablj,di->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('lkdc,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('lkcd,adij,bl->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,lmdc,abim,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('jk,mldc,abml,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('jk,lmdc,ablm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,lmdc,al,dbim->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,mldc,daim,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,mlcd,adim,bl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,lkde,aeij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,lkde,aeij,dl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lked,eail,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkde,aeil,dj->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkde,aelj,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('bc,lkde,al,deij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  0.50 * einsum('bc,lked,al,edij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,lmic,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkdj,al,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lkid,al,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,aldc,bl,di->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('jk,lbdc,al,di->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb += -1.00 * einsum('bc,aked,dj,ei->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('jk,lmdc,al,bm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbabb +=  1.00 * einsum('bc,lked,al,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_abbabb


def get_cc_j_doubles_singles_abbbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate =  1.00 * einsum('jk,lc,abli->abjick', kd_bb[ob, ob], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -1.00 * einsum('ac,kd,dbij->abjick', kd_aa[va, va], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab += -1.00 * einsum('ac,kbij->abjick', kd_aa[va, va], g_bbbb[ob, vb, ob, ob])
    contracted_intermediate =  1.00 * einsum('ik,abcj->abjick', kd_bb[ob, ob], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -1.00 * einsum('ac,lkij,bl->abjick', kd_aa[va, va], g_bbbb[ob, ob, ob, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,alcj,bl->abjick', kd_bb[ob, ob], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,lbcj,al->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kbdj,di->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,abcd,di->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkcj,abli->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,mlcj,abml->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,lmcj,ablm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkdj,dbli->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -1.00 * einsum('akcd,dbij->abjick', g_abab[va, ob, va, vb], t2_bbbb)
    contracted_intermediate =  1.00 * einsum('jk,lacd,dbli->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,alcd,dbil->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lbcd,adli->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -0.50 * einsum('ac,kbde,deij->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkcd,abli,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab +=  1.00 * einsum('lkcd,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,mlcd,abmi,dl->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,abmi,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,lmcd,ablm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,al,dbmi->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmcd,al,dbim->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,admi,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab += -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lked,ebli,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebil,dj->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab += -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ik,lmcj,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,alcd,bl,di->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lbcd,al,di->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab +=  1.00 * einsum('ac,kbde,dj,ei->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jk,lmcd,al,bm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_abbbab +=  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_abbbab


def get_cc_j_doubles_singles_baaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate =  1.00 * einsum('jk,lc,bail->abjick', kd_aa[oa, oa], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -1.00 * einsum('ac,kd,dbij->abjick', kd_bb[vb, vb], f_aa[oa, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba += -1.00 * einsum('ac,kbij->abjick', kd_bb[vb, vb], g_aaaa[oa, va, oa, oa])
    contracted_intermediate =  1.00 * einsum('ik,bajc->abjick', kd_aa[oa, oa], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -1.00 * einsum('ac,lkij,bl->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, oa, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,lajc,bl->abjick', kd_aa[oa, oa], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,bljc,al->abjick', kd_aa[oa, oa], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kbdj,di->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, oa], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,badc,di->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kljc,bail->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,mljc,baml->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,lmjc,balm->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kljd,bdil->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -1.00 * einsum('kadc,dbij->abjick', g_abab[oa, vb, va, vb], t2_aaaa)
    contracted_intermediate =  1.00 * einsum('jk,ladc,dbil->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lacd,bdil->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,bldc,dail->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -0.50 * einsum('ac,kbde,deij->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kldc,bail,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba +=  1.00 * einsum('kldc,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,lmdc,baim,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,baim,dl->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,mldc,baml,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,lmdc,balm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mldc,al,dbim->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t2_aaaa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,al,bdim->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmdc,daim,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba += -1.00 * einsum('ac,kled,ebij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebil,dj->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,beil,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba += -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ik,mljc,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,ladc,bl,di->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,bldc,al,di->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba +=  1.00 * einsum('ac,kbde,dj,ei->abjick', kd_bb[vb, vb], g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jk,mldc,al,bm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_baaaba +=  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_baaaba


def get_cc_j_doubles_singles_baabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_baabaa = -1.00 * einsum('jk,lc,bali->abjick', kd_aa[oa, oa], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kd,daji->abjick', kd_aa[va, va], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kaji->abjick', kd_aa[va, va], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,baci->abjick', kd_aa[oa, oa], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,klji,al->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,laci,bl->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,blci,al->abjick', kd_aa[oa, oa], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kajd,di->abjick', kd_aa[va, va], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kadi,dj->abjick', kd_aa[va, va], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,bacd,di->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('lkcj,bali->abjick', g_aaaa[oa, oa, va, oa], t2_abab)
    cc_j_doubles_singles_baabaa += -1.00 * einsum('klci,bajl->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('jk,mlci,baml->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('jk,lmci,balm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,lkdj,dali->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,kljd,dail->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,kldi,dajl->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('kacd,bdji->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('kbcd,daji->abjick', g_aaaa[oa, va, va, va], t2_abab)
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,lacd,bdli->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,lbcd,dali->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,blcd,dail->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -0.50 * einsum('bc,kade,deji->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -0.50 * einsum('bc,kaed,edji->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('lkcd,bali,dj->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('klcd,bajl,di->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('klcd,al,bdji->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('lkcd,daji,bl->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,mlcd,bami,dl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,mlcd,bami,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('jk,mlcd,baml,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('jk,lmcd,balm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,mlcd,al,bdmi->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,mlcd,dami,bl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,lmcd,daim,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,lkde,eaji,dl->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kled,eaji,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,lkde,eali,dj->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,klde,eail,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,kled,eajl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('bc,klde,al,deji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  0.50 * einsum('bc,kled,al,edji->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,mlci,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,kljd,al,di->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,kldi,al,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,lacd,bl,di->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('jk,blcd,al,di->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa += -1.00 * einsum('bc,kade,dj,ei->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabaa +=  1.00 * einsum('bc,klde,al,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_baabaa


def get_cc_j_doubles_singles_baabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_baabbb = -1.00 * einsum('ik,lc,bajl->abjick', kd_bb[ob, ob], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,kd,bdji->abjick', kd_bb[vb, vb], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,bkji->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,bajc->abjick', kd_bb[ob, ob], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkji,bl->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,lajc,bl->abjick', kd_bb[ob, ob], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,bljc,al->abjick', kd_bb[ob, ob], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,bkjd,di->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,bkdi,dj->abjick', kd_bb[vb, vb], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,badc,dj->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('lkjc,bali->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('lkci,bajl->abjick', g_bbbb[ob, ob, vb, ob], t2_abab)
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ik,mljc,baml->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ik,lmjc,balm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkjd,bdli->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkdi,dbjl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,lkdi,bdjl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('kacd,bdji->abjick', g_bbbb[ob, vb, vb, vb], t2_abab)
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('bkdc,daji->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,ladc,dbjl->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,lacd,bdjl->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,bldc,dajl->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -0.50 * einsum('ac,bkde,deji->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -0.50 * einsum('ac,bked,edji->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('lkdc,bali,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('lkcd,bajl,di->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('lkcd,al,bdji->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('lkdc,daji,bl->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,lmdc,bajm,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,mlcd,bajm,dl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ik,mldc,baml,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ik,lmdc,balm,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,mldc,al,dbjm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t2_aaaa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,mlcd,al,bdjm->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,lmdc,dajm,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,lkde,beji,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,lkde,beji,dl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkde,beli,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lked,ebjl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkde,bejl,di->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ac,lkde,bl,deji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  0.50 * einsum('ac,lked,bl,edji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,mljc,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkjd,bl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkdi,bl,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,ladc,bl,dj->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ik,bldc,al,dj->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb += -1.00 * einsum('ac,bkde,dj,ei->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ik,mldc,al,bm,dj->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_baabbb +=  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_baabbb


def get_cc_j_doubles_singles_babaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_babaaa =  1.00 * einsum('ik,lc,balj->abjick', kd_aa[oa, oa], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kd,daij->abjick', kd_aa[va, va], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kaij->abjick', kd_aa[va, va], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,bacj->abjick', kd_aa[oa, oa], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,klij,al->abjick', kd_aa[va, va], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,lacj,bl->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,blcj,al->abjick', kd_aa[oa, oa], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kadj,di->abjick', kd_aa[va, va], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kaid,dj->abjick', kd_aa[va, va], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,bacd,dj->abjick', kd_aa[oa, oa], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('klcj,bail->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_babaaa += -1.00 * einsum('lkci,balj->abjick', g_aaaa[oa, oa, va, oa], t2_abab)
    cc_j_doubles_singles_babaaa += -0.50 * einsum('ik,mlcj,baml->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -0.50 * einsum('ik,lmcj,balm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,kldj,dail->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,lkdi,dalj->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,klid,dajl->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('kacd,bdij->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_babaaa += -1.00 * einsum('kbcd,daij->abjick', g_aaaa[oa, va, va, va], t2_abab)
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,lacd,bdlj->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,lbcd,dalj->abjick', kd_aa[oa, oa], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,blcd,dajl->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  0.50 * einsum('bc,kade,deij->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  0.50 * einsum('bc,kaed,edij->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('klcd,bail,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('lkcd,balj,di->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('klcd,al,bdij->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('lkcd,daij,bl->abjick', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,mlcd,bamj,dl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,mlcd,bamj,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -0.50 * einsum('ik,mlcd,baml,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -0.50 * einsum('ik,lmcd,balm,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,mlcd,al,bdmj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,mlcd,damj,bl->abjick', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,lmcd,dajm,bl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,lkde,eaij,dl->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kled,eaij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,kled,eail,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,lkde,ealj,di->abjick', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,klde,eajl,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -0.50 * einsum('bc,klde,al,deij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -0.50 * einsum('bc,kled,al,edij->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,kldj,al,di->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,klid,al,dj->abjick', kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,lacd,bl,dj->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('ik,blcd,al,dj->abjick', kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa +=  1.00 * einsum('bc,kaed,dj,ei->abjick', kd_aa[va, va], g_abab[oa, vb, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('ik,mlcd,al,bm,dj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babaaa += -1.00 * einsum('bc,kled,al,dj,ei->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_babaaa


def get_cc_j_doubles_singles_bababb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_bababb =  1.00 * einsum('jk,lc,bail->abjick', kd_bb[ob, ob], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,kd,bdij->abjick', kd_bb[vb, vb], f_bb[ob, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,bkij->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, ob])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,baic->abjick', kd_bb[ob, ob], g_abab[va, vb, oa, vb])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkij,bl->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,laic,bl->abjick', kd_bb[ob, ob], g_abab[oa, vb, oa, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,blic,al->abjick', kd_bb[ob, ob], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,bkdj,di->abjick', kd_bb[vb, vb], g_abab[va, ob, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,bkid,dj->abjick', kd_bb[vb, vb], g_abab[va, ob, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,badc,di->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('lkcj,bail->abjick', g_bbbb[ob, ob, vb, ob], t2_abab)
    cc_j_doubles_singles_bababb +=  1.00 * einsum('lkic,balj->abjick', g_abab[oa, ob, oa, vb], t2_abab)
    cc_j_doubles_singles_bababb += -0.50 * einsum('jk,mlic,baml->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -0.50 * einsum('jk,lmic,balm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkdj,dbil->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,lkdj,bdil->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkid,bdlj->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('kacd,bdij->abjick', g_bbbb[ob, vb, vb, vb], t2_abab)
    cc_j_doubles_singles_bababb += -1.00 * einsum('bkdc,daij->abjick', g_abab[va, ob, va, vb], t2_abab)
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,ladc,dbil->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,lacd,bdil->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,bldc,dail->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  0.50 * einsum('ac,bkde,deij->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  0.50 * einsum('ac,bked,edij->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('lkcd,bail,dj->abjick', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('lkdc,balj,di->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('lkcd,al,bdij->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('lkdc,daij,bl->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,lmdc,baim,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,mlcd,baim,dl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -0.50 * einsum('jk,mldc,baml,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -0.50 * einsum('jk,lmdc,balm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,mldc,al,dbim->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t2_aaaa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,mlcd,al,bdim->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,lmdc,daim,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,lkde,beij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,lkde,beij,dl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lked,ebil,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkde,beil,dj->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkde,belj,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -0.50 * einsum('ac,lked,bl,edij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,mlic,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lkid,bl,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,ladc,bl,di->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('jk,bldc,al,di->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb +=  1.00 * einsum('ac,bked,dj,ei->abjick', kd_bb[vb, vb], g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('jk,mldc,al,bm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bababb += -1.00 * einsum('ac,lked,bl,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_bababb


def get_cc_j_doubles_singles_babbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('jk,lc,bali->abjick', kd_bb[ob, ob], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bc,kd,daij->abjick', kd_aa[va, va], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bc,kaij->abjick', kd_aa[va, va], g_bbbb[ob, vb, ob, ob])
    contracted_intermediate = -1.00 * einsum('ik,bacj->abjick', kd_bb[ob, ob], g_abab[va, vb, va, ob])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bc,lkij,al->abjick', kd_aa[va, va], g_bbbb[ob, ob, ob, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,blcj,al->abjick', kd_bb[ob, ob], g_abab[va, ob, va, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,bacd,di->abjick', kd_bb[ob, ob], g_abab[va, vb, va, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcj,bali->abjick', g_abab[oa, ob, va, ob], t2_abab)
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,mlcj,baml->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,lmcj,balm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,lkdj,dali->abjick', kd_aa[va, va], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,lkdj,dail->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bkcd,daij->abjick', g_abab[va, ob, va, vb], t2_bbbb)
    contracted_intermediate = -1.00 * einsum('jk,lacd,bdli->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lbcd,dali->abjick', kd_bb[ob, ob], g_aaaa[oa, va, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,blcd,dail->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  0.50 * einsum('bc,kade,deij->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkcd,bali,dj->abjick', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab += -1.00 * einsum('lkcd,daij,bl->abjick', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jk,mlcd,bami,dl->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,bami,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jk,mlcd,baml,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jk,lmcd,balm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,bdmi->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,dami,bl->abjick', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmcd,daim,bl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bc,lkde,eaij,dl->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00 * einsum('bc,lkde,eaij,dl->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bc,lked,eali,dj->abjick', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,lkde,eail,dj->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab +=  0.50 * einsum('bc,lkde,al,deij->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,lkdj,al,di->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,blcd,al,di->abjick', kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab += -1.00 * einsum('bc,kade,dj,ei->abjick', kd_aa[va, va], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_babbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    cc_j_doubles_singles_babbab += -1.00 * einsum('bc,lkde,al,dj,ei->abjick', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    return cc_j_doubles_singles_babbab


def get_cc_j_doubles_singles_bbabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_bbabba = -1.00 * einsum('jk,lc,abil->abjick', kd_aa[oa, oa], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kd,dbji->abjick', kd_bb[vb, vb], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kaji->abjick', kd_bb[vb, vb], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba += -1.00 * einsum('jk,abci->abjick', kd_aa[oa, oa], g_bbbb[vb, vb, vb, ob])
    contracted_intermediate = -1.00 * einsum('ac,klji,bl->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,laci,bl->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kajd,di->abjick', kd_bb[vb, vb], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kadi,dj->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba += -1.00 * einsum('jk,abcd,di->abjick', kd_aa[oa, oa], g_bbbb[vb, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00 * einsum('kljc,abil->abjick', g_abab[oa, ob, oa, vb], t2_bbbb)
    cc_j_doubles_singles_bbabba += -0.50 * einsum('jk,mlci,abml->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,lkdj,dbli->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kljd,dbil->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kldi,dbjl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kadc,dbji->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,ladc,dbli->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,dbil->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,kade,deji->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,kaed,edji->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba +=  1.00 * einsum('kldc,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kldc,al,dbji->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba += -1.00 * einsum('jk,lmdc,abim,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba += -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba += -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jk,mldc,al,dbmi->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,dbim->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebji,dl->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kled,ebji,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebli,dj->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,klde,ebil,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kled,ebjl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,klde,bl,deji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,kled,bl,edji->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba +=  1.00 * einsum('jk,mlci,al,bm->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,kljd,bl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kldi,bl,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kade,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbabba +=  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,klde,bl,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_bbabba


def get_cc_j_doubles_singles_bbbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    cc_j_doubles_singles_bbbaba =  1.00 * einsum('ik,lc,abjl->abjick', kd_aa[oa, oa], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,kd,dbij->abjick', kd_bb[vb, vb], f_aa[oa, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaij->abjick', kd_bb[vb, vb], g_abab[oa, vb, oa, ob])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba +=  1.00 * einsum('ik,abcj->abjick', kd_aa[oa, oa], g_bbbb[vb, vb, vb, ob])
    contracted_intermediate =  1.00 * einsum('ac,klij,bl->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, ob], t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaid,dj->abjick', kd_bb[vb, vb], g_abab[oa, vb, oa, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba +=  1.00 * einsum('ik,abcd,dj->abjick', kd_aa[oa, oa], g_bbbb[vb, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba += -1.00 * einsum('klic,abjl->abjick', g_abab[oa, ob, oa, vb], t2_bbbb)
    cc_j_doubles_singles_bbbaba +=  0.50 * einsum('ik,mlcj,abml->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kldj,dbil->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdi,dblj->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klid,dbjl->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kadc,dbij->abjick', g_abab[oa, vb, va, vb], t2_abab)
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,ladc,dblj->abjick', kd_aa[oa, oa], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacd,dbjl->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,kade,deij->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,kaed,edij->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba += -1.00 * einsum('kldc,abjl,di->abjick', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kldc,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba +=  1.00 * einsum('ik,lmdc,abjm,dl->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00 * einsum('ik,mlcd,abjm,dl->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  0.50 * einsum('ik,mlcd,abml,dj->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ik,mldc,al,dbmj->abjick', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,mlcd,al,dbjm->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kled,ebij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kled,ebil,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,eblj,di->abjick', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,ebjl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,klde,bl,deij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,kled,bl,edij->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba += -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kldj,bl,di->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klid,bl,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacd,bl,dj->abjick', kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaed,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, vb, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    cc_j_doubles_singles_bbbaba += -1.00 * einsum('ik,mlcd,al,bm,dj->abjick', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,kled,bl,dj,ei->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (1, 3), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_bbbaba


def get_cc_j_doubles_singles_bbbbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('klcj,abil->abjick', g_abab[oa, ob, va, ob], t2_bbbb)
    cc_j_doubles_singles_bbbbaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kacd,dbij->abjick', g_abab[oa, vb, va, vb], t2_bbbb)
    cc_j_doubles_singles_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klcd,abil,dj->abjick', g_abab[oa, ob, va, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klcd,al,dbij->abjick', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_bbbbaa


def get_cc_j_doubles_singles_bbbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
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
    
    contracted_intermediate = -1.00 * einsum('jk,lc,abil->abjick', kd_bb[ob, ob], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kd,dbij->abjick', kd_bb[vb, vb], f_bb[ob, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaij->abjick', kd_bb[vb, vb], g_bbbb[ob, vb, ob, ob])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,abcj->abjick', kd_bb[ob, ob], g_bbbb[vb, vb, vb, ob])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkij,bl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, ob, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd_bb[vb, vb], g_bbbb[ob, vb, vb, ob], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,abcd,di->abjick', kd_bb[ob, ob], g_bbbb[vb, vb, vb, vb], t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcj,abil->abjick', g_bbbb[ob, ob, vb, ob], t2_bbbb)
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,mlcj,abml->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkdj,dbli->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, ob], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kacd,dbij->abjick', g_bbbb[ob, vb, vb, vb], t2_bbbb)
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,ladc,dbli->abjick', kd_bb[ob, ob], g_abab[oa, vb, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,dbil->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,kade,deij->abjick', kd_bb[vb, vb], g_bbbb[ob, vb, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,abil,dj->abjick', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,al,dbij->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmdc,abim,dl->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,mlcd,abml,di->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mldc,al,dbmi->abjick', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,dbim->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lked,ebli,dj->abjick', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebil,dj->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,lkde,bl,deij->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,mlcj,al,bm->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kade,dj,ei->abjick', kd_bb[vb, vb], g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (1, 3), (2, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles_bbbbbb
