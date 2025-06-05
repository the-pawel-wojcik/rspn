from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_lhe1e2cc_aaaaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibckj', g_aaaa[oa, oa, va, va], l1_aa)
    lhe1e2cc_aaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ic->aibckj', g_aaaa[oa, oa, va, va], l1_aa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,ja->aibckj', g_aaaa[oa, oa, va, va], l1_aa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->aibckj', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_aaaaaa +=  1.00 * einsum('jkal,ilbc->aibckj', g_aaaa[oa, oa, va, oa], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kicl,jlab->aibckj', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idab,jkdc->aibckj', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdab,kidc->aibckj', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_aaaaaa +=  1.00 * einsum('idbc,jkda->aibckj', g_aaaa[oa, va, va, va], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('liab,dl,jkcd->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljab,dl,kicd->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_aaaaaa += -1.00 * einsum('jkad,dl,libc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klad,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_aaaaaa += -1.00 * einsum('libc,dl,jkad->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('libd,dl,jkac->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilbd,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicd,dl,ljab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->aibckj', f_aa[oa, va], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ib,jkac->aibckj', f_aa[oa, va], l2_aaaa)
    lhe1e2cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_aaaaaa


def get_lhe1e2cc_aaabab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_aaabab =  1.00 * einsum('kiab,jc->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_aaabab += -1.00 * einsum('ijac,kb->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aaabab +=  1.00 * einsum('kjac,ib->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aaabab +=  1.00 * einsum('ijbc,ka->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('kial,ljbc->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijal,klbc->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_aaabab += -1.00 * einsum('kjal,ilbc->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aaabab += -1.00 * einsum('ijlc,klab->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('idab,kjdc->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('idac,kjbd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_aaabab +=  1.00 * einsum('djac,kidb->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_aaabab +=  1.00 * einsum('idbc,kjad->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liab,dl,kjdc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilac,dl,kjbd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijad,dl,klbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_aaabab +=  1.00 * einsum('ljac,dl,kibd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab += -1.00 * einsum('kjad,dl,ilbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab += -1.00 * einsum('lkad,dl,ijbc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab +=  1.00 * einsum('klad,dl,ijbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab += -1.00 * einsum('ilbc,dl,kjad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabab += -1.00 * einsum('libd,dl,kjac->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab +=  1.00 * einsum('ilbd,dl,kjac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabab +=  1.00 * einsum('ijdc,dl,lkab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabab +=  1.00 * einsum('ka,ijbc->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_aaabab +=  1.00 * einsum('ib,kjac->aibckj', f_aa[oa, va], l2_abab)
    return lhe1e2cc_aaabab


def get_lhe1e2cc_aaabba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_aaabba =  1.00 * einsum('ikac,jb->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aaabba += -1.00 * einsum('jiab,kc->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_aaabba += -1.00 * einsum('jkac,ib->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aaabba += -1.00 * einsum('ikbc,ja->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ikal,jlbc->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jial,lkbc->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_aaabba +=  1.00 * einsum('jkal,ilbc->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aaabba +=  1.00 * einsum('iklc,jlab->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_aaabba +=  1.00 * einsum('idab,jkdc->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aaabba +=  1.00 * einsum('idac,jkbd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aaabba += -1.00 * einsum('dkac,jidb->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_aaabba += -1.00 * einsum('jdab,ikdc->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aaabba += -1.00 * einsum('jdac,ikbd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aaabba += -1.00 * einsum('idbc,jkad->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aaabba +=  1.00 * einsum('liab,dl,jkdc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('ilac,dl,jkbd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('lkac,dl,jibd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ikad,dl,jlbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiad,dl,lkbc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_aaabba += -1.00 * einsum('ljab,dl,ikdc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba +=  1.00 * einsum('jlac,dl,ikbd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabba +=  1.00 * einsum('jkad,dl,ilbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba +=  1.00 * einsum('ljad,dl,ikbc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('jlad,dl,ikbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba +=  1.00 * einsum('ilbc,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabba +=  1.00 * einsum('libd,dl,jkac->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('ilbd,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('ikdc,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aaabba += -1.00 * einsum('ja,ikbc->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_aaabba += -1.00 * einsum('ib,jkac->aibckj', f_aa[oa, va], l2_abab)
    return lhe1e2cc_aaabba


def get_lhe1e2cc_aabaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_aabaab = -1.00 * einsum('kiac,jb->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_aabaab +=  1.00 * einsum('ijab,kc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aabaab += -1.00 * einsum('kjab,ic->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aabaab += -1.00 * einsum('ijcb,ka->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aabaab +=  1.00 * einsum('kial,ljcb->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aabaab += -1.00 * einsum('ijal,klcb->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aabaab +=  1.00 * einsum('ijlb,klac->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_aabaab +=  1.00 * einsum('kjal,ilcb->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aabaab += -1.00 * einsum('kicl,ljab->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aabaab +=  1.00 * einsum('ijcl,klab->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('idab,kjcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idac,kjdb->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_aabaab += -1.00 * einsum('djab,kidc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_aabaab += -1.00 * einsum('idcb,kjad->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ilab,dl,kjcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liac,dl,kjdb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_aabaab +=  1.00 * einsum('kiad,dl,ljcb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('ijad,dl,klcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('ijdb,dl,lkac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('ljab,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab +=  1.00 * einsum('kjad,dl,ilcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab +=  1.00 * einsum('lkad,dl,ijcb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('klad,dl,ijcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab +=  1.00 * einsum('ilcb,dl,kjad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabaab +=  1.00 * einsum('licd,dl,kjab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('ilcd,dl,kjab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('kicd,dl,ljab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabaab +=  1.00 * einsum('ijcd,dl,klab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabaab += -1.00 * einsum('ka,ijcb->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_aabaab += -1.00 * einsum('ic,kjab->aibckj', f_aa[oa, va], l2_abab)
    return lhe1e2cc_aabaab


def get_lhe1e2cc_aababa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_aababa = -1.00 * einsum('ikab,jc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aababa +=  1.00 * einsum('jiac,kb->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_aababa +=  1.00 * einsum('jkab,ic->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aababa +=  1.00 * einsum('ikcb,ja->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_aababa +=  1.00 * einsum('ikal,jlcb->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aababa += -1.00 * einsum('iklb,jlac->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_aababa += -1.00 * einsum('jial,lkcb->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aababa += -1.00 * einsum('jkal,ilcb->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aababa += -1.00 * einsum('ikcl,jlab->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('jicl,lkab->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_aababa += -1.00 * einsum('idab,jkcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aababa += -1.00 * einsum('idac,jkdb->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('dkab,jidc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_aababa +=  1.00 * einsum('jdab,ikcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('jdac,ikdb->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('idcb,jkad->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('ilab,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('liac,dl,jkdb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('lkab,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('ikad,dl,jlcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('ikdb,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('jiad,dl,lkcb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('jlab,dl,ikcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('ljac,dl,ikdb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('jkad,dl,ilcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('ljad,dl,ikcb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('jlad,dl,ikcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('ilcb,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('licd,dl,jkab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('ilcd,dl,jkab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa += -1.00 * einsum('ikcd,dl,jlab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('jicd,dl,lkab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aababa +=  1.00 * einsum('ja,ikcb->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_aababa +=  1.00 * einsum('ic,jkab->aibckj', f_aa[oa, va], l2_abab)
    return lhe1e2cc_aababa


def get_lhe1e2cc_aabbbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('ikab,jc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_aabbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikal,jlbc->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('iklb,ljac->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('iklc,ljab->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idab,jkdc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('dkab,ijdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('djab,ikdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilab,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkab,dl,ijdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikad,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikdb,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljab,dl,ikdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikdc,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_aabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return lhe1e2cc_aabbbb


def get_lhe1e2cc_ababaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kiac,jb->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_ababaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_ababaa +=  1.00 * einsum('jkab,ic->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('kibc,ja->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_ababaa += -1.00 * einsum('jkal,libc->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate = -1.00 * einsum('kilc,jlab->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_ababaa +=  1.00 * einsum('diac,jkdb->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_ababaa += -1.00 * einsum('kdab,jidc->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_ababaa += -1.00 * einsum('kdac,jibd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_ababaa +=  1.00 * einsum('jdab,kidc->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_ababaa +=  1.00 * einsum('jdac,kibd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_ababaa += -1.00 * einsum('dibc,jkda->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_ababaa +=  1.00 * einsum('liac,dl,jkbd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa += -1.00 * einsum('lkab,dl,jidc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa +=  1.00 * einsum('klac,dl,jibd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kiad,dl,jlbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_ababaa +=  1.00 * einsum('ljab,dl,kidc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa += -1.00 * einsum('jlac,dl,kibd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_ababaa += -1.00 * einsum('jkad,dl,libc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klad,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_ababaa += -1.00 * einsum('libc,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa +=  1.00 * einsum('lidc,dl,jkab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_ababaa += -1.00 * einsum('licd,dl,jkab->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kidc,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_ababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_ababaa +=  1.00 * einsum('ic,jkab->aibckj', f_bb[ob, vb], l2_aaaa)
    return lhe1e2cc_ababaa


def get_lhe1e2cc_abbaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_abbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa += -1.00 * einsum('jkac,ib->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('kicb,ja->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,jlcb->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kilb,jlac->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa +=  1.00 * einsum('jkal,licb->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('kicl,jlab->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa += -1.00 * einsum('diab,jkdc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_abbaaa +=  1.00 * einsum('kdab,jicd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_abbaaa +=  1.00 * einsum('kdac,jidb->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_abbaaa += -1.00 * einsum('jdab,kicd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_abbaaa += -1.00 * einsum('jdac,kidb->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    lhe1e2cc_abbaaa +=  1.00 * einsum('dicb,jkda->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_abbaaa += -1.00 * einsum('liab,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa += -1.00 * einsum('klab,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00 * einsum('lkac,dl,jidb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kiad,dl,jlcb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kidb,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa +=  1.00 * einsum('jlab,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbaaa += -1.00 * einsum('ljac,dl,kidb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00 * einsum('jkad,dl,licb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkad,dl,jicb->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,jicb->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa +=  1.00 * einsum('licb,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa += -1.00 * einsum('lidb,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00 * einsum('libd,dl,jkac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kicd,dl,jlab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,jicb->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_abbaaa += -1.00 * einsum('ib,jkac->aibckj', f_bb[ob, vb], l2_aaaa)
    return lhe1e2cc_abbaaa


def get_lhe1e2cc_abbbab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_abbbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjab,ic->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbab += -1.00 * einsum('jibc,ka->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_abbbab +=  1.00 * einsum('kial,jlbc->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_abbbab += -1.00 * einsum('kilb,ljac->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_abbbab +=  1.00 * einsum('jibl,klac->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_abbbab += -1.00 * einsum('kjal,ilbc->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_abbbab +=  1.00 * einsum('kilc,ljab->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_abbbab += -1.00 * einsum('jicl,klab->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('diab,kjdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kdab,jidc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('djab,kidc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbab +=  1.00 * einsum('idbc,kjad->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liab,dl,kjdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klab,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbab += -1.00 * einsum('kiad,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab += -1.00 * einsum('kidb,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbbab +=  1.00 * einsum('jibd,dl,klac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljab,dl,kidc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbab +=  1.00 * einsum('kjad,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00 * einsum('lkad,dl,jibc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab += -1.00 * einsum('klad,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00 * einsum('libc,dl,kjad->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lidb,dl,kjac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('libd,dl,kjac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbab +=  1.00 * einsum('kidc,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbbab += -1.00 * einsum('jicd,dl,klab->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbab += -1.00 * einsum('ka,jibc->aibckj', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ib,kjac->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_abbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_abbbab


def get_lhe1e2cc_abbbba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('jiab,kc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_abbbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ic->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbba +=  1.00 * einsum('kibc,ja->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_abbbba += -1.00 * einsum('kibl,jlac->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_abbbba += -1.00 * einsum('jial,klbc->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_abbbba +=  1.00 * einsum('jilb,lkac->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_abbbba +=  1.00 * einsum('jkal,ilbc->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_abbbba +=  1.00 * einsum('kicl,jlab->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_abbbba += -1.00 * einsum('jilc,lkab->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('diab,jkdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdab,kidc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbba += -1.00 * einsum('idbc,jkad->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('liab,dl,jkdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    lhe1e2cc_abbbba += -1.00 * einsum('kibd,dl,jlac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00 * einsum('jiad,dl,lkbc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00 * einsum('jidb,dl,lkac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlab,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbba += -1.00 * einsum('jkad,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba += -1.00 * einsum('ljad,dl,kibc->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00 * einsum('jlad,dl,kibc->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba += -1.00 * einsum('libc,dl,jkad->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lidb,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libd,dl,jkac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_abbbba +=  1.00 * einsum('kicd,dl,jlab->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_abbbba += -1.00 * einsum('jidc,dl,lkab->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_abbbba +=  1.00 * einsum('ja,kibc->aibckj', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ib,jkac->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_abbbba


def get_lhe1e2cc_baaaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('ijba,kc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_baaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjba,ic->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaab +=  1.00 * einsum('kibc,ja->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_baaaab += -1.00 * einsum('kibl,ljca->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_baaaab += -1.00 * einsum('ijla,klbc->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_baaaab +=  1.00 * einsum('ijbl,klca->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_baaaab +=  1.00 * einsum('kjla,ilbc->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_baaaab +=  1.00 * einsum('kicl,ljba->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_baaaab += -1.00 * einsum('ijcl,klba->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('idba,kjcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('djba,kidc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaab += -1.00 * einsum('idbc,kjda->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('ilba,dl,kjcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    lhe1e2cc_baaaab += -1.00 * einsum('kibd,dl,ljca->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaab +=  1.00 * einsum('ijda,dl,lkbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaab +=  1.00 * einsum('ijbd,dl,klca->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljba,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaab += -1.00 * einsum('kjda,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaab +=  1.00 * einsum('ljda,dl,kibc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab += -1.00 * einsum('ljad,dl,kibc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab += -1.00 * einsum('libc,dl,kjda->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('libd,dl,kjca->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilbd,dl,kjca->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaab +=  1.00 * einsum('kicd,dl,ljba->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaab += -1.00 * einsum('ijcd,dl,klba->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaab +=  1.00 * einsum('ja,kibc->aibckj', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('ib,kjca->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_baaaab


def get_lhe1e2cc_baaaba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('ikba,jc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_baaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkba,ic->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaba += -1.00 * einsum('jibc,ka->aibckj', g_aaaa[oa, oa, va, va], l1_bb)
    lhe1e2cc_baaaba +=  1.00 * einsum('ikla,jlbc->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_baaaba += -1.00 * einsum('ikbl,jlca->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_baaaba +=  1.00 * einsum('jibl,lkca->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe1e2cc_baaaba += -1.00 * einsum('jkla,ilbc->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_baaaba +=  1.00 * einsum('ikcl,jlba->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_baaaba += -1.00 * einsum('jicl,lkba->aibckj', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('idba,jkcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('dkba,jidc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdba,ikcd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaba +=  1.00 * einsum('idbc,jkda->aibckj', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('ilba,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkba,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaba += -1.00 * einsum('ikda,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaba += -1.00 * einsum('ikbd,dl,jlca->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00 * einsum('jibd,dl,lkca->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlba,dl,ikcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaba +=  1.00 * einsum('jkda,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaba += -1.00 * einsum('lkda,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00 * einsum('lkad,dl,jibc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00 * einsum('libc,dl,jkda->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('libd,dl,jkca->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilbd,dl,jkca->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_baaaba +=  1.00 * einsum('ikcd,dl,jlba->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baaaba += -1.00 * einsum('jicd,dl,lkba->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baaaba += -1.00 * einsum('ka,jibc->aibckj', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ib,jkca->aibckj', f_aa[oa, va], l2_abab)
    lhe1e2cc_baaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_baaaba


def get_lhe1e2cc_baabbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('ikba,jc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_baabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb += -1.00 * einsum('jkac,ib->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ikbc,ja->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikla,ljbc->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikbl,jlac->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb +=  1.00 * einsum('jkal,ilbc->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('iklc,ljba->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb += -1.00 * einsum('idba,jkdc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_baabbb +=  1.00 * einsum('dkba,ijdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_baabbb +=  1.00 * einsum('kdac,ijbd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_baabbb += -1.00 * einsum('djba,ikdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_baabbb += -1.00 * einsum('jdac,ikbd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_baabbb +=  1.00 * einsum('idbc,jkda->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_baabbb += -1.00 * einsum('ilba,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baabbb += -1.00 * einsum('lkba,dl,ijdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb +=  1.00 * einsum('lkac,dl,ijbd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ikda,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikbd,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb +=  1.00 * einsum('ljba,dl,ikdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb += -1.00 * einsum('ljac,dl,ikbd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baabbb +=  1.00 * einsum('jkad,dl,ilbc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkda,dl,ijbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb +=  1.00 * einsum('ilbc,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baabbb +=  1.00 * einsum('libd,dl,jkac->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_baabbb += -1.00 * einsum('ilbd,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ikdc,dl,ljba->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ijbc->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_baabbb += -1.00 * einsum('ib,jkac->aibckj', f_aa[oa, va], l2_bbbb)
    return lhe1e2cc_baabbb


def get_lhe1e2cc_bababb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('ikca,jb->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bababb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_bababb +=  1.00 * einsum('jkab,ic->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ikcb,ja->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikla,ljcb->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_bababb += -1.00 * einsum('jkal,ilcb->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('ikcl,jlab->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_bababb +=  1.00 * einsum('idca,jkdb->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bababb += -1.00 * einsum('kdab,ijcd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bababb += -1.00 * einsum('dkca,ijdb->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bababb +=  1.00 * einsum('jdab,ikcd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bababb +=  1.00 * einsum('djca,ikdb->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bababb += -1.00 * einsum('idcb,jkda->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bababb +=  1.00 * einsum('ilca,dl,jkbd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bababb += -1.00 * einsum('lkab,dl,ijcd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bababb +=  1.00 * einsum('lkca,dl,ijdb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ikda,dl,ljcb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_bababb +=  1.00 * einsum('ljab,dl,ikcd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bababb += -1.00 * einsum('ljca,dl,ikdb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bababb += -1.00 * einsum('jkad,dl,ilcb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkda,dl,ijcb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,ijcb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_bababb += -1.00 * einsum('ilcb,dl,jkad->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bababb += -1.00 * einsum('licd,dl,jkab->aibckj', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bababb +=  1.00 * einsum('ilcd,dl,jkab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ikcd,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,ijcb->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_bababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_bababb +=  1.00 * einsum('ic,jkab->aibckj', f_aa[oa, va], l2_bbbb)
    return lhe1e2cc_bababb


def get_lhe1e2cc_bbaaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kiba,jc->aibckj', g_abab[oa, ob, va, vb], l1_aa)
    lhe1e2cc_bbaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kila,jlbc->aibckj', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibl,jlca->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicl,jlba->aibckj', g_abab[oa, ob, va, ob], l2_abab)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('diba,jkdc->aibckj', g_abab[va, ob, va, vb], l2_aaaa)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kdba,jicd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdba,kicd->aibckj', g_abab[oa, vb, va, vb], l2_abab)
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liba,dl,jkcd->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klba,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kida,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibd,dl,jlca->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlba,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicd,dl,jlba->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    return lhe1e2cc_bbaaaa


def get_lhe1e2cc_bbabab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_bbabab = -1.00 * einsum('kiba,jc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabab +=  1.00 * einsum('jiac,kb->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_bbabab +=  1.00 * einsum('kjba,ic->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabab +=  1.00 * einsum('kibc,ja->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabab +=  1.00 * einsum('kila,ljbc->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbabab += -1.00 * einsum('kibl,jlac->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_bbabab += -1.00 * einsum('jial,klbc->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbabab += -1.00 * einsum('kjla,libc->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbabab += -1.00 * einsum('kilc,ljba->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('jicl,klba->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbabab += -1.00 * einsum('diba,kjdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbabab += -1.00 * einsum('idac,kjbd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('kdba,jidc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bbabab +=  1.00 * einsum('djba,kidc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('jdac,kibd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('dibc,kjda->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('liba,dl,kjdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('liac,dl,kjbd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('klba,dl,jicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('kida,dl,ljbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('kibd,dl,ljac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('jiad,dl,klbc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('ljba,dl,kidc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('ljac,dl,kibd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('kjda,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('ljda,dl,kibc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('ljad,dl,kibc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('libc,dl,kjda->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('lidc,dl,kjba->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('licd,dl,kjba->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab += -1.00 * einsum('kidc,dl,ljba->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('jicd,dl,klba->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabab +=  1.00 * einsum('ja,kibc->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_bbabab +=  1.00 * einsum('ic,kjba->aibckj', f_bb[ob, vb], l2_abab)
    return lhe1e2cc_bbabab


def get_lhe1e2cc_bbabba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_bbabba = -1.00 * einsum('kiac,jb->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_bbabba +=  1.00 * einsum('jiba,kc->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabba += -1.00 * einsum('jkba,ic->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabba += -1.00 * einsum('jibc,ka->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbabba +=  1.00 * einsum('kial,jlbc->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbabba += -1.00 * einsum('jila,lkbc->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbabba +=  1.00 * einsum('jibl,klac->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_bbabba +=  1.00 * einsum('jkla,libc->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbabba += -1.00 * einsum('kicl,jlba->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbabba +=  1.00 * einsum('jilc,lkba->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('diba,jkdc->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idac,jkbd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_bbabba += -1.00 * einsum('jdba,kidc->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bbabba += -1.00 * einsum('dibc,jkda->aibckj', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liba,dl,jkdc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liac,dl,jkbd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_bbabba +=  1.00 * einsum('kiad,dl,jlbc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('jida,dl,lkbc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('jibd,dl,lkac->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('jlba,dl,kicd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabba +=  1.00 * einsum('jkda,dl,libc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('lkda,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba +=  1.00 * einsum('lkad,dl,jibc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba +=  1.00 * einsum('libc,dl,jkda->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('lidc,dl,jkba->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba +=  1.00 * einsum('licd,dl,jkba->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('kicd,dl,jlba->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbabba +=  1.00 * einsum('jidc,dl,lkba->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbabba += -1.00 * einsum('ka,jibc->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_bbabba += -1.00 * einsum('ic,jkba->aibckj', f_bb[ob, vb], l2_abab)
    return lhe1e2cc_bbabba


def get_lhe1e2cc_bbbaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_bbbaab =  1.00 * einsum('kica,jb->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbbaab += -1.00 * einsum('jiab,kc->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_bbbaab += -1.00 * einsum('kjca,ib->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbbaab += -1.00 * einsum('kicb,ja->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kila,ljcb->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jial,klcb->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_bbbaab +=  1.00 * einsum('kjla,licb->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbbaab +=  1.00 * einsum('kicl,jlab->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe1e2cc_bbbaab +=  1.00 * einsum('idab,kjcd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbbaab +=  1.00 * einsum('dica,kjdb->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbbaab += -1.00 * einsum('kdca,jidb->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bbbaab += -1.00 * einsum('jdab,kicd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbbaab += -1.00 * einsum('djca,kidb->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbbaab += -1.00 * einsum('dicb,kjda->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbbaab +=  1.00 * einsum('liab,dl,kjcd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('lica,dl,kjdb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('klca,dl,jibd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('kida,dl,ljcb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiad,dl,klcb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_bbbaab += -1.00 * einsum('ljab,dl,kicd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00 * einsum('ljca,dl,kidb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00 * einsum('kjda,dl,licb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('ljda,dl,kicb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00 * einsum('ljad,dl,kicb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00 * einsum('licb,dl,kjda->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('lidb,dl,kjca->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab +=  1.00 * einsum('libd,dl,kjca->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('kicd,dl,ljab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaab += -1.00 * einsum('ja,kicb->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_bbbaab += -1.00 * einsum('ib,kjca->aibckj', f_bb[ob, vb], l2_abab)
    return lhe1e2cc_bbbaab


def get_lhe1e2cc_bbbaba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe1e2cc_bbbaba =  1.00 * einsum('kiab,jc->aibckj', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe1e2cc_bbbaba += -1.00 * einsum('jica,kb->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbbaba +=  1.00 * einsum('jkca,ib->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    lhe1e2cc_bbbaba +=  1.00 * einsum('jicb,ka->aibckj', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kial,jlcb->aibckj', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jila,lkcb->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_bbbaba += -1.00 * einsum('jkla,licb->aibckj', g_abab[oa, ob, oa, vb], l2_abab)
    lhe1e2cc_bbbaba += -1.00 * einsum('jicl,klab->aibckj', g_abab[oa, ob, va, ob], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('idab,jkcd->aibckj', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('dica,jkdb->aibckj', g_abab[va, ob, va, vb], l2_abab)
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    lhe1e2cc_bbbaba +=  1.00 * einsum('jdca,kidb->aibckj', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe1e2cc_bbbaba +=  1.00 * einsum('dicb,jkda->aibckj', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liab,dl,jkcd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lica,dl,jkdb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,jlcb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jida,dl,lkcb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->biackj', contracted_intermediate) 
    lhe1e2cc_bbbaba +=  1.00 * einsum('jlca,dl,kibd->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaba += -1.00 * einsum('jkda,dl,licb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00 * einsum('lkda,dl,jicb->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba += -1.00 * einsum('lkad,dl,jicb->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba += -1.00 * einsum('licb,dl,jkda->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00 * einsum('lidb,dl,jkca->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba += -1.00 * einsum('libd,dl,jkca->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00 * einsum('jicd,dl,lkab->aibckj', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbaba +=  1.00 * einsum('ka,jicb->aibckj', f_bb[ob, vb], l2_abab)
    lhe1e2cc_bbbaba +=  1.00 * einsum('ib,jkca->aibckj', f_bb[ob, vb], l2_abab)
    return lhe1e2cc_bbbaba


def get_lhe1e2cc_bbbbbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('kiab,jc->aibckj', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe1e2cc_bbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->aicbjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ic->aibckj', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,ja->aibckj', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->aibckj', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    lhe1e2cc_bbbbbb +=  1.00 * einsum('jkal,ilbc->aibckj', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kicl,jlab->aibckj', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('idab,jkdc->aibckj', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdab,kidc->aibckj', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_bbbbbb +=  1.00 * einsum('idbc,jkda->aibckj', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('liab,dl,jkcd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->akbcij', contracted_intermediate)  + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate)  +  1.00000 * einsum('aibckj->akcbij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate)  + -1.00000 * einsum('aibckj->biackj', contracted_intermediate)  +  1.00000 * einsum('aibckj->biacjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljab,dl,kicd->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    lhe1e2cc_bbbbbb += -1.00 * einsum('jkad,dl,libc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkda,dl,jibc->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    lhe1e2cc_bbbbbb += -1.00 * einsum('libc,dl,jkad->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lidb,dl,jkac->aibckj', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libd,dl,jkac->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kicd,dl,ljab->aibckj', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->aibckj', f_bb[ob, vb], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aibcjk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ib,jkac->aibckj', f_bb[ob, vb], l2_bbbb)
    lhe1e2cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibckj->aicbkj', contracted_intermediate) 
    return lhe1e2cc_bbbbbb
