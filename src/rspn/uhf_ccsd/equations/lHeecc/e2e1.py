from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_lhe2e1cc_aaaaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e1cc_aaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjick', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjick', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_aaaaaa +=  1.00 * einsum('kicl,jlab->abjick', g_aaaa[oa, oa, va, oa], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjick', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjick', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_aaaaaa +=  1.00 * einsum('kdbc,ijda->abjick', g_aaaa[oa, va, va, va], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aaaaaa += -1.00 * einsum('lkbc,dl,ijad->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcd,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_aaaaaa += -1.00 * einsum('kicd,dl,ljab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e1cc_aaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_aaaaaa


def get_lhe2e1cc_aaaabb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('jkac,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_aaaabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkal,ilbc->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_aaaabb += -1.00 * einsum('jklc,ilab->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_aaaabb +=  1.00 * einsum('iklc,jlab->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_aaaabb +=  1.00 * einsum('dkac,ijdb->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jdac,ikbd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_aaaabb += -1.00 * einsum('dkbc,ijda->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_aaaabb +=  1.00 * einsum('lkac,dl,ijbd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlac,dl,ikbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkad,dl,ilbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_aaaabb += -1.00 * einsum('lkbc,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaaabb +=  1.00 * einsum('jkdc,dl,liab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaaabb += -1.00 * einsum('ikdc,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    return lhe2e1cc_aaaabb


def get_lhe2e1cc_aaabba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_aaabba = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,libc->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aaabba +=  1.00 * einsum('jilc,klab->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_aaabba += -1.00 * einsum('kilc,jlab->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_aaabba += -1.00 * einsum('kdab,jidc->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_aaabba += -1.00 * einsum('kdac,jibd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('diac,kjdb->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aaabba +=  1.00 * einsum('kdbc,jiad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_aaabba += -1.00 * einsum('lkab,dl,jidc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00 * einsum('klac,dl,jibd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jlac,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kiad,dl,jlbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klad,dl,jibc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aaabba += -1.00 * einsum('klbc,dl,jiad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaabba += -1.00 * einsum('jidc,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aaabba += -1.00 * einsum('lidc,dl,kjab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00 * einsum('licd,dl,kjab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aaabba +=  1.00 * einsum('kidc,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ka,jibc->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_aaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aaabba += -1.00 * einsum('ic,kjab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e1cc_aaabba


def get_lhe2e1cc_aababa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_aababa =  1.00 * einsum('kiab,jc->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,ljbc->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijlc,klab->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_aababa +=  1.00 * einsum('kdab,ijdc->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_aababa +=  1.00 * einsum('kdac,ijbd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('idac,kjbd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aababa += -1.00 * einsum('kdbc,ijad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_aababa +=  1.00 * einsum('lkab,dl,ijdc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa += -1.00 * einsum('klac,dl,ijbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilac,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjad,dl,ilbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aababa +=  1.00 * einsum('klbc,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ijdc,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_aababa +=  1.00 * einsum('ljdc,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_aababa += -1.00 * einsum('ljcd,dl,kiab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_aababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_aababa +=  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e1cc_aababa


def get_lhe2e1cc_abaaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_abaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abaaab += -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_abaaab += -1.00 * einsum('jkcb,ia->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_abaaab +=  1.00 * einsum('ikcb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkal,ilcb->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jklb,ilac->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abaaab += -1.00 * einsum('ijcl,lkab->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_abaaab +=  1.00 * einsum('jkcl,ilab->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_abaaab += -1.00 * einsum('ikcl,jlab->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_abaaab += -1.00 * einsum('dkab,ijdc->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jdac,ikdb->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdcb,ikad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abaaab +=  1.00 * einsum('dkcb,ijda->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_abaaab += -1.00 * einsum('lkab,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,ikdb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcb,dl,ikad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkad,dl,ilcb->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkdb,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abaaab += -1.00 * einsum('lkdb,dl,ijac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00 * einsum('lkbd,dl,ijac->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00 * einsum('lkcb,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab += -1.00 * einsum('ijcd,dl,lkab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abaaab +=  1.00 * einsum('jkcd,dl,ilab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljcd,dl,ikab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlcd,dl,ikab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abaaab += -1.00 * einsum('ikcd,dl,jlab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abaaab += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jc,ikab->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_abaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_abaaab


def get_lhe2e1cc_ababaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_ababaa =  1.00 * einsum('kiab,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_ababaa +=  1.00 * einsum('jicb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_ababaa +=  1.00 * einsum('kjac,ib->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_ababaa += -1.00 * einsum('kicb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_ababaa += -1.00 * einsum('kjal,licb->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_ababaa += -1.00 * einsum('kial,jlcb->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('kilb,jlac->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_ababaa += -1.00 * einsum('jicl,klab->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('kjcl,liab->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('kicl,jlab->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('kdab,jicd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('kdac,jidb->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_ababaa += -1.00 * einsum('jdac,kidb->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('jdcb,kiad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_ababaa += -1.00 * einsum('dicb,kjda->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_ababaa += -1.00 * einsum('kdcb,jiad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_ababaa += -1.00 * einsum('klab,dl,jicd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('lkac,dl,jidb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('ljac,dl,kidb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('jlcb,dl,kiad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('licb,dl,kjad->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('kjad,dl,licb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('kiad,dl,jlcb->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('kidb,dl,ljac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('lkad,dl,jicb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('klad,dl,jicb->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('klcb,dl,jiad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('jicd,dl,klab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('kjcd,dl,liab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababaa += -1.00 * einsum('ljcd,dl,kiab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('jlcd,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('kicd,dl,jlab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababaa +=  1.00 * einsum('ka,jicb->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_ababaa +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e1cc_ababaa


def get_lhe2e1cc_ababbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_ababbb =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_ababbb +=  1.00 * einsum('kibc,ja->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_ababbb +=  1.00 * einsum('jkal,ilbc->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_ababbb += -1.00 * einsum('jklb,liac->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_ababbb += -1.00 * einsum('kibl,jlac->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('jilc,lkab->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_ababbb +=  1.00 * einsum('kicl,jlab->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('dkab,jidc->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_ababbb += -1.00 * einsum('jdac,kidb->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_ababbb +=  1.00 * einsum('diac,jkdb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_ababbb += -1.00 * einsum('idbc,jkad->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_ababbb +=  1.00 * einsum('kdbc,jiad->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('lkab,dl,jidc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_ababbb += -1.00 * einsum('jlac,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('liac,dl,jkdb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('libc,dl,jkad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('jkad,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('jkdb,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('kibd,dl,jlac->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb +=  1.00 * einsum('lkdb,dl,jiac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('lkbd,dl,jiac->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb +=  1.00 * einsum('lkbc,dl,jiad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jidc,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_ababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_ababbb +=  1.00 * einsum('lidc,dl,jkab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb += -1.00 * einsum('licd,dl,jkab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb +=  1.00 * einsum('kicd,dl,jlab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_ababbb +=  1.00 * einsum('kb,jiac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_ababbb +=  1.00 * einsum('ic,jkab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e1cc_ababbb


def get_lhe2e1cc_abbaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_abbaaa = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ijcb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_abbaaa += -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_abbaaa +=  1.00 * einsum('kjal,ilcb->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_abbaaa += -1.00 * einsum('kjlb,ilac->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_abbaaa +=  1.00 * einsum('kial,ljcb->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_abbaaa += -1.00 * einsum('kicl,ljab->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_abbaaa += -1.00 * einsum('kdab,ijcd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_abbaaa += -1.00 * einsum('kdac,ijdb->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_abbaaa +=  1.00 * einsum('djcb,kida->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_abbaaa +=  1.00 * einsum('idac,kjdb->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_abbaaa += -1.00 * einsum('idcb,kjad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_abbaaa +=  1.00 * einsum('kdcb,ijad->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_abbaaa +=  1.00 * einsum('klab,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('lkac,dl,ijdb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('ljcb,dl,kiad->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('liac,dl,kjdb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('ilcb,dl,kjad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('kjad,dl,ilcb->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('kjdb,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('kiad,dl,ljcb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00 * einsum('lkad,dl,ijcb->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('klad,dl,ijcb->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('klcb,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ijcd,dl,klab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_abbaaa +=  1.00 * einsum('licd,dl,kjab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('ilcd,dl,kjab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('kicd,dl,ljab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbaaa += -1.00 * einsum('ka,ijcb->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_abbaaa += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e1cc_abbaaa


def get_lhe2e1cc_abbabb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_abbabb = -1.00 * einsum('ikab,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_abbabb += -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_abbabb += -1.00 * einsum('kjbc,ia->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_abbabb +=  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_abbabb +=  1.00 * einsum('kjbl,ilac->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_abbabb += -1.00 * einsum('ikal,jlbc->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_abbabb +=  1.00 * einsum('iklb,ljac->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_abbabb +=  1.00 * einsum('ijlc,lkab->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_abbabb += -1.00 * einsum('kjcl,ilab->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_abbabb += -1.00 * einsum('iklc,ljab->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('dkab,ijdc->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_abbabb += -1.00 * einsum('djac,ikdb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_abbabb +=  1.00 * einsum('jdbc,ikad->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_abbabb +=  1.00 * einsum('idac,kjdb->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_abbabb += -1.00 * einsum('kdbc,ijad->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('lkab,dl,ijdc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_abbabb +=  1.00 * einsum('ljac,dl,ikdb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ljbc,dl,ikad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ilac,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('kjbd,dl,ilac->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ikad,dl,ljbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ikdb,dl,ljac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('lkdb,dl,ijac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('lkbd,dl,ijac->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('lkbc,dl,ijad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ijdc,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('kjcd,dl,ilab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('ljdc,dl,ikab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb +=  1.00 * einsum('ljcd,dl,ikab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('ikdc,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbabb += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_abbabb += -1.00 * einsum('jc,ikab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e1cc_abbabb


def get_lhe2e1cc_abbbba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_abbbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abbbba +=  1.00 * einsum('ijbc,ka->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_abbbba +=  1.00 * einsum('kjac,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_abbbba += -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjlb,liac->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abbbba +=  1.00 * einsum('ijcl,klab->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_abbbba += -1.00 * einsum('kjlc,liab->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_abbbba +=  1.00 * einsum('kilc,ljab->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,kiad->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klab,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljac,dl,kidb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,dl,kiad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjdb,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abbbba += -1.00 * einsum('lkad,dl,ijbc->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00 * einsum('klad,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00 * einsum('ijcd,dl,klab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba += -1.00 * einsum('kjdc,dl,liab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljdc,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_abbbba +=  1.00 * einsum('kidc,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_abbbba +=  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_abbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_abbbba


def get_lhe2e1cc_baaaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_baaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_baaaab +=  1.00 * einsum('ijbc,ka->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_baaaab +=  1.00 * einsum('jkca,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_baaaab += -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkla,ilbc->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkbl,ilca->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_baaaab +=  1.00 * einsum('ijcl,lkba->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_baaaab += -1.00 * einsum('jkcl,ilba->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_baaaab +=  1.00 * einsum('ikcl,jlba->abjick', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('dkba,ijdc->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jdca,ikbd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,ikda->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkba,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlca,dl,ikbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,dl,ikda->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkda,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkbd,dl,ilca->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_baaaab +=  1.00 * einsum('lkda,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab += -1.00 * einsum('lkad,dl,ijbc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00 * einsum('ijcd,dl,lkba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baaaab += -1.00 * einsum('jkcd,dl,ilba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ljcd,dl,ikba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcd,dl,ikba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_baaaab +=  1.00 * einsum('ikcd,dl,jlba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baaaab +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jc,ikba->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_baaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_baaaab


def get_lhe2e1cc_baabaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_baabaa = -1.00 * einsum('kiba,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_baabaa += -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_baabaa += -1.00 * einsum('kjbc,ia->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_baabaa +=  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_baabaa +=  1.00 * einsum('kjbl,lica->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_baabaa += -1.00 * einsum('kila,jlbc->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_baabaa +=  1.00 * einsum('kibl,jlca->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_baabaa +=  1.00 * einsum('jicl,klba->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_baabaa += -1.00 * einsum('kjcl,liba->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e1cc_baabaa += -1.00 * einsum('kicl,jlba->abjick', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('kdba,jicd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_baabaa += -1.00 * einsum('jdca,kibd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_baabaa +=  1.00 * einsum('jdbc,kida->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_baabaa +=  1.00 * einsum('dica,kjdb->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_baabaa += -1.00 * einsum('kdbc,jida->abjick', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('klba,dl,jicd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_baabaa +=  1.00 * einsum('jlca,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('ljbc,dl,kida->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('lica,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('kjbd,dl,lica->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('kida,dl,ljbc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('kibd,dl,jlca->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('lkbd,dl,jica->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('klbd,dl,jica->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('lkbc,dl,jida->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('jicd,dl,klba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('kjcd,dl,liba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabaa +=  1.00 * einsum('ljcd,dl,kiba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('jlcd,dl,kiba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('kicd,dl,jlba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabaa += -1.00 * einsum('kb,jica->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_baabaa += -1.00 * einsum('jc,kiba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e1cc_baabaa


def get_lhe2e1cc_baabbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_baabbb = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('jibc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_baabbb += -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_baabbb +=  1.00 * einsum('jkla,libc->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_baabbb += -1.00 * einsum('jkbl,ilac->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_baabbb +=  1.00 * einsum('kial,jlbc->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('jilc,lkba->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_baabbb += -1.00 * einsum('kicl,jlba->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_baabbb += -1.00 * einsum('dkba,jidc->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_baabbb += -1.00 * einsum('kdac,jibd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_baabbb +=  1.00 * einsum('jdbc,kida->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_baabbb +=  1.00 * einsum('idac,jkbd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_baabbb += -1.00 * einsum('dibc,jkda->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_baabbb +=  1.00 * einsum('dkbc,jida->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_baabbb +=  1.00 * einsum('lkba,dl,jidc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb += -1.00 * einsum('lkac,dl,jibd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('jlbc,dl,kiad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('liac,dl,jkbd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('libc,dl,jkda->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('jkda,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('jkbd,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('kiad,dl,jlbc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb += -1.00 * einsum('lkda,dl,jibc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('lkad,dl,jibc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb += -1.00 * einsum('lkbc,dl,jida->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jidc,dl,lkba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_baabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_baabbb += -1.00 * einsum('lidc,dl,jkba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb +=  1.00 * einsum('licd,dl,jkba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb += -1.00 * einsum('kicd,dl,jlba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_baabbb += -1.00 * einsum('ka,jibc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_baabbb += -1.00 * einsum('ic,jkba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e1cc_baabbb


def get_lhe2e1cc_babaaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_babaaa =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e1cc_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_babaaa +=  1.00 * einsum('kibc,ja->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e1cc_babaaa +=  1.00 * einsum('kjla,ilbc->abjick', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e1cc_babaaa += -1.00 * einsum('kjbl,ilca->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_babaaa += -1.00 * einsum('kibl,ljca->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijcl,klba->abjick', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e1cc_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_babaaa +=  1.00 * einsum('kicl,ljba->abjick', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('kdba,ijcd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_babaaa += -1.00 * einsum('djca,kidb->abjick', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e1cc_babaaa +=  1.00 * einsum('idca,kjbd->abjick', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e1cc_babaaa += -1.00 * einsum('idbc,kjda->abjick', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e1cc_babaaa +=  1.00 * einsum('kdbc,ijda->abjick', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('klba,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    lhe2e1cc_babaaa += -1.00 * einsum('ljca,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('ilca,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('libc,dl,kjda->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('kjda,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('kjbd,dl,ilca->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('kibd,dl,ljca->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babaaa += -1.00 * einsum('lkbd,dl,ijca->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa +=  1.00 * einsum('klbd,dl,ijca->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa +=  1.00 * einsum('lkbc,dl,ijda->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,klba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_babaaa += -1.00 * einsum('licd,dl,kjba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa +=  1.00 * einsum('ilcd,dl,kjba->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babaaa +=  1.00 * einsum('kicd,dl,ljba->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babaaa +=  1.00 * einsum('kb,ijca->abjick', f_aa[oa, va], l2_abab)
    lhe2e1cc_babaaa +=  1.00 * einsum('ic,kjba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e1cc_babaaa


def get_lhe2e1cc_bababb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_bababb =  1.00 * einsum('ikba,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bababb +=  1.00 * einsum('ijbc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bababb +=  1.00 * einsum('kjac,ib->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_bababb += -1.00 * einsum('ikbc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bababb += -1.00 * einsum('kjal,ilbc->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_bababb += -1.00 * einsum('ikla,ljbc->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('ikbl,jlac->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bababb += -1.00 * einsum('ijlc,lkba->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('kjcl,ilba->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('iklc,ljba->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('dkba,ijdc->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('kdac,ijbd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_bababb += -1.00 * einsum('jdac,ikbd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('djbc,ikda->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bababb += -1.00 * einsum('idbc,kjda->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_bababb += -1.00 * einsum('dkbc,ijda->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bababb += -1.00 * einsum('lkba,dl,ijdc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('lkac,dl,ijbd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ljac,dl,ikbd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ljbc,dl,ikda->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ilbc,dl,kjad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('kjad,dl,ilbc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ikda,dl,ljbc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ikbd,dl,ljac->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('lkda,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('lkad,dl,ijbc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('lkbc,dl,ijda->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ijdc,dl,lkba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('kjcd,dl,ilba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('ljdc,dl,ikba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb += -1.00 * einsum('ljcd,dl,ikba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('ikdc,dl,ljba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bababb +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_bababb +=  1.00 * einsum('jc,ikba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e1cc_bababb


def get_lhe2e1cc_babbba(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_babbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_babbba += -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e1cc_babbba += -1.00 * einsum('kjbc,ia->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_babbba +=  1.00 * einsum('kibc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjla,libc->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbl,ilac->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_babbba += -1.00 * einsum('ijcl,klba->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_babbba +=  1.00 * einsum('kjlc,liba->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_babbba += -1.00 * einsum('kilc,ljba->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_babbba += -1.00 * einsum('kdba,ijdc->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('djbc,kida->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_babbba +=  1.00 * einsum('kdbc,ijda->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_babbba += -1.00 * einsum('klba,dl,ijcd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljbc,dl,kida->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjda,dl,libc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjbd,dl,liac->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_babbba +=  1.00 * einsum('lkbd,dl,ijac->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba += -1.00 * einsum('klbd,dl,ijac->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00 * einsum('klbc,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babbba += -1.00 * einsum('ijcd,dl,klba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00 * einsum('kjdc,dl,liba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ljdc,dl,kiba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljcd,dl,kiba->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_babbba += -1.00 * einsum('kidc,dl,ljba->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_babbba += -1.00 * einsum('kb,ijac->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jc,kiba->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_babbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_babbba


def get_lhe2e1cc_bbabab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_bbabab =  1.00 * einsum('kiab,jc->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkla,licb->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,jlcb->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jicl,klab->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_bbabab +=  1.00 * einsum('kdab,jicd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_bbabab +=  1.00 * einsum('dkca,jidb->abjick', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdca,kidb->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('dica,jkdb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbabab += -1.00 * einsum('dkcb,jida->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bbabab +=  1.00 * einsum('lkab,dl,jicd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbabab += -1.00 * einsum('lkca,dl,jidb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlca,dl,kibd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lica,dl,jkdb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkda,dl,licb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,jlcb->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,dl,jicb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,jicb->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbabab +=  1.00 * einsum('lkcb,dl,jida->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jicd,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_bbabab += -1.00 * einsum('ljcd,dl,kiab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbabab +=  1.00 * einsum('jlcd,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,jicb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_bbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbabab +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e1cc_bbabab


def get_lhe2e1cc_bbbaab(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    lhe2e1cc_bbbaab = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,ilcb->abjick', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikla,ljcb->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbbaab +=  1.00 * einsum('ijcl,klab->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bbbaab += -1.00 * einsum('ikcl,jlab->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bbbaab += -1.00 * einsum('kdab,ijcd->abjick', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e1cc_bbbaab += -1.00 * einsum('dkca,ijdb->abjick', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('djca,ikdb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('idca,kjdb->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbbaab +=  1.00 * einsum('dkcb,ijda->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bbbaab += -1.00 * einsum('lkab,dl,ijcd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00 * einsum('lkca,dl,ijdb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ljca,dl,ikdb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilca,dl,kjbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,ilcb->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikda,dl,ljcb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkda,dl,ijcb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,ijcb->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbbaab += -1.00 * einsum('lkcb,dl,ijda->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab += -1.00 * einsum('ijcd,dl,lkab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00 * einsum('licd,dl,kjab->abjick', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab += -1.00 * einsum('ilcd,dl,kjab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbaab +=  1.00 * einsum('ikcd,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ka,ijcb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e1cc_bbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbbaab += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e1cc_bbbaab


def get_lhe2e1cc_bbbbaa(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kjca,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bbbbaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e1cc_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjla,licb->abjick', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e1cc_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_bbbbaa += -1.00 * einsum('kjcl,ilab->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bbbbaa +=  1.00 * einsum('kicl,jlab->abjick', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e1cc_bbbbaa +=  1.00 * einsum('kdca,ijdb->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('djca,kidb->abjick', g_abab[va, ob, va, vb], l2_abab)
    lhe2e1cc_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_bbbbaa += -1.00 * einsum('kdcb,ijda->abjick', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e1cc_bbbbaa +=  1.00 * einsum('klca,dl,ijbd->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljca,dl,kidb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjda,dl,licb->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_bbbbaa += -1.00 * einsum('klcb,dl,ijad->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbbaa +=  1.00 * einsum('kjcd,dl,liab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbaa += -1.00 * einsum('kicd,dl,ljab->abjick', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    return lhe2e1cc_bbbbaa


def get_lhe2e1cc_bbbbbb(
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
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    
    contracted_intermediate = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e1cc_bbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjick', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjick', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e1cc_bbbbbb +=  1.00 * einsum('kicl,jlab->abjick', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjick', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjick', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    lhe2e1cc_bbbbbb +=  1.00 * einsum('kdbc,ijda->abjick', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->acjibk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,dl,ijbc->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e1cc_bbbbbb += -1.00 * einsum('lkbc,dl,ijad->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljdc,dl,kiab->abjick', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e1cc_bbbbbb += -1.00 * einsum('kicd,dl,ljab->abjick', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e1cc_bbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e1cc_bbbbbb
