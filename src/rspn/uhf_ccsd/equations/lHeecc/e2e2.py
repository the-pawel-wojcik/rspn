from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_lhe2e2cc_aaaaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa +=  1.00 * einsum('kicl,jlab->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa +=  1.00 * einsum('kdbc,ijda->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa += -1.00 * einsum('lkbc,dl,ijad->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa += -1.00 * einsum('kicd,dl,ljab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_aaaaaaaa


def get_lhe2e2cc_aaaaaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaaaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    lhe2e2cc_aaaaaaba +=  1.00 * einsum('kdbc,ijda->abjicdk', g_aaaa[oa, va, va, va], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_aaaaaaba


def get_lhe2e2cc_aaaaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaaabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_aaaaabaa +=  1.00 * einsum('kicl,jlab->abjiclk', g_aaaa[oa, oa, va, oa], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_aaaaabaa


def get_lhe2e2cc_aaaaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaaabba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_aa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcd,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_aaaa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_aaaaabba


def get_lhe2e2cc_aaaabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaabaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaaabaab += -1.00 * einsum('jklc,ilab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00 * einsum('iklc,jlab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00 * einsum('dkac,ijdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaabaab += -1.00 * einsum('dkbc,ijda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00 * einsum('lkac,dl,ijbd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaabaab += -1.00 * einsum('lkbc,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaabaab +=  1.00 * einsum('jkdc,dl,liab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaabaab += -1.00 * einsum('ikdc,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    return lhe2e2cc_aaaabaab


def get_lhe2e2cc_aaaababb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaababb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aaaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    lhe2e2cc_aaaababb +=  1.00 * einsum('dkac,ijdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaababb += -1.00 * einsum('dkbc,ijda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aaaababb


def get_lhe2e2cc_aaaabbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaabbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaaabbab += -1.00 * einsum('jklc,ilab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaaabbab +=  1.00 * einsum('iklc,jlab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jdac,ikbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aaaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    return lhe2e2cc_aaaabbab


def get_lhe2e2cc_aaaabbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_aaaabbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,ikbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlac,dl,ikbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkad,dl,ilbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return lhe2e2cc_aaaabbbb


def get_lhe2e2cc_aaabbaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aaabbaaa = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,libc->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('jilc,klab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('kilc,jlab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('kdab,jidc->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('diac,kjdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aaabbaaa += -1.00 * einsum('lkab,dl,jidc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('liac,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,jibc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_aaabbaaa += -1.00 * einsum('jidc,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbaaa += -1.00 * einsum('lidc,dl,kjab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('kidc,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ka,jibc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaabbaaa += -1.00 * einsum('ic,kjab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aaabbaaa


def get_lhe2e2cc_aaabbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aaabbaba = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aaabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aaabbaba += -1.00 * einsum('kdab,jidc->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('diac,kjdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,jibc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aaabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaabbaba += -1.00 * einsum('ic,kjab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aaabbaba


def get_lhe2e2cc_aaabbbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aaabbbaa = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,libc->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_aaabbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aaabbbaa +=  1.00 * einsum('jilc,klab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaabbbaa += -1.00 * einsum('kilc,jlab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aaabbbaa += -1.00 * einsum('kdac,jibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aaabbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aaabbbaa +=  1.00 * einsum('kdbc,jiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ka,jibc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aaabbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaabbbaa += -1.00 * einsum('ic,kjab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aaabbbaa


def get_lhe2e2cc_aaabbbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aaabbbba = -1.00 * einsum('kjab,ic->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kial,jlbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aaabbbba += -1.00 * einsum('kdac,jibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aaabbbba +=  1.00 * einsum('kdbc,jiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00 * einsum('klac,dl,jibd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('jlac,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kiad,dl,jlbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klad,dl,jibc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_aaabbbba += -1.00 * einsum('klbc,dl,jiad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aaabbbba +=  1.00 * einsum('licd,dl,kjab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ka,jibc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aaabbbba += -1.00 * einsum('ic,kjab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aaabbbba


def get_lhe2e2cc_aababaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aababaaa =  1.00 * einsum('kiab,jc->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,ljbc->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijlc,klab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_aababaaa +=  1.00 * einsum('kdab,ijdc->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aababaaa +=  1.00 * einsum('lkab,dl,ijdc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,ljbc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijdc,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_aababaaa +=  1.00 * einsum('ljdc,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aababaaa +=  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aababaaa


def get_lhe2e2cc_aabababa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aabababa =  1.00 * einsum('kiab,jc->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aabababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aabababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aabababa +=  1.00 * einsum('kdab,ijdc->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aabababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aabababa +=  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aabababa


def get_lhe2e2cc_aababbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aababbaa =  1.00 * einsum('kiab,jc->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aababbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,ljbc->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_aababbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijlc,klab->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_aababbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_aababbaa +=  1.00 * einsum('kdac,ijbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('idac,kjbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aababbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aababbaa += -1.00 * einsum('kdbc,ijad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aababbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aababbaa +=  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aababbaa


def get_lhe2e2cc_aababbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_aababbba =  1.00 * einsum('kiab,jc->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_aababbba +=  1.00 * einsum('kdac,ijbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('idac,kjbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_aababbba += -1.00 * einsum('kdbc,ijad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_aababbba += -1.00 * einsum('klac,dl,ijbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ilac,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjad,dl,ilbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_aababbba +=  1.00 * einsum('klbc,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_aababbba += -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_aababbba +=  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_aaaa)
    return lhe2e2cc_aababbba


def get_lhe2e2cc_abaaaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abaaaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abaaaaab += -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abaaaaab += -1.00 * einsum('jkcb,ia->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abaaaaab +=  1.00 * einsum('ikcb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('jklb,ilac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abaaaaab += -1.00 * einsum('ijcl,lkab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abaaaaab += -1.00 * einsum('dkab,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jdac,ikdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_abaaaaab +=  1.00 * einsum('dkcb,ijda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaab += -1.00 * einsum('lkab,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,ikdb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkdb,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaaaab += -1.00 * einsum('lkdb,dl,ijac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaaaab +=  1.00 * einsum('lkcb,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaaaab += -1.00 * einsum('ijcd,dl,lkab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljcd,dl,ikab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaaaab += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jc,ikab->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abaaaaab


def get_lhe2e2cc_abaaaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abaaaabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abaaaabb += -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abaaaabb += -1.00 * einsum('jkcb,ia->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abaaaabb +=  1.00 * einsum('ikcb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkal,ilcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abaaaabb +=  1.00 * einsum('jkcl,ilab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abaaaabb += -1.00 * einsum('ikcl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abaaaabb += -1.00 * einsum('dkab,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jdac,ikdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_abaaaabb +=  1.00 * einsum('dkcb,ijda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaabb += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jc,ikab->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abaaaabb


def get_lhe2e2cc_abaaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abaaabab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abaaabab += -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abaaabab += -1.00 * einsum('jkcb,ia->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abaaabab +=  1.00 * einsum('ikcb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('jklb,ilac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_abaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abaaabab += -1.00 * einsum('ijcl,lkab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate = -1.00 * einsum('jdcb,ikad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_abaaabab += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jc,ikab->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abaaabab


def get_lhe2e2cc_abaaabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abaaabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abaaabbb += -1.00 * einsum('ijac,kb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abaaabbb += -1.00 * einsum('jkcb,ia->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abaaabbb +=  1.00 * einsum('ikcb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkal,ilcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abaaabbb +=  1.00 * einsum('jkcl,ilab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abaaabbb += -1.00 * einsum('ikcl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('jdcb,ikad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcb,dl,ikad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkad,dl,ilcb->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaabbb +=  1.00 * einsum('lkbd,dl,ijac->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaabbb +=  1.00 * einsum('jkcd,dl,ilab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('jlcd,dl,ikab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaabbb += -1.00 * einsum('ikcd,dl,jlab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abaaabbb += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jc,ikab->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abaaabbb


def get_lhe2e2cc_ababaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababaaaa =  1.00 * einsum('kiab,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('jicb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kjac,ib->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_ababaaaa += -1.00 * einsum('kicb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaaa += -1.00 * einsum('kjal,licb->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kilb,jlac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kjcl,liab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kdac,jidb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_ababaaaa += -1.00 * einsum('jdac,kidb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_ababaaaa += -1.00 * einsum('dicb,kjda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('lkac,dl,jidb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('ljac,dl,kidb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('licb,dl,kjad->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('kjad,dl,licb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('kidb,dl,ljac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('lkad,dl,jicb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kjcd,dl,liab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa += -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababaaaa +=  1.00 * einsum('ka,jicb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_ababaaaa


def get_lhe2e2cc_ababaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababaaba =  1.00 * einsum('kiab,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaba +=  1.00 * einsum('jicb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaba +=  1.00 * einsum('kjac,ib->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_ababaaba += -1.00 * einsum('kicb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_ababaaba += -1.00 * einsum('kial,jlcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_ababaaba += -1.00 * einsum('jicl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_ababaaba +=  1.00 * einsum('kicl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_ababaaba +=  1.00 * einsum('kdac,jidb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_ababaaba += -1.00 * einsum('jdac,kidb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_ababaaba += -1.00 * einsum('dicb,kjda->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaba +=  1.00 * einsum('ka,jicb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_ababaaba +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_ababaaba


def get_lhe2e2cc_abababaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abababaa =  1.00 * einsum('kiab,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababaa +=  1.00 * einsum('jicb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababaa +=  1.00 * einsum('kjac,ib->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abababaa += -1.00 * einsum('kicb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababaa += -1.00 * einsum('kjal,licb->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abababaa +=  1.00 * einsum('kilb,jlac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_abababaa +=  1.00 * einsum('kjcl,liab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abababaa +=  1.00 * einsum('kdab,jicd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababaa +=  1.00 * einsum('jdcb,kiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababaa += -1.00 * einsum('kdcb,jiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababaa +=  1.00 * einsum('ka,jicb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abababaa +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abababaa


def get_lhe2e2cc_abababba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abababba =  1.00 * einsum('kiab,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababba +=  1.00 * einsum('jicb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababba +=  1.00 * einsum('kjac,ib->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abababba += -1.00 * einsum('kicb,ja->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abababba += -1.00 * einsum('kial,jlcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abababba += -1.00 * einsum('jicl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('kicl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('kdab,jicd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('jdcb,kiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababba += -1.00 * einsum('kdcb,jiad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abababba += -1.00 * einsum('klab,dl,jicd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba += -1.00 * einsum('jlcb,dl,kiad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba += -1.00 * einsum('kiad,dl,jlcb->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba +=  1.00 * einsum('klad,dl,jicb->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba +=  1.00 * einsum('klcb,dl,jiad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba += -1.00 * einsum('jicd,dl,klab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba +=  1.00 * einsum('jlcd,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba +=  1.00 * einsum('kicd,dl,jlab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abababba +=  1.00 * einsum('ka,jicb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abababba


def get_lhe2e2cc_ababbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababbaab =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_ababbaab +=  1.00 * einsum('kibc,ja->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_ababbaab += -1.00 * einsum('jklb,liac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jilc,lkab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('dkab,jidc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_ababbaab +=  1.00 * einsum('diac,jkdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('lkab,dl,jidc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    lhe2e2cc_ababbaab += -1.00 * einsum('liac,dl,jkdb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbaab += -1.00 * einsum('jkdb,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbaab +=  1.00 * einsum('lkdb,dl,jiac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('jidc,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_ababbaab +=  1.00 * einsum('lidc,dl,jkab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbaab +=  1.00 * einsum('kb,jiac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00 * einsum('ic,jkab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_ababbaab


def get_lhe2e2cc_ababbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababbabb =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_ababbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_ababbabb +=  1.00 * einsum('kibc,ja->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_ababbabb +=  1.00 * einsum('jkal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_ababbabb += -1.00 * einsum('kibl,jlac->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_ababbabb +=  1.00 * einsum('kicl,jlab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('dkab,jidc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_ababbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_ababbabb +=  1.00 * einsum('diac,jkdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_ababbabb +=  1.00 * einsum('kb,jiac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_ababbabb +=  1.00 * einsum('ic,jkab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_ababbabb


def get_lhe2e2cc_ababbbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababbbab =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_ababbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_ababbbab +=  1.00 * einsum('kibc,ja->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_ababbbab += -1.00 * einsum('jklb,liac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jilc,lkab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_ababbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_ababbbab += -1.00 * einsum('jdac,kidb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_ababbbab += -1.00 * einsum('idbc,jkad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_ababbbab +=  1.00 * einsum('kdbc,jiad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_ababbbab +=  1.00 * einsum('kb,jiac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_ababbbab +=  1.00 * einsum('ic,jkab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_ababbbab


def get_lhe2e2cc_ababbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_ababbbbb =  1.00 * einsum('jkab,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('jiac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kibc,ja->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_ababbbbb +=  1.00 * einsum('jkal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_ababbbbb += -1.00 * einsum('kibl,jlac->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kicl,jlab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_ababbbbb += -1.00 * einsum('jdac,kidb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_ababbbbb += -1.00 * einsum('idbc,jkad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kdbc,jiad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_ababbbbb += -1.00 * einsum('jlac,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb += -1.00 * einsum('libc,dl,jkad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb += -1.00 * einsum('jkad,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb += -1.00 * einsum('kibd,dl,jlac->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb += -1.00 * einsum('lkbd,dl,jiac->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb +=  1.00 * einsum('lkbc,dl,jiad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb += -1.00 * einsum('licd,dl,jkab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kicd,dl,jlab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kb,jiac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_ababbbbb +=  1.00 * einsum('ic,jkab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_ababbbbb


def get_lhe2e2cc_abbaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbaaaaa = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ijcb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kjlb,ilac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('kial,ljcb->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kicl,ljab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kdac,ijdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('djcb,kida->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('idac,kjdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abbaaaaa += -1.00 * einsum('lkac,dl,ijdb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('ljcb,dl,kiad->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('liac,dl,kjdb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('kjdb,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('kiad,dl,ljcb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('lkad,dl,ijcb->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('licd,dl,kjab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kicd,dl,ljab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaaaaa += -1.00 * einsum('ka,ijcb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abbaaaaa += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abbaaaaa


def get_lhe2e2cc_abbaaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbaaaba = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ijcb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_abbaaaba += -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abbaaaba +=  1.00 * einsum('kjal,ilcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_abbaaaba += -1.00 * einsum('kdac,ijdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abbaaaba +=  1.00 * einsum('djcb,kida->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_abbaaaba +=  1.00 * einsum('idac,kjdb->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_abbaaaba += -1.00 * einsum('ka,ijcb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abbaaaba += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abbaaaba


def get_lhe2e2cc_abbaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbaabaa = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ijcb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_abbaabaa += -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abbaabaa += -1.00 * einsum('kjlb,ilac->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_abbaabaa +=  1.00 * einsum('kial,ljcb->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abbaabaa += -1.00 * einsum('kicl,ljab->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_abbaabaa += -1.00 * einsum('kdab,ijcd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabaa += -1.00 * einsum('idcb,kjad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabaa +=  1.00 * einsum('kdcb,ijad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabaa += -1.00 * einsum('ka,ijcb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abbaabaa += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abbaabaa


def get_lhe2e2cc_abbaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbaabba = -1.00 * einsum('kjab,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('ijcb,ka->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_abbaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_abbaabba += -1.00 * einsum('kiac,jb->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_abbaabba +=  1.00 * einsum('kjal,ilcb->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_abbaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_abbaabba += -1.00 * einsum('kdab,ijcd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('idcb,kjad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('kdcb,ijad->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('klab,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba +=  1.00 * einsum('ilcb,dl,kjad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba +=  1.00 * einsum('kjad,dl,ilcb->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba += -1.00 * einsum('klad,dl,ijcb->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba += -1.00 * einsum('klcb,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ijcd,dl,klab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abbaabba += -1.00 * einsum('ilcd,dl,kjab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbaabba += -1.00 * einsum('ka,ijcb->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_abbaabba


def get_lhe2e2cc_abbabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbabaab = -1.00 * einsum('ikab,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabaab += -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabaab += -1.00 * einsum('kjbc,ia->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbabaab +=  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabaab +=  1.00 * einsum('iklb,ljac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00 * einsum('ijlc,lkab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbabaab += -1.00 * einsum('iklc,ljab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('dkab,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_abbabaab += -1.00 * einsum('djac,ikdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('lkab,dl,ijdc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    lhe2e2cc_abbabaab +=  1.00 * einsum('ljac,dl,ikdb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab +=  1.00 * einsum('ikdb,dl,ljac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab += -1.00 * einsum('lkdb,dl,ijac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab +=  1.00 * einsum('ijdc,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab += -1.00 * einsum('ljdc,dl,ikab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab += -1.00 * einsum('ikdc,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabaab += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbabaab += -1.00 * einsum('jc,ikab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_abbabaab


def get_lhe2e2cc_abbababb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbababb = -1.00 * einsum('ikab,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbababb += -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbababb += -1.00 * einsum('kjbc,ia->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbababb +=  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbababb +=  1.00 * einsum('kjbl,ilac->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_abbababb += -1.00 * einsum('ikal,jlbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_abbababb += -1.00 * einsum('kjcl,ilab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('dkab,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_abbababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_abbababb += -1.00 * einsum('djac,ikdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_abbababb += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbababb += -1.00 * einsum('jc,ikab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_abbababb


def get_lhe2e2cc_abbabbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbabbab = -1.00 * einsum('ikab,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbab += -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbab += -1.00 * einsum('kjbc,ia->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbabbab +=  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbab +=  1.00 * einsum('iklb,ljac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbabbab +=  1.00 * einsum('ijlc,lkab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbabbab += -1.00 * einsum('iklc,ljab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbabbab +=  1.00 * einsum('jdbc,ikad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbabbab +=  1.00 * einsum('idac,kjdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_abbabbab += -1.00 * einsum('kdbc,ijad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbabbab += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbabbab += -1.00 * einsum('jc,ikab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_abbabbab


def get_lhe2e2cc_abbabbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_abbabbbb = -1.00 * einsum('ikab,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbbb += -1.00 * einsum('ijac,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbbb += -1.00 * einsum('kjbc,ia->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ikac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbabbbb +=  1.00 * einsum('kjbl,ilac->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_abbabbbb += -1.00 * einsum('ikal,jlbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_abbabbbb += -1.00 * einsum('kjcl,ilab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00 * einsum('jdbc,ikad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00 * einsum('idac,kjdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_abbabbbb += -1.00 * einsum('kdbc,ijad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ljbc,dl,ikad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ilac,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb +=  1.00 * einsum('kjbd,dl,ilac->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ikad,dl,ljbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb +=  1.00 * einsum('lkbd,dl,ijac->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb += -1.00 * einsum('lkbc,dl,ijad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb += -1.00 * einsum('kjcd,dl,ilab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ljcd,dl,ikab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbabbbb += -1.00 * einsum('kb,ijac->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbabbbb += -1.00 * einsum('jc,ikab->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_abbabbbb


def get_lhe2e2cc_abbbbaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abbbbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('ijbc,ka->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('kjac,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbbbaaa += -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('kjlb,liac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa += -1.00 * einsum('kjlc,liab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('kilc,ljab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljac,dl,kidb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjdb,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa += -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbaaa += -1.00 * einsum('kjdc,dl,liab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljdc,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('kidc,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abbbbaaa


def get_lhe2e2cc_abbbbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abbbbaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abbbbaba +=  1.00 * einsum('ijbc,ka->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbbbaba +=  1.00 * einsum('kjac,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbbbaba += -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_abbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abbbbaba +=  1.00 * einsum('ijcl,klab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('djac,kidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_abbbbaba +=  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abbbbaba


def get_lhe2e2cc_abbbbbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abbbbbaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abbbbbaa +=  1.00 * einsum('ijbc,ka->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbbbbaa +=  1.00 * einsum('kjac,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbbbbaa += -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('kjlb,liac->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abbbbbaa += -1.00 * einsum('kjlc,liab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_abbbbbaa +=  1.00 * einsum('kilc,ljab->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,kiad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_abbbbbaa +=  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abbbbbaa


def get_lhe2e2cc_abbbbbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_abbbbbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_abbbbbba +=  1.00 * einsum('ijbc,ka->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_abbbbbba +=  1.00 * einsum('kjac,ib->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_abbbbbba += -1.00 * einsum('kiac,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_abbbbbba +=  1.00 * einsum('ijcl,klab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,kiad->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klab,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,dl,kiad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbbba +=  1.00 * einsum('klad,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbbba +=  1.00 * einsum('ijcd,dl,klab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbbba +=  1.00 * einsum('ka,ijbc->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_abbbbbba


def get_lhe2e2cc_baaaaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_baaaaaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_baaaaaab +=  1.00 * einsum('ijbc,ka->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baaaaaab +=  1.00 * einsum('jkca,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baaaaaab += -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkla,ilbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_baaaaaab +=  1.00 * einsum('ijcl,lkba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('dkba,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,ikda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkba,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,dl,ikda->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkda,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaaaab +=  1.00 * einsum('lkda,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaaaab +=  1.00 * einsum('ijcd,dl,lkba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ljcd,dl,ikba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaaaab +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jc,ikba->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_baaaaaab


def get_lhe2e2cc_baaaaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_baaaaabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_baaaaabb +=  1.00 * einsum('ijbc,ka->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baaaaabb +=  1.00 * einsum('jkca,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baaaaabb += -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('jkbl,ilca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_baaaaabb += -1.00 * einsum('jkcl,ilba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baaaaabb +=  1.00 * einsum('ikcl,jlba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('dkba,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdbc,ikda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_baaaaabb +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jc,ikba->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_baaaaabb


def get_lhe2e2cc_baaaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_baaaabab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_baaaabab +=  1.00 * einsum('ijbc,ka->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baaaabab +=  1.00 * einsum('jkca,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baaaabab += -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jkla,ilbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_baaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_baaaabab +=  1.00 * einsum('ijcl,lkba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdca,ikbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_baaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_baaaabab +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jc,ikba->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_baaaabab


def get_lhe2e2cc_baaaabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_baaaabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ijbc,ka->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baaaabbb +=  1.00 * einsum('jkca,ib->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baaaabbb += -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('jkbl,ilca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_baaaabbb += -1.00 * einsum('jkcl,ilba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ikcl,jlba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdca,ikbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlca,dl,ikbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkbd,dl,ilca->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaabbb += -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaabbb += -1.00 * einsum('jkcd,dl,ilba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('jlcd,dl,ikba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ikcd,dl,jlba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jc,ikba->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_baaaabbb


def get_lhe2e2cc_baabaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabaaaa = -1.00 * einsum('kiba,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaaa += -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaaa += -1.00 * einsum('kjbc,ia->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baabaaaa +=  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaaa +=  1.00 * einsum('kjbl,lica->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_baabaaaa += -1.00 * einsum('kila,jlbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_baabaaaa += -1.00 * einsum('kjcl,liba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00 * einsum('jdbc,kida->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00 * einsum('dica,kjdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaaa += -1.00 * einsum('kdbc,jida->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00 * einsum('ljbc,dl,kida->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa +=  1.00 * einsum('lica,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa +=  1.00 * einsum('kjbd,dl,lica->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa +=  1.00 * einsum('kida,dl,ljbc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa +=  1.00 * einsum('lkbd,dl,jica->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa += -1.00 * einsum('lkbc,dl,jida->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa += -1.00 * einsum('kjcd,dl,liba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa +=  1.00 * einsum('ljcd,dl,kiba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabaaaa += -1.00 * einsum('kb,jica->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baabaaaa += -1.00 * einsum('jc,kiba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_baabaaaa


def get_lhe2e2cc_baabaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabaaba = -1.00 * einsum('kiba,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaba += -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaba += -1.00 * einsum('kjbc,ia->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baabaaba +=  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baabaaba +=  1.00 * einsum('kibl,jlca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baabaaba +=  1.00 * einsum('jicl,klba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baabaaba += -1.00 * einsum('kicl,jlba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baabaaba +=  1.00 * einsum('jdbc,kida->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baabaaba +=  1.00 * einsum('dica,kjdb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaba += -1.00 * einsum('kdbc,jida->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_baabaaba += -1.00 * einsum('kb,jica->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baabaaba += -1.00 * einsum('jc,kiba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_baabaaba


def get_lhe2e2cc_baababaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baababaa = -1.00 * einsum('kiba,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababaa += -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababaa += -1.00 * einsum('kjbc,ia->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baababaa +=  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababaa +=  1.00 * einsum('kjbl,lica->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_baababaa += -1.00 * einsum('kila,jlbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_baababaa += -1.00 * einsum('kjcl,liba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate = -1.00 * einsum('kdba,jicd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_baababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_baababaa += -1.00 * einsum('jdca,kibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_baababaa += -1.00 * einsum('kb,jica->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baababaa += -1.00 * einsum('jc,kiba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_baababaa


def get_lhe2e2cc_baababba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baababba = -1.00 * einsum('kiba,jc->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababba += -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababba += -1.00 * einsum('kjbc,ia->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_baababba +=  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_baababba +=  1.00 * einsum('kibl,jlca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baababba +=  1.00 * einsum('jicl,klba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_baababba += -1.00 * einsum('kicl,jlba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('kdba,jicd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_baababba += -1.00 * einsum('jdca,kibd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('klba,dl,jicd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    lhe2e2cc_baababba +=  1.00 * einsum('jlca,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba +=  1.00 * einsum('kibd,dl,jlca->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba += -1.00 * einsum('klbd,dl,jica->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba +=  1.00 * einsum('jicd,dl,klba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba += -1.00 * einsum('jlcd,dl,kiba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba += -1.00 * einsum('kicd,dl,jlba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baababba += -1.00 * einsum('kb,jica->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_baababba += -1.00 * einsum('jc,kiba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_baababba


def get_lhe2e2cc_baabbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabbaab = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('jibc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_baabbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_baabbaab += -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_baabbaab +=  1.00 * einsum('jkla,libc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jilc,lkba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_baabbaab += -1.00 * einsum('dkba,jidc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('dibc,jkda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('dkbc,jida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('lkba,dl,jidc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab +=  1.00 * einsum('libc,dl,jkda->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab +=  1.00 * einsum('jkda,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab += -1.00 * einsum('lkda,dl,jibc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab += -1.00 * einsum('lkbc,dl,jida->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('jidc,dl,lkba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_baabbaab += -1.00 * einsum('lidc,dl,jkba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbaab += -1.00 * einsum('ka,jibc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('ic,jkba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_baabbaab


def get_lhe2e2cc_baabbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabbabb = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('jibc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_baabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_baabbabb += -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_baabbabb += -1.00 * einsum('jkbl,ilac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_baabbabb +=  1.00 * einsum('kial,jlbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_baabbabb += -1.00 * einsum('kicl,jlba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_baabbabb += -1.00 * einsum('dkba,jidc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbabb += -1.00 * einsum('dibc,jkda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbabb +=  1.00 * einsum('dkbc,jida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_baabbabb += -1.00 * einsum('ka,jibc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_baabbabb += -1.00 * einsum('ic,jkba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_baabbabb


def get_lhe2e2cc_baabbbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabbbab = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('jibc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_baabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_baabbbab += -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_baabbbab +=  1.00 * einsum('jkla,libc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jilc,lkba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_baabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_baabbbab += -1.00 * einsum('kdac,jibd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_baabbbab +=  1.00 * einsum('jdbc,kida->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_baabbbab +=  1.00 * einsum('idac,jkbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_baabbbab += -1.00 * einsum('ka,jibc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_baabbbab += -1.00 * einsum('ic,jkba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_baabbbab


def get_lhe2e2cc_baabbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_baabbbbb = -1.00 * einsum('jkba,ic->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('jibc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_baabbbbb += -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_baabbbbb += -1.00 * einsum('jkbl,ilac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_baabbbbb +=  1.00 * einsum('kial,jlbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_baabbbbb += -1.00 * einsum('kicl,jlba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_baabbbbb += -1.00 * einsum('kdac,jibd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_baabbbbb +=  1.00 * einsum('jdbc,kida->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_baabbbbb +=  1.00 * einsum('idac,jkbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_baabbbbb += -1.00 * einsum('lkac,dl,jibd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('jlbc,dl,kiad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('liac,dl,jkbd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('jkbd,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('kiad,dl,jlbc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('lkad,dl,jibc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb +=  1.00 * einsum('licd,dl,jkba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb += -1.00 * einsum('kicd,dl,jlba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_baabbbbb += -1.00 * einsum('ka,jibc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_baabbbbb += -1.00 * einsum('ic,jkba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_baabbbbb


def get_lhe2e2cc_babaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_babaaaaa =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kibc,ja->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kjla,ilbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_babaaaaa += -1.00 * einsum('kibl,ljca->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kicl,ljba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_babaaaaa += -1.00 * einsum('djca,kidb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_babaaaaa += -1.00 * einsum('idbc,kjda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kdbc,ijda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_babaaaaa += -1.00 * einsum('ljca,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa += -1.00 * einsum('libc,dl,kjda->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa += -1.00 * einsum('kjda,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa += -1.00 * einsum('kibd,dl,ljca->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa += -1.00 * einsum('lkbd,dl,ijca->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa +=  1.00 * einsum('lkbc,dl,ijda->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa += -1.00 * einsum('licd,dl,kjba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kicd,dl,ljba->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kb,ijca->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_babaaaaa +=  1.00 * einsum('ic,kjba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_babaaaaa


def get_lhe2e2cc_babaaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_babaaaba =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_babaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_babaaaba +=  1.00 * einsum('kibc,ja->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_babaaaba += -1.00 * einsum('kjbl,ilca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijcl,klba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_babaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_babaaaba += -1.00 * einsum('djca,kidb->abjicdk', g_abab[va, ob, va, vb], l2_aaaa)
    lhe2e2cc_babaaaba += -1.00 * einsum('idbc,kjda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_babaaaba +=  1.00 * einsum('kdbc,ijda->abjicdk', g_aaaa[oa, va, va, va], l2_abab)
    lhe2e2cc_babaaaba +=  1.00 * einsum('kb,ijca->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_babaaaba +=  1.00 * einsum('ic,kjba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_babaaaba


def get_lhe2e2cc_babaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_babaabaa =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_babaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_babaabaa +=  1.00 * einsum('kibc,ja->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_babaabaa +=  1.00 * einsum('kjla,ilbc->abjiclk', g_abab[oa, ob, oa, vb], l2_aaaa)
    lhe2e2cc_babaabaa += -1.00 * einsum('kibl,ljca->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    lhe2e2cc_babaabaa +=  1.00 * einsum('kicl,ljba->abjiclk', g_aaaa[oa, oa, va, oa], l2_abab)
    contracted_intermediate =  1.00 * einsum('kdba,ijcd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_babaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_babaabaa +=  1.00 * einsum('idca,kjbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_babaabaa +=  1.00 * einsum('kb,ijca->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_babaabaa +=  1.00 * einsum('ic,kjba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_babaabaa


def get_lhe2e2cc_babaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_babaabba =  1.00 * einsum('kjba,ic->abjick', g_abab[oa, ob, va, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_aa)
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate) 
    lhe2e2cc_babaabba +=  1.00 * einsum('kibc,ja->abjick', g_aaaa[oa, oa, va, va], l1_bb)
    lhe2e2cc_babaabba += -1.00 * einsum('kjbl,ilca->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijcl,klba->abjiclk', g_abab[oa, ob, va, ob], l2_abab)
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kdba,ijcd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    lhe2e2cc_babaabba +=  1.00 * einsum('idca,kjbd->abjicdk', g_abab[oa, vb, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('klba,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    lhe2e2cc_babaabba += -1.00 * einsum('ilca,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaabba += -1.00 * einsum('kjbd,dl,ilca->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaabba +=  1.00 * einsum('klbd,dl,ijca->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,klba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_babaabba +=  1.00 * einsum('ilcd,dl,kjba->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babaabba +=  1.00 * einsum('kb,ijca->abjick', f_aa[oa, va], l2_abab)
    lhe2e2cc_babaabba +=  1.00 * einsum('ic,kjba->abjick', f_aa[oa, va], l2_abab)
    return lhe2e2cc_babaabba


def get_lhe2e2cc_bababaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bababaab =  1.00 * einsum('ikba,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababaab +=  1.00 * einsum('ijbc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababaab +=  1.00 * einsum('kjac,ib->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_bababaab += -1.00 * einsum('ikbc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababaab += -1.00 * einsum('ikla,ljbc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababaab += -1.00 * einsum('ijlc,lkba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('iklc,ljba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('dkba,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('djbc,ikda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab += -1.00 * einsum('dkbc,ijda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab += -1.00 * einsum('lkba,dl,ijdc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab += -1.00 * einsum('ljbc,dl,ikda->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab += -1.00 * einsum('ikda,dl,ljbc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab +=  1.00 * einsum('lkda,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab +=  1.00 * einsum('lkbc,dl,ijda->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab += -1.00 * einsum('ijdc,dl,lkba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab +=  1.00 * einsum('ljdc,dl,ikba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab +=  1.00 * einsum('ikdc,dl,ljba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababaab +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('jc,ikba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_bababaab


def get_lhe2e2cc_babababb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_babababb =  1.00 * einsum('ikba,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babababb +=  1.00 * einsum('ijbc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babababb +=  1.00 * einsum('kjac,ib->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_babababb += -1.00 * einsum('ikbc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babababb += -1.00 * einsum('kjal,ilbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_babababb +=  1.00 * einsum('ikbl,jlac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_babababb +=  1.00 * einsum('kjcl,ilba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_babababb +=  1.00 * einsum('dkba,ijdc->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_babababb +=  1.00 * einsum('djbc,ikda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_babababb += -1.00 * einsum('dkbc,ijda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_babababb +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_babababb +=  1.00 * einsum('jc,ikba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_babababb


def get_lhe2e2cc_bababbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bababbab =  1.00 * einsum('ikba,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbab +=  1.00 * einsum('ijbc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbab +=  1.00 * einsum('kjac,ib->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_bababbab += -1.00 * einsum('ikbc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbab += -1.00 * einsum('ikla,ljbc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababbab += -1.00 * einsum('ijlc,lkba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababbab +=  1.00 * einsum('iklc,ljba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bababbab +=  1.00 * einsum('kdac,ijbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_bababbab += -1.00 * einsum('jdac,ikbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_bababbab += -1.00 * einsum('idbc,kjda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bababbab +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bababbab +=  1.00 * einsum('jc,ikba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_bababbab


def get_lhe2e2cc_bababbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bababbbb =  1.00 * einsum('ikba,jc->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbbb +=  1.00 * einsum('ijbc,ka->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbbb +=  1.00 * einsum('kjac,ib->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_bababbbb += -1.00 * einsum('ikbc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bababbbb += -1.00 * einsum('kjal,ilbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bababbbb +=  1.00 * einsum('ikbl,jlac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bababbbb +=  1.00 * einsum('kjcl,ilba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bababbbb +=  1.00 * einsum('kdac,ijbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_bababbbb += -1.00 * einsum('jdac,ikbd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_bababbbb += -1.00 * einsum('idbc,kjda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bababbbb +=  1.00 * einsum('lkac,dl,ijbd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('ljac,dl,ikbd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('ilbc,dl,kjad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('kjad,dl,ilbc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('ikbd,dl,ljac->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb +=  1.00 * einsum('kjcd,dl,ilba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb += -1.00 * einsum('ljcd,dl,ikba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bababbbb +=  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bababbbb +=  1.00 * einsum('jc,ikba->abjick', f_bb[ob, vb], l2_abab)
    return lhe2e2cc_bababbbb


def get_lhe2e2cc_babbbaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_babbbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_babbbaaa += -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_babbbaaa += -1.00 * einsum('kjbc,ia->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babbbaaa +=  1.00 * einsum('kibc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjla,libc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_babbbaaa +=  1.00 * einsum('kjlc,liba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_babbbaaa += -1.00 * einsum('kilc,ljba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('djbc,kida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljbc,dl,kida->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjda,dl,libc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbaaa +=  1.00 * einsum('lkbd,dl,ijac->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbaaa +=  1.00 * einsum('kjdc,dl,liba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ljdc,dl,kiba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbaaa += -1.00 * einsum('kidc,dl,ljba->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbaaa += -1.00 * einsum('kb,ijac->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jc,kiba->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_babbbaaa


def get_lhe2e2cc_babbbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_babbbaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_babbbaba += -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_babbbaba += -1.00 * einsum('kjbc,ia->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babbbaba +=  1.00 * einsum('kibc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('kjbl,ilac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_babbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_babbbaba += -1.00 * einsum('ijcl,klba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    contracted_intermediate = -1.00 * einsum('djbc,kida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_babbbaba += -1.00 * einsum('kb,ijac->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jc,kiba->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_babbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_babbbaba


def get_lhe2e2cc_babbbbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_babbbbaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_babbbbaa += -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_babbbbaa += -1.00 * einsum('kjbc,ia->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babbbbaa +=  1.00 * einsum('kibc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate = -1.00 * einsum('kjla,libc->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_babbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_babbbbaa +=  1.00 * einsum('kjlc,liba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_babbbbaa += -1.00 * einsum('kilc,ljba->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_babbbbaa += -1.00 * einsum('kdba,ijdc->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_babbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_babbbbaa +=  1.00 * einsum('kdbc,ijda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_babbbbaa += -1.00 * einsum('kb,ijac->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jc,kiba->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_babbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_babbbbaa


def get_lhe2e2cc_babbbbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_babbbbba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    lhe2e2cc_babbbbba += -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    lhe2e2cc_babbbbba += -1.00 * einsum('kjbc,ia->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_babbbbba +=  1.00 * einsum('kibc,ja->abjick', g_abab[oa, ob, va, vb], l1_bb)
    contracted_intermediate =  1.00 * einsum('kjbl,ilac->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate) 
    lhe2e2cc_babbbbba += -1.00 * einsum('ijcl,klba->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_babbbbba += -1.00 * einsum('kdba,ijdc->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jdac,kibd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate) 
    lhe2e2cc_babbbbba +=  1.00 * einsum('kdbc,ijda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_babbbbba += -1.00 * einsum('klba,dl,ijcd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjbd,dl,liac->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbbba += -1.00 * einsum('klbd,dl,ijac->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbbba +=  1.00 * einsum('klbc,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbbba += -1.00 * einsum('ijcd,dl,klba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ljcd,dl,kiba->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbbba += -1.00 * einsum('kb,ijac->abjick', f_aa[oa, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jc,kiba->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_babbbbba


def get_lhe2e2cc_bbabaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbabaaab =  1.00 * einsum('kiab,jc->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkla,licb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbabaaab +=  1.00 * einsum('dkca,jidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('dica,jkdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbabaaab += -1.00 * einsum('dkcb,jida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab += -1.00 * einsum('lkca,dl,jidb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('lica,dl,jkdb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkda,dl,licb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,dl,jicb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_bbabaaab +=  1.00 * einsum('lkcb,dl,jida->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbabaaab += -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,jicb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbabaaab +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbabaaab


def get_lhe2e2cc_bbabaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbabaabb =  1.00 * einsum('kiab,jc->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbabaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,jlcb->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bbabaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jicl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbabaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_bbabaabb +=  1.00 * einsum('dkca,jidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('dica,jkdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbabaabb += -1.00 * einsum('dkcb,jida->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ka,jicb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbabaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbabaabb +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbabaabb


def get_lhe2e2cc_bbababab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbababab =  1.00 * einsum('kiab,jc->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkla,licb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbababab +=  1.00 * einsum('kdab,jicd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdca,kidb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,jicb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbababab +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbababab


def get_lhe2e2cc_bbababbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbababbb =  1.00 * einsum('kiab,jc->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate = -1.00 * einsum('jica,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kial,jlcb->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jicl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_bbababbb +=  1.00 * einsum('kdab,jicd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jdca,kidb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbababbb +=  1.00 * einsum('lkab,dl,jicd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('jlca,dl,kibd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiad,dl,jlcb->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,jicb->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jicd,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_bbababbb +=  1.00 * einsum('jlcd,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,jicb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbababbb +=  1.00 * einsum('jc,kiab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbababbb


def get_lhe2e2cc_bbbaaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbbaaaab = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikla,ljcb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbbaaaab += -1.00 * einsum('dkca,ijdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('djca,ikdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbbaaaab +=  1.00 * einsum('dkcb,ijda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00 * einsum('lkca,dl,ijdb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ljca,dl,ikdb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikda,dl,ljcb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkda,dl,ijcb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_bbbaaaab += -1.00 * einsum('lkcb,dl,ijda->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaaaab +=  1.00 * einsum('licd,dl,kjab->abjicdlk', g_aaaa[oa, oa, va, va], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ka,ijcb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbaaaab += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbbaaaab


def get_lhe2e2cc_bbbaaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbbaaabb = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,ilcb->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bbbaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbbaaabb +=  1.00 * einsum('ijcl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbaaabb += -1.00 * einsum('ikcl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbaaabb += -1.00 * einsum('dkca,ijdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('djca,ikdb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbbaaabb +=  1.00 * einsum('dkcb,ijda->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ka,ijcb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbbaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbaaabb += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbbaaabb


def get_lhe2e2cc_bbbaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbbaabab = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikla,ljcb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbbaabab += -1.00 * einsum('kdab,ijcd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('idca,kjdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,ijcb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbaabab += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbbaabab


def get_lhe2e2cc_bbbaabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    
    lhe2e2cc_bbbaabbb = -1.00 * einsum('kjab,ic->abjick', g_bbbb[ob, ob, vb, vb], l1_aa)
    contracted_intermediate =  1.00 * einsum('ijca,kb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjal,ilcb->abjiclk', g_bbbb[ob, ob, vb, ob], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate) 
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ijcl,klab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ikcl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('kdab,ijcd->abjicdk', g_bbbb[ob, vb, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('idca,kjdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate) 
    lhe2e2cc_bbbaabbb += -1.00 * einsum('lkab,dl,ijcd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ilca,dl,kjbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,ilcb->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,dl,ijcb->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ijcd,dl,lkab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ilcd,dl,kjab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ikcd,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ka,ijcb->abjick', f_bb[ob, vb], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ic,kjab->abjick', f_aa[oa, va], l2_bbbb)
    return lhe2e2cc_bbbaabbb


def get_lhe2e2cc_bbbbaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjla,licb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('djca,kidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljca,dl,kidb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjda,dl,licb->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return lhe2e2cc_bbbbaaaa


def get_lhe2e2cc_bbbbaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbaaba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbbaaba += -1.00 * einsum('kjcl,ilab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbbaaba +=  1.00 * einsum('kicl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('djca,kidb->abjicdk', g_abab[va, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    return lhe2e2cc_bbbbaaba


def get_lhe2e2cc_bbbbabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjla,licb->abjiclk', g_abab[oa, ob, oa, vb], l2_abab)
    lhe2e2cc_bbbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    lhe2e2cc_bbbbabaa +=  1.00 * einsum('kdca,ijdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabaa += -1.00 * einsum('kdcb,ijda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    return lhe2e2cc_bbbbabaa


def get_lhe2e2cc_bbbbabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbabba =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kica,jb->abjick', g_abab[oa, ob, va, vb], l1_bb)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    lhe2e2cc_bbbbabba += -1.00 * einsum('kjcl,ilab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00 * einsum('kicl,jlab->abjiclk', g_abab[oa, ob, va, ob], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00 * einsum('kdca,ijdb->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabba += -1.00 * einsum('kdcb,ijda->abjicdk', g_abab[oa, vb, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00 * einsum('klca,dl,ijbd->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbabba += -1.00 * einsum('klcb,dl,ijad->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbabba +=  1.00 * einsum('kjcd,dl,liab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbabba += -1.00 * einsum('kicd,dl,ljab->abjicdlk', g_abab[oa, ob, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    return lhe2e2cc_bbbbabba


def get_lhe2e2cc_bbbbbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbbaab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,dl,ijbc->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljdc,dl,kiab->abjicdlk', g_abab[oa, ob, va, vb], t1_aa, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_bbbbbaab


def get_lhe2e2cc_bbbbbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbbabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_bbbbbabb +=  1.00 * einsum('kicl,jlab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_bbbbbabb


def get_lhe2e2cc_bbbbbbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbbbab =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    lhe2e2cc_bbbbbbab +=  1.00 * einsum('kdbc,ijda->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_bbbbbbab


def get_lhe2e2cc_bbbbbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'd', 'l', 'k') """
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
    lhe2e2cc_bbbbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,kb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abjkci', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->bajkci', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jb->abjick', g_bbbb[ob, ob, vb, vb], l1_bb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjal,ilbc->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abijclk', contracted_intermediate)  + -1.00000 * einsum('abjiclk->bajiclk', contracted_intermediate)  +  1.00000 * einsum('abjiclk->baijclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcl,klab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjiclk->abjkcli', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb +=  1.00 * einsum('kicl,jlab->abjiclk', g_bbbb[ob, ob, vb, ob], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kdab,ijdc->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->acjibdk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jdac,kidb->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdk->abijcdk', contracted_intermediate)  + -1.00000 * einsum('abjicdk->bajicdk', contracted_intermediate)  +  1.00000 * einsum('abjicdk->baijcdk', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb +=  1.00 * einsum('kdbc,ijda->abjicdk', g_bbbb[ob, vb, vb, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('lkab,dl,ijcd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->acjibdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljac,dl,kibd->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjad,dl,libc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ijbc->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb += -1.00 * einsum('lkbc,dl,ijad->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ijcd,dl,lkab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,kiab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb += -1.00 * einsum('kicd,dl,ljab->abjicdlk', g_bbbb[ob, ob, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ka,ijbc->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,kiab->abjick', f_bb[ob, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    return lhe2e2cc_bbbbbbbb
