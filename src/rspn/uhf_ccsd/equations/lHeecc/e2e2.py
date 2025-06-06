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
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa +=  1.00 * einsum('klab,ijcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    lhe2e2cc_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaaaaaa +=  1.00 * einsum('licd,kjab->abjicdlk', g_aaaa[oa, oa, va, va], l2_aaaa)
    return lhe2e2cc_aaaaaaaa


def get_lhe2e2cc_aaaaabab(
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
    
    lhe2e2cc_aaaaabab =  1.00 * einsum('ljab,ikcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabab += -1.00 * einsum('liab,jkcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijac,lkbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkad,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaaabab +=  1.00 * einsum('liac,jkbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabab += -1.00 * einsum('lkad,ijbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('ijbc,lkad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkbd,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaaabab += -1.00 * einsum('libc,jkad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabab +=  1.00 * einsum('lkbd,ijac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jkcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_aaaaabab


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
    
    lhe2e2cc_aaaaabba = -1.00 * einsum('kjab,ilcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba +=  1.00 * einsum('kiab,jlcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba += -1.00 * einsum('ijac,klbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba +=  1.00 * einsum('jlad,kibc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kjac,ilbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilad,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_aaaaabba +=  1.00 * einsum('ijbc,klad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba += -1.00 * einsum('jlbd,kiac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('kjbc,ilad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilbd,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_aaaaabba +=  1.00 * einsum('jlcd,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaaabba += -1.00 * einsum('ilcd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
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
    
    lhe2e2cc_aaaabaab = -1.00 * einsum('ljab,ikdc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaabaab +=  1.00 * einsum('liab,jkdc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijad,lkbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaabaab += -1.00 * einsum('liad,jkbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaabaab +=  1.00 * einsum('lkac,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ijbd,lkac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_aaaabaab +=  1.00 * einsum('libd,jkac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaabaab += -1.00 * einsum('lkbc,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jkdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_aaaabaab


def get_lhe2e2cc_aaaababa(
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
    
    lhe2e2cc_aaaababa =  1.00 * einsum('kjab,ildc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa += -1.00 * einsum('kiab,jldc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa +=  1.00 * einsum('ijad,klbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa += -1.00 * einsum('jlac,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('kjad,ilbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_aaaababa += -1.00 * einsum('ijbd,klac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa +=  1.00 * einsum('jlbc,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kjbd,ilac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilbc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_aaaababa += -1.00 * einsum('jldc,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaaababa +=  1.00 * einsum('ildc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aaaababa


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
    
    contracted_intermediate =  1.00 * einsum('jlac,ikbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkac,ilbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilac,jkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlbc,ikad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkbc,ilad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilbc,jkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaaabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    return lhe2e2cc_aaaabbbb


def get_lhe2e2cc_aaababaa(
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
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_aaababaa += -1.00 * einsum('klab,jicd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa += -1.00 * einsum('jiad,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa +=  1.00 * einsum('ljac,kibd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa += -1.00 * einsum('kjac,libd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa += -1.00 * einsum('kiad,ljbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa +=  1.00 * einsum('liad,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa += -1.00 * einsum('lkac,jibd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa +=  1.00 * einsum('jibd,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa += -1.00 * einsum('ljbc,kiad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa +=  1.00 * einsum('kjbc,liad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa +=  1.00 * einsum('kibd,ljac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa += -1.00 * einsum('libd,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa +=  1.00 * einsum('lkbc,jiad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaababaa += -1.00 * einsum('jicd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa += -1.00 * einsum('kicd,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaababaa +=  1.00 * einsum('licd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aaababaa


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
    
    contracted_intermediate =  1.00 * einsum('ljab,kidc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('klab,jidc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('jiac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('ljad,kibc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('kjad,libc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('kiac,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('liac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('lkad,jibc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('jibc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('ljbd,kiac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('kjbd,liac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('kibc,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('libc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('lkbd,jiac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('jidc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa +=  1.00 * einsum('kidc,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aaabbaaa += -1.00 * einsum('lidc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aaabbaaa


def get_lhe2e2cc_aaabbbab(
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
    
    lhe2e2cc_aaabbbab = -1.00 * einsum('ljab,kicd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('jiac,lkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liac,jkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jibc,lkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('libc,jkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_aaabbbab += -1.00 * einsum('kicd,ljab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    return lhe2e2cc_aaabbbab


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
    
    lhe2e2cc_aaabbbba =  1.00 * einsum('kjab,licd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('jiac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jlbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klac,jibd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jibc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,jlad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klbc,jiad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aaabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_aaabbbba +=  1.00 * einsum('licd,kjab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    return lhe2e2cc_aaabbbba


def get_lhe2e2cc_aabaabaa(
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
    
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aabaabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_aabaabaa +=  1.00 * einsum('klab,ijcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijad,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aabaabaa +=  1.00 * einsum('kjad,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabaabaa +=  1.00 * einsum('kiac,ljbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aabaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijbd,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aabaabaa += -1.00 * einsum('kjbd,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabaabaa += -1.00 * einsum('kibc,ljad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aabaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aabaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aabaabaa +=  1.00 * einsum('kjcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aabaabaa


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
    
    contracted_intermediate = -1.00 * einsum('liab,kjdc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aababaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_aababaaa += -1.00 * einsum('klab,ijdc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aababaaa += -1.00 * einsum('kjac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa += -1.00 * einsum('kiad,ljbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('liad,kjbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aababaaa +=  1.00 * einsum('kjbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00 * einsum('kibd,ljac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('libd,kjac->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijdc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_aababaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_aababaaa += -1.00 * einsum('kjdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    return lhe2e2cc_aababaaa


def get_lhe2e2cc_aababbab(
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
    
    lhe2e2cc_aababbab =  1.00 * einsum('liab,kjcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ijac,lkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikac,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkac,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijbc,lkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikbc,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkbc,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_aababbab +=  1.00 * einsum('kjcd,liab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    return lhe2e2cc_aababbab


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
    
    lhe2e2cc_aababbba = -1.00 * einsum('kiab,ljcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,ilbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjbc,ilad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilbc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aababbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_aababbba += -1.00 * einsum('ljcd,kiab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    return lhe2e2cc_aababbba


def get_lhe2e2cc_aabbbbaa(
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
    
    lhe2e2cc_aabbbbaa =  1.00 * einsum('klab,ijcd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ljac,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_aabbbbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_aabbbbaa +=  1.00 * einsum('ijcd,klab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    return lhe2e2cc_aabbbbaa


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
    
    lhe2e2cc_abaaaaab = -1.00 * einsum('jkab,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaab +=  1.00 * einsum('ikab,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaab += -1.00 * einsum('lkab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('ijac,lkdb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,jkdb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcb,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijcd,lkab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abaaaaab += -1.00 * einsum('licd,jkab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_abaaaaab


def get_lhe2e2cc_abaaaaba(
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
    
    lhe2e2cc_abaaaaba =  1.00 * einsum('jlab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaba += -1.00 * einsum('ilab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaba +=  1.00 * einsum('klab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ijac,kldb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,ildb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlcb,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilcb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_abaaaaba +=  1.00 * einsum('ijcd,klab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjcd,ilab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_abaaaaba


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
    
    contracted_intermediate = -1.00 * einsum('jlab,ikcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilab,jkcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_abaaabbb += -1.00 * einsum('ijac,klbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abaaabbb +=  1.00 * einsum('jlad,ikcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkad,ilcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaabbb += -1.00 * einsum('ilad,jkcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00 * einsum('jlcb,ikad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkcb,ilad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaabbb += -1.00 * einsum('ilcb,jkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00 * einsum('lkbd,ijac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abaaabbb += -1.00 * einsum('jlcd,ikab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkcd,ilab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaaabbb +=  1.00 * einsum('ilcd,jkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abaaabbb


def get_lhe2e2cc_abaababb(
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
    
    contracted_intermediate =  1.00 * einsum('jlab,ikdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilab,jkdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_abaababb +=  1.00 * einsum('ijad,klbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abaababb += -1.00 * einsum('jlac,ikdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkac,ildb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaababb +=  1.00 * einsum('ilac,jkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb += -1.00 * einsum('jldb,ikac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkdb,ilac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaababb +=  1.00 * einsum('ildb,jkac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb += -1.00 * einsum('lkbc,ijad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abaababb +=  1.00 * einsum('jldc,ikab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkdc,ilab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abaababb += -1.00 * einsum('ildc,jkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abaababb


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
    
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljac,kidb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,lidb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkac,jidb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jicb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kicb,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('licb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_ababaaaa += -1.00 * einsum('ljcd,kiab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_ababaaaa +=  1.00 * einsum('kjcd,liab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_ababaaaa


def get_lhe2e2cc_abababab(
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
    
    lhe2e2cc_abababab = -1.00 * einsum('jkab,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab += -1.00 * einsum('liab,jkcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00 * einsum('lkab,jicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab += -1.00 * einsum('jiad,lkcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00 * einsum('ljac,kibd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abababab +=  1.00 * einsum('jkad,licb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('liad,jkcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abababab += -1.00 * einsum('jicb,lkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00 * einsum('jkcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00 * einsum('kibd,ljac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('licb,jkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abababab +=  1.00 * einsum('jicd,lkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab += -1.00 * einsum('jkcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababab += -1.00 * einsum('licd,jkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abababab


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
    
    lhe2e2cc_abababba =  1.00 * einsum('jlab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('kiab,jlcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba += -1.00 * einsum('klab,jicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jiad,klcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abababba += -1.00 * einsum('kjac,libd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abababba += -1.00 * einsum('kiad,jlcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00 * einsum('klad,jicb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jicb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abababba += -1.00 * einsum('kicb,jlad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba += -1.00 * einsum('libd,kjac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abababba +=  1.00 * einsum('klcb,jiad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jicd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abababba +=  1.00 * einsum('kicd,jlab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
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
    
    lhe2e2cc_ababbaab =  1.00 * einsum('jkab,lidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00 * einsum('liab,jkdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab += -1.00 * einsum('lkab,jidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00 * einsum('jiac,lkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab += -1.00 * einsum('ljad,kibc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_ababbaab += -1.00 * einsum('jkac,lidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liac,jkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_ababbaab +=  1.00 * einsum('jidb,lkac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab += -1.00 * einsum('jkdb,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab += -1.00 * einsum('kibc,ljad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('lidb,jkac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_ababbaab += -1.00 * einsum('jidc,lkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00 * einsum('jkdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaab +=  1.00 * einsum('lidc,jkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_ababbaab


def get_lhe2e2cc_ababbaba(
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
    
    lhe2e2cc_ababbaba = -1.00 * einsum('jlab,kidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba += -1.00 * einsum('kiab,jldc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba +=  1.00 * einsum('klab,jidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jiac,kldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_ababbaba +=  1.00 * einsum('kjad,libc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_ababbaba +=  1.00 * einsum('kiac,jldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba += -1.00 * einsum('klac,jidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jidb,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_ababbaba +=  1.00 * einsum('kidb,jlac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba +=  1.00 * einsum('libc,kjad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_ababbaba += -1.00 * einsum('kldb,jiac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jidc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_ababbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_ababbaba += -1.00 * einsum('kidc,jlab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_ababbaba


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
    
    contracted_intermediate =  1.00 * einsum('jlab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_ababbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,jlad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('libc,jkad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_ababbbbb +=  1.00 * einsum('kicd,jlab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_ababbbbb += -1.00 * einsum('licd,jkab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
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
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abbaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,ljdb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liac,kjdb->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijcb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_abbaaaaa += -1.00 * einsum('kicd,ljab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_abbaaaaa +=  1.00 * einsum('licd,kjab->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_abbaaaaa


def get_lhe2e2cc_abbaabab(
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
    
    lhe2e2cc_abbaabab =  1.00 * einsum('ljab,ikcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab +=  1.00 * einsum('ikab,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab += -1.00 * einsum('lkab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijad,lkcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbaabab += -1.00 * einsum('ikad,ljcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab += -1.00 * einsum('liac,kjbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abbaabab +=  1.00 * einsum('lkad,ijcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijcb,lkad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbaabab += -1.00 * einsum('kjbd,liac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbaabab += -1.00 * einsum('ikcb,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab +=  1.00 * einsum('lkcb,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijcd,lkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbaabab +=  1.00 * einsum('ikcd,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abbaabab


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
    
    lhe2e2cc_abbaabba = -1.00 * einsum('kjab,ilcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('ilab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('klab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('ijad,klcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('kjad,ilcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('kiac,ljbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ilad,kjcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abbaabba += -1.00 * einsum('ijcb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00 * einsum('ljbd,kiac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbaabba +=  1.00 * einsum('kjcb,ilad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ilcb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abbaabba +=  1.00 * einsum('ijcd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('kjcd,ilab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbaabba += -1.00 * einsum('ilcd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
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
    
    lhe2e2cc_abbabaab = -1.00 * einsum('ljab,ikdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab += -1.00 * einsum('ikab,ljdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00 * einsum('lkab,ijdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijac,lkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbabaab +=  1.00 * einsum('ikac,ljdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00 * einsum('liad,kjbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abbabaab += -1.00 * einsum('lkac,ijdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijdb,lkac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbabaab +=  1.00 * einsum('kjbc,liad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbabaab +=  1.00 * einsum('ikdb,ljac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab += -1.00 * einsum('lkdb,ijac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijdc,lkab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbabaab += -1.00 * einsum('ikdc,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abbabaab


def get_lhe2e2cc_abbababa(
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
    
    lhe2e2cc_abbababa =  1.00 * einsum('kjab,ildc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00 * einsum('ilab,kjdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa += -1.00 * einsum('klab,ijdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00 * einsum('ijac,kldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa += -1.00 * einsum('kjac,ildb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa += -1.00 * einsum('kiad,ljbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ilac,kjdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abbababa +=  1.00 * einsum('ijdb,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa += -1.00 * einsum('ljbc,kiad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbababa += -1.00 * einsum('kjdb,ilac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ildb,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_abbababa += -1.00 * einsum('ijdc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00 * einsum('kjdc,ilab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbababa +=  1.00 * einsum('ildc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abbababa


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
    
    contracted_intermediate = -1.00 * einsum('ilab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbabbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikac,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljbc,ikad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjbc,ilad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkbc,ijad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_abbabbbb +=  1.00 * einsum('ljcd,ikab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbabbbb += -1.00 * einsum('kjcd,ilab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    return lhe2e2cc_abbabbbb


def get_lhe2e2cc_abbbabaa(
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
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_abbbabaa +=  1.00 * einsum('ljad,kicb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjad,licb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbabaa += -1.00 * einsum('liad,kjcb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa +=  1.00 * einsum('lkac,ijbd->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abbbabaa += -1.00 * einsum('ijbd,klac->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbbabaa +=  1.00 * einsum('ljcb,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbabaa += -1.00 * einsum('licb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa += -1.00 * einsum('ljcd,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbabaa +=  1.00 * einsum('licd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abbbabaa


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
    
    contracted_intermediate =  1.00 * einsum('ljab,kidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liab,kjdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_abbbbaaa += -1.00 * einsum('ljac,kidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjac,lidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('liac,kjdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa += -1.00 * einsum('lkad,ijbc->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('ijbc,klad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_abbbbaaa += -1.00 * einsum('ljdb,kiac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjdb,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('lidb,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00 * einsum('ljdc,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_abbbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_abbbbaaa += -1.00 * einsum('lidc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_abbbbaaa


def get_lhe2e2cc_abbbbbab(
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
    
    lhe2e2cc_abbbbbab = -1.00 * einsum('ljab,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbab +=  1.00 * einsum('liab,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbab += -1.00 * einsum('lkab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ljac,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijbc,lkad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    lhe2e2cc_abbbbbab += -1.00 * einsum('ijcd,lkab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_abbbbbab


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
    
    lhe2e2cc_abbbbbba =  1.00 * einsum('kjab,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbba += -1.00 * einsum('kiab,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbba +=  1.00 * einsum('klab,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klac,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_abbbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_abbbbbba +=  1.00 * einsum('licd,kjab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
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
    
    lhe2e2cc_baaaaaab =  1.00 * einsum('jkba,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaab += -1.00 * einsum('ikba,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaab +=  1.00 * einsum('lkba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jkca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkca,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,lkda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,jkda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,lkba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baaaaaab +=  1.00 * einsum('licd,jkba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_baaaaaab


def get_lhe2e2cc_baaaaaba(
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
    
    lhe2e2cc_baaaaaba = -1.00 * einsum('jlba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaba +=  1.00 * einsum('ilba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaba += -1.00 * einsum('klba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jlca,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilca,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijbc,klda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,ilda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    lhe2e2cc_baaaaaba += -1.00 * einsum('ijcd,klba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjcd,ilba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baaaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_baaaaaba


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
    
    contracted_intermediate =  1.00 * einsum('jlba,ikcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilba,jkcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_baaaabbb += -1.00 * einsum('jlca,ikbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkca,ilbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ilca,jkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb += -1.00 * einsum('lkad,ijbc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ijbc,klad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baaaabbb += -1.00 * einsum('jlbd,ikca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkbd,ilca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaabbb +=  1.00 * einsum('ilbd,jkca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00 * einsum('jlcd,ikba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkcd,ilba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaaabbb += -1.00 * einsum('ilcd,jkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_baaaabbb


def get_lhe2e2cc_baaababb(
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
    
    contracted_intermediate = -1.00 * einsum('jlba,ikdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilba,jkdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_baaababb +=  1.00 * einsum('jlda,ikbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkda,ilbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaababb += -1.00 * einsum('ilda,jkbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb +=  1.00 * einsum('lkac,ijbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_baaababb += -1.00 * einsum('ijbd,klac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baaababb +=  1.00 * einsum('jlbc,ikda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jkbc,ilda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaababb += -1.00 * einsum('ilbc,jkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb += -1.00 * einsum('jldc,ikba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jkdc,ilba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baaababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_baaababb +=  1.00 * einsum('ildc,jkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_baaababb


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
    
    contracted_intermediate = -1.00 * einsum('liba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jica,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kica,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lica,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljbc,kida->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjbc,lida->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkbc,jida->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_baabaaaa +=  1.00 * einsum('ljcd,kiba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_baabaaaa += -1.00 * einsum('kjcd,liba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_baabaaaa


def get_lhe2e2cc_baababab(
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
    
    lhe2e2cc_baababab =  1.00 * einsum('jkba,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00 * einsum('liba,jkcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab += -1.00 * einsum('lkba,jicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00 * einsum('jica,lkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab += -1.00 * einsum('jkca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab += -1.00 * einsum('kiad,ljbc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('lica,jkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_baababab +=  1.00 * einsum('jibd,lkca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab += -1.00 * einsum('ljbc,kiad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baababab += -1.00 * einsum('jkbd,lica->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('libd,jkca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_baababab += -1.00 * einsum('jicd,lkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00 * einsum('jkcd,liba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababab +=  1.00 * einsum('licd,jkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_baababab


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
    
    lhe2e2cc_baababba = -1.00 * einsum('jlba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba += -1.00 * einsum('kiba,jlcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00 * einsum('klba,jicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jica,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baababba +=  1.00 * einsum('kica,jlbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00 * einsum('liad,kjbc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_baababba += -1.00 * einsum('klca,jibd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jibd,klca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baababba +=  1.00 * einsum('kjbc,liad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baababba +=  1.00 * einsum('kibd,jlca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba += -1.00 * einsum('klbd,jica->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jicd,klba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baababba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baababba += -1.00 * einsum('kicd,jlba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
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
    
    lhe2e2cc_baabbaab = -1.00 * einsum('jkba,lidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('liba,jkdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('lkba,jidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('jida,lkbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('jkda,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('kiac,ljbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('lida,jkbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_baabbaab += -1.00 * einsum('jibc,lkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00 * einsum('ljbd,kiac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baabbaab +=  1.00 * einsum('jkbc,lida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('libc,jkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_baabbaab +=  1.00 * einsum('jidc,lkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('jkdc,liba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaab += -1.00 * einsum('lidc,jkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_baabbaab


def get_lhe2e2cc_baabbaba(
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
    
    lhe2e2cc_baabbaba =  1.00 * einsum('jlba,kidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba +=  1.00 * einsum('kiba,jldc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba += -1.00 * einsum('klba,jidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jida,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baabbaba += -1.00 * einsum('kida,jlbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba += -1.00 * einsum('liac,kjbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_baabbaba +=  1.00 * einsum('klda,jibc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jibc,klda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baabbaba += -1.00 * einsum('kjbd,liac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_baabbaba += -1.00 * einsum('kibc,jlda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba +=  1.00 * einsum('klbc,jida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jidc,klba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_baabbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_baabbaba +=  1.00 * einsum('kidc,jlba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_baabbaba


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
    
    contracted_intermediate = -1.00 * einsum('jlba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_baabbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kiac,jlbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liac,jkbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jibc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_baabbbbb += -1.00 * einsum('kicd,jlba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_baabbbbb +=  1.00 * einsum('licd,jkba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
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
    
    contracted_intermediate =  1.00 * einsum('ljba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_babaaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijca,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_aaaa)
    lhe2e2cc_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kibc,ljda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('libc,kjda->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_babaaaaa +=  1.00 * einsum('kicd,ljba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    lhe2e2cc_babaaaaa += -1.00 * einsum('licd,kjba->abjicdlk', g_aaaa[oa, oa, va, va], l2_abab)
    return lhe2e2cc_babaaaaa


def get_lhe2e2cc_babaabab(
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
    
    lhe2e2cc_babaabab = -1.00 * einsum('ljba,ikcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab += -1.00 * einsum('ikba,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab +=  1.00 * einsum('lkba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijca,lkbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_babaabab +=  1.00 * einsum('kjad,libc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_babaabab +=  1.00 * einsum('ikca,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab += -1.00 * einsum('lkca,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijbd,lkca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_babaabab +=  1.00 * einsum('ikbd,ljca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab +=  1.00 * einsum('libc,kjad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_babaabab += -1.00 * einsum('lkbd,ijca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijcd,lkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_babaabab += -1.00 * einsum('ikcd,ljba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_babaabab


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
    
    lhe2e2cc_babaabba =  1.00 * einsum('kjba,ilcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00 * einsum('ilba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba += -1.00 * einsum('klba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00 * einsum('ijca,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba += -1.00 * einsum('ljad,kibc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_babaabba += -1.00 * einsum('kjca,ilbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ilca,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_babaabba +=  1.00 * einsum('ijbd,klca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba += -1.00 * einsum('kjbd,ilca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba += -1.00 * einsum('kibc,ljad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ilbd,kjca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_babaabba += -1.00 * einsum('ijcd,klba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00 * einsum('kjcd,ilba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babaabba +=  1.00 * einsum('ilcd,kjba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
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
    
    lhe2e2cc_bababaab =  1.00 * einsum('ljba,ikdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('ikba,ljdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab += -1.00 * einsum('lkba,ijdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijda,lkbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bababaab += -1.00 * einsum('kjac,libd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_bababaab += -1.00 * einsum('ikda,ljbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00 * einsum('lkda,ijbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijbc,lkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bababaab += -1.00 * einsum('ikbc,ljda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab += -1.00 * einsum('libd,kjac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_bababaab +=  1.00 * einsum('lkbc,ijda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijdc,lkba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bababaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bababaab +=  1.00 * einsum('ikdc,ljba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_bababaab


def get_lhe2e2cc_babababa(
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
    
    lhe2e2cc_babababa = -1.00 * einsum('kjba,ildc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa += -1.00 * einsum('ilba,kjdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00 * einsum('klba,ijdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa += -1.00 * einsum('ijda,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00 * einsum('ljac,kibd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_babababa +=  1.00 * einsum('kjda,ilbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ilda,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_babababa += -1.00 * einsum('ijbc,klda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00 * einsum('kjbc,ilda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00 * einsum('kibd,ljac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ilbc,kjda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_babababa +=  1.00 * einsum('ijdc,klba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa += -1.00 * einsum('kjdc,ilba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babababa += -1.00 * einsum('ildc,kjba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_babababa


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
    
    contracted_intermediate =  1.00 * einsum('ilba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bababbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljac,ikbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,ilbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkac,ijbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikbc,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilbc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_bababbbb += -1.00 * einsum('ljcd,ikba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bababbbb +=  1.00 * einsum('kjcd,ilba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    return lhe2e2cc_bababbbb


def get_lhe2e2cc_babbabaa(
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
    
    contracted_intermediate =  1.00 * einsum('ljba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_babbabaa +=  1.00 * einsum('ijad,klbc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_babbabaa += -1.00 * einsum('ljca,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbabaa +=  1.00 * einsum('lica,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa += -1.00 * einsum('ljbd,kica->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjbd,lica->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbabaa +=  1.00 * einsum('libd,kjca->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa += -1.00 * einsum('lkbc,ijad->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_babbabaa +=  1.00 * einsum('ljcd,kiba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjcd,liba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbabaa += -1.00 * einsum('licd,kjba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_babbabaa


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
    
    contracted_intermediate = -1.00 * einsum('ljba,kidc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liba,kjdc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_babbbaaa += -1.00 * einsum('ijac,klbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    lhe2e2cc_babbbaaa +=  1.00 * einsum('ljda,kibc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjda,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbaaa += -1.00 * einsum('lida,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00 * einsum('ljbc,kida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjbc,lida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbaaa += -1.00 * einsum('libc,kjda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00 * einsum('lkbd,ijac->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    lhe2e2cc_babbbaaa += -1.00 * einsum('ljdc,kiba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('kjdc,liba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_babbbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_babbbaaa +=  1.00 * einsum('lidc,kjba->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    return lhe2e2cc_babbbaaa


def get_lhe2e2cc_babbbbab(
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
    
    lhe2e2cc_babbbbab =  1.00 * einsum('ljba,kicd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbab += -1.00 * einsum('liba,kjcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbab +=  1.00 * einsum('lkba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ijac,lkbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljbc,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_babbbbab +=  1.00 * einsum('ijcd,lkba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('kjcd,liba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_babbbbab


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
    
    lhe2e2cc_babbbbba = -1.00 * einsum('kjba,licd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbba +=  1.00 * einsum('kiba,ljcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbba += -1.00 * einsum('klba,ijcd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klbc,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijcd,klba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_babbbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_babbbbba += -1.00 * einsum('licd,kjba->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    return lhe2e2cc_babbbbba


def get_lhe2e2cc_bbaaaabb(
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
    
    lhe2e2cc_bbaaaabb =  1.00 * einsum('klab,ijcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jlca,ikdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkca,ildb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilca,jkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlcb,ikda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkcb,ilda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilcb,jkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbaaaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_bbaaaabb +=  1.00 * einsum('ijcd,klab->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    return lhe2e2cc_bbaaaabb


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
    
    lhe2e2cc_bbabaaab = -1.00 * einsum('kiab,ljcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('jica,lkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkca,lidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lica,jkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jicb,lkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkcb,lida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('licb,jkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_bbabaaab += -1.00 * einsum('ljcd,kiab->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    return lhe2e2cc_bbabaaab


def get_lhe2e2cc_bbabaaba(
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
    
    lhe2e2cc_bbabaaba =  1.00 * einsum('liab,kjcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('jica,kldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kica,jldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klca,jidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jicb,klda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kicb,jlda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('klcb,jida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_bbabaaba +=  1.00 * einsum('kjcd,liab->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    return lhe2e2cc_bbabaaba


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
    
    contracted_intermediate = -1.00 * einsum('liab,jkcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbababbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_bbababbb += -1.00 * einsum('klab,jicd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('jica,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbababbb += -1.00 * einsum('jkca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb += -1.00 * einsum('kiad,jlcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('liad,jkcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jicb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbababbb +=  1.00 * einsum('jkcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00 * einsum('kibd,jlca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('libd,jkca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jicd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbababbb += -1.00 * einsum('jkcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbababbb


def get_lhe2e2cc_bbabbabb(
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
    
    contracted_intermediate =  1.00 * einsum('liab,jkdc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbabbabb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_bbabbabb +=  1.00 * einsum('klab,jidc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('jida,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbabbabb +=  1.00 * einsum('jkda,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbabbabb +=  1.00 * einsum('kiac,jldb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('liac,jkdb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jidb,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbabbabb += -1.00 * einsum('jkdb,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbabbabb += -1.00 * einsum('kibc,jlda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('libc,jkda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jidc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    lhe2e2cc_bbabbabb +=  1.00 * einsum('jkdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbabbabb


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
    
    lhe2e2cc_bbbaaaab =  1.00 * einsum('kjab,licd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('ijca,lkdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ikca,ljdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkca,ijdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijcb,lkda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ikcb,ljda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkcb,ijda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    lhe2e2cc_bbbaaaab +=  1.00 * einsum('licd,kjab->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    return lhe2e2cc_bbbaaaab


def get_lhe2e2cc_bbbaaaba(
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
    
    lhe2e2cc_bbbaaaba = -1.00 * einsum('ljab,kicd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_aaaa)
    contracted_intermediate = -1.00 * einsum('ijca,kldb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjca,ildb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ilca,kjdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcb,klda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjcb,ilda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ilcb,kjda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    lhe2e2cc_bbbaaaba += -1.00 * einsum('kicd,ljab->abjicdlk', g_aaaa[oa, oa, va, va], l2_bbbb)
    return lhe2e2cc_bbbaaaba


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
    
    contracted_intermediate =  1.00 * einsum('ljab,ikcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('klab,ijcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ijca,klbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ljad,ikcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('kjad,ilcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ikca,ljbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ilca,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('lkad,ijcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ijcb,klad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ljbd,ikca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('kjbd,ilca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ikcb,ljad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ilcb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('lkbd,ijca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ijcd,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb +=  1.00 * einsum('ikcd,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbaabbb += -1.00 * einsum('ilcd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbbaabbb


def get_lhe2e2cc_bbbababb(
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
    
    contracted_intermediate = -1.00 * einsum('ljab,ikdc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_bbbababb += -1.00 * einsum('klab,ijdc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb += -1.00 * einsum('ijda,klbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb +=  1.00 * einsum('ljac,ikdb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb += -1.00 * einsum('kjac,ildb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb += -1.00 * einsum('ikda,ljbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb +=  1.00 * einsum('ilda,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb += -1.00 * einsum('lkac,ijdb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb +=  1.00 * einsum('ijdb,klac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb += -1.00 * einsum('ljbc,ikda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb +=  1.00 * einsum('kjbc,ilda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb +=  1.00 * einsum('ikdb,ljac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb += -1.00 * einsum('ildb,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb +=  1.00 * einsum('lkbc,ijda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbababb += -1.00 * einsum('ijdc,klab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb += -1.00 * einsum('ikdc,ljab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbababb +=  1.00 * einsum('ildc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbbababb


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
    
    contracted_intermediate =  1.00 * einsum('ljca,kidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjca,lidb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lica,kjdb->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcb,kida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjcb,lida->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('licb,kjda->abjicdlk', g_abab[oa, ob, va, vb], l2_abab)
    lhe2e2cc_bbbbaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    return lhe2e2cc_bbbbaaaa


def get_lhe2e2cc_bbbbabab(
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
    
    lhe2e2cc_bbbbabab =  1.00 * einsum('kjab,licd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab += -1.00 * einsum('kiab,ljcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab +=  1.00 * einsum('ijad,lkcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab += -1.00 * einsum('ljca,kibd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('kjad,licb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lica,kjbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_bbbbabab += -1.00 * einsum('ijbd,lkca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab +=  1.00 * einsum('ljcb,kiad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kjbd,lica->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('licb,kjad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_bbbbabab += -1.00 * einsum('ljcd,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabab +=  1.00 * einsum('licd,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbbbabab


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
    
    lhe2e2cc_bbbbabba = -1.00 * einsum('ljab,kicd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabba +=  1.00 * einsum('liab,kjcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate = -1.00 * einsum('ijad,klcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjca,libd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbabba += -1.00 * einsum('liad,kjcb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabba +=  1.00 * einsum('klca,ijbd->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('ijbd,klca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjcb,liad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbabba +=  1.00 * einsum('libd,kjca->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbabba += -1.00 * einsum('klcb,ijad->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
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
    
    lhe2e2cc_bbbbbaab = -1.00 * einsum('kjab,lidc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab +=  1.00 * einsum('kiab,ljdc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab += -1.00 * einsum('ijac,lkdb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab +=  1.00 * einsum('ljda,kibc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kjac,lidb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lida,kjbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_bbbbbaab +=  1.00 * einsum('ijbc,lkda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab += -1.00 * einsum('ljdb,kiac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('kjbc,lida->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lidb,kjac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate) 
    lhe2e2cc_bbbbbaab +=  1.00 * einsum('ljdc,kiab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaab += -1.00 * einsum('lidc,kjab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    return lhe2e2cc_bbbbbaab


def get_lhe2e2cc_bbbbbaba(
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
    
    lhe2e2cc_bbbbbaba =  1.00 * einsum('ljab,kidc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaba += -1.00 * einsum('liab,kjdc->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    contracted_intermediate =  1.00 * einsum('ijac,kldb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjda,libc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbbaba +=  1.00 * einsum('liac,kjdb->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaba += -1.00 * einsum('klda,ijbc->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ijbc,klda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjdb,liac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbbaba += -1.00 * einsum('libc,kjda->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_abab)
    lhe2e2cc_bbbbbaba +=  1.00 * einsum('kldb,ijac->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('kjdc,liab->abjicdlk', g_abab[oa, ob, va, vb], l2_bbbb)
    lhe2e2cc_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return lhe2e2cc_bbbbbaba


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
    
    contracted_intermediate = -1.00 * einsum('ljab,kicd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('liab,kjcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb +=  1.00 * einsum('klab,ijcd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    contracted_intermediate = -1.00 * einsum('ijac,klbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjac,libd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('liac,kjbd->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijbc,klad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjldcik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjbc,liad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('libc,kjad->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjkcdli', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjkdcli', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijcd,klab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjlcdik', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjcd,liab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    lhe2e2cc_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    lhe2e2cc_bbbbbbbb +=  1.00 * einsum('licd,kjab->abjicdlk', g_bbbb[ob, ob, vb, vb], l2_bbbb)
    return lhe2e2cc_bbbbbbbb
