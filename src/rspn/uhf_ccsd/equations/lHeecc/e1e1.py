from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_lhe1e1cc_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    lhe1e1cc_aaaa = -1.00 * einsum('jiab->aibj', g_aaaa[oa, oa, va, va])
    contracted_intermediate =  1.00 * einsum('jiak,kb->aibj', g_aaaa[oa, oa, va, oa], l1_aa)
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('icab,jc->aibj', g_aaaa[oa, va, va, va], l1_aa)
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    lhe1e1cc_aaaa += -0.50 * einsum('jikl,klab->aibj', g_aaaa[oa, oa, oa, oa], l2_aaaa)
    lhe1e1cc_aaaa +=  1.00 * einsum('jcak,ikcb->aibj', g_aaaa[oa, va, va, oa], l2_aaaa)
    lhe1e1cc_aaaa += -1.00 * einsum('jcak,ikbc->aibj', g_abab[oa, vb, va, ob], l2_abab)
    lhe1e1cc_aaaa +=  1.00 * einsum('icbk,jkca->aibj', g_aaaa[oa, va, va, oa], l2_aaaa)
    lhe1e1cc_aaaa += -1.00 * einsum('icbk,jkac->aibj', g_abab[oa, vb, va, ob], l2_abab)
    lhe1e1cc_aaaa += -0.50 * einsum('dcab,jidc->aibj', g_aaaa[va, va, va, va], l2_aaaa)
    contracted_intermediate =  1.00 * einsum('kiab,ck,jc->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,ck,kb->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_aaaa +=  1.00 * einsum('kjac,ck,ib->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('jkac,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('kibc,ck,ja->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('ikbc,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('ljak,cl,ikbc->aibj', g_aaaa[oa, oa, va, oa], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('jlak,cl,ikbc->aibj', g_abab[oa, ob, va, ob], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('libk,cl,jkac->aibj', g_aaaa[oa, oa, va, oa], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ilbk,cl,jkac->aibj', g_abab[oa, ob, va, ob], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('jicl,ck,klab->aibj', g_aaaa[oa, oa, va, oa], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('kdab,ck,jidc->aibj', g_aaaa[oa, va, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('jcad,dk,kicb->aibj', g_aaaa[oa, va, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('jcad,dk,ikbc->aibj', g_abab[oa, vb, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('icbd,dk,kjca->aibj', g_aaaa[oa, va, va, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('icbd,dk,jkac->aibj', g_abab[oa, vb, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('liab,dckl,kjdc->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('liab,dclk,jkdc->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('liab,cdlk,jkcd->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,dckl,klbc->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,dckl,klbc->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,dclk,lkbc->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_aaaa += -0.250 * einsum('lkab,dclk,jidc->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ljad,dckl,kibc->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ljad,dclk,ikbc->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('jlad,cdkl,kibc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('jlad,dckl,ikbc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('libd,dckl,kjac->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('libd,dclk,jkac->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ilbd,cdkl,kjac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ilbd,dckl,jkac->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -0.250 * einsum('jicd,cdkl,klab->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  0.50 * einsum('lkab,cl,dk,jidc->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ljad,cl,dk,kibc->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('jlad,cl,dk,ikbc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('libd,cl,dk,kjac->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('ilbd,cl,dk,jkac->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  0.50 * einsum('jicd,cl,dk,klab->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('ja,ib->aibj', f_aa[oa, va], l1_aa)
    lhe1e1cc_aaaa += -1.00 * einsum('ib,ja->aibj', f_aa[oa, va], l1_aa)
    return lhe1e1cc_aaaa


def get_lhe1e1cc_aabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    lhe1e1cc_aabb =  1.00 * einsum('ijab->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_aabb += -1.00 * einsum('ijak,kb->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_aabb += -1.00 * einsum('ijkb,ka->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_aabb +=  1.00 * einsum('icab,jc->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_aabb +=  1.00 * einsum('cjab,ic->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_aabb +=  0.50 * einsum('ijkl,klab->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_aabb +=  0.50 * einsum('ijlk,lkab->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_aabb += -1.00 * einsum('cjak,ikcb->aibj', g_abab[va, ob, va, ob], l2_abab)
    lhe1e1cc_aabb += -1.00 * einsum('ickb,kjac->aibj', g_abab[oa, vb, oa, vb], l2_abab)
    lhe1e1cc_aabb +=  0.50 * einsum('dcab,ijdc->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_aabb +=  0.50 * einsum('cdab,ijcd->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_aabb += -1.00 * einsum('ikab,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('kjab,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('ijac,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('ijcb,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ljak,cl,ikcb->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ilkb,cl,kjac->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ijcl,ck,klab->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ijlc,ck,lkab->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('kdab,ck,ijcd->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('dkab,ck,ijdc->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('cjad,dk,ikcb->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('icdb,dk,kjac->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ilab,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ilab,cdkl,kjcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ilab,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ljab,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ljab,dclk,ikdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ljab,cdlk,ikcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijad,cdkl,klcb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijad,cdlk,lkcb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijad,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijdb,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijdb,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -0.50 * einsum('ijdb,dclk,lkac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('lkab,dclk,ijdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('klab,dckl,ijdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('lkab,cdlk,ijcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('klab,cdkl,ijcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ljad,cdlk,ikcb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ildb,dckl,kjac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('ijcd,cdkl,klab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('ijcd,cdlk,lkab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('ijdc,dckl,klab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.250 * einsum('ijdc,dclk,lkab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.50 * einsum('lkab,cl,dk,ijcd->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.50 * einsum('klab,cl,dk,ijdc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ljad,cl,dk,ikcb->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lhe1e1cc_aabb +=  1.00 * einsum('ildb,cl,dk,kjac->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lhe1e1cc_aabb +=  0.50 * einsum('ijcd,cl,dk,lkab->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_aabb +=  0.50 * einsum('ijdc,cl,dk,klab->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return lhe1e1cc_aabb


def get_lhe1e1cc_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    lhe1e1cc_abba = -1.00 * einsum('jiab->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_abba +=  1.00 * einsum('jiak,kb->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_abba +=  1.00 * einsum('jikb,ka->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_abba += -1.00 * einsum('ciab,jc->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_abba += -1.00 * einsum('jcab,ic->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_abba += -0.50 * einsum('jikl,klab->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_abba += -0.50 * einsum('jilk,lkab->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_abba += -1.00 * einsum('jcak,kicb->aibj', g_aaaa[oa, va, va, oa], l2_abab)
    lhe1e1cc_abba +=  1.00 * einsum('jcak,ikcb->aibj', g_abab[oa, vb, va, ob], l2_bbbb)
    lhe1e1cc_abba +=  1.00 * einsum('cikb,jkca->aibj', g_abab[va, ob, oa, vb], l2_aaaa)
    lhe1e1cc_abba += -1.00 * einsum('icbk,jkac->aibj', g_bbbb[ob, vb, vb, ob], l2_abab)
    lhe1e1cc_abba += -0.50 * einsum('dcab,jidc->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_abba += -0.50 * einsum('cdab,jicd->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_abba +=  1.00 * einsum('kiab,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jkab,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jiac,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jicb,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('kjac,ck,ib->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jkac,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('kicb,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('kibc,ck,ja->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ljak,cl,kicb->aibj', g_aaaa[oa, oa, va, oa], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jlak,cl,ikbc->aibj', g_abab[oa, ob, va, ob], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('likb,cl,jkac->aibj', g_abab[oa, ob, oa, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('libk,cl,jkac->aibj', g_bbbb[ob, ob, vb, ob], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jicl,ck,klab->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jilc,ck,lkab->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('kdab,ck,jicd->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('dkab,ck,jidc->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jcad,dk,kicb->aibj', g_aaaa[oa, va, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jcad,dk,kicb->aibj', g_abab[oa, vb, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('cidb,dk,kjca->aibj', g_abab[va, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('icbd,dk,jkac->aibj', g_bbbb[ob, vb, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('liab,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('liab,dclk,jkdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('liab,cdlk,jkcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jlab,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jlab,cdkl,kicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jlab,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jiad,cdkl,klcb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jiad,cdlk,lkcb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jiad,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jidb,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jidb,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  0.50 * einsum('jidb,dclk,lkac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('lkab,dclk,jidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('klab,dckl,jidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('lkab,cdlk,jicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('klab,cdkl,jicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ljad,dckl,kicb->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ljad,dclk,kibc->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jlad,cdkl,kicb->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jlad,dckl,kibc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('lidb,dckl,kjac->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('lidb,dclk,jkac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('libd,cdkl,kjac->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('libd,dckl,jkac->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('jicd,cdkl,klab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('jicd,cdlk,lkab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('jidc,dckl,klab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.250 * einsum('jidc,dclk,lkab->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.50 * einsum('lkab,cl,dk,jicd->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.50 * einsum('klab,cl,dk,jidc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ljad,cl,dk,kicb->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jlad,cl,dk,kibc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('lidb,cl,dk,kjac->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('libd,cl,dk,jkac->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.50 * einsum('jicd,cl,dk,lkab->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_abba += -0.50 * einsum('jidc,cl,dk,klab->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ja,ib->aibj', f_aa[oa, va], l1_bb)
    lhe1e1cc_abba += -1.00 * einsum('ib,ja->aibj', f_bb[ob, vb], l1_aa)
    return lhe1e1cc_abba


def get_lhe1e1cc_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    lhe1e1cc_baab = -1.00 * einsum('ijba->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_baab +=  1.00 * einsum('ijka,kb->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_baab +=  1.00 * einsum('ijbk,ka->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_baab += -1.00 * einsum('icba,jc->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_baab += -1.00 * einsum('cjba,ic->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_baab += -0.50 * einsum('ijkl,klba->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_baab += -0.50 * einsum('ijlk,lkba->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_baab +=  1.00 * einsum('cjka,ikcb->aibj', g_abab[va, ob, oa, vb], l2_aaaa)
    lhe1e1cc_baab += -1.00 * einsum('jcak,ikbc->aibj', g_bbbb[ob, vb, vb, ob], l2_abab)
    lhe1e1cc_baab += -1.00 * einsum('icbk,kjca->aibj', g_aaaa[oa, va, va, oa], l2_abab)
    lhe1e1cc_baab +=  1.00 * einsum('icbk,jkca->aibj', g_abab[oa, vb, va, ob], l2_bbbb)
    lhe1e1cc_baab += -0.50 * einsum('dcba,ijdc->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_baab += -0.50 * einsum('cdba,ijcd->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_baab +=  1.00 * einsum('ikba,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kjba,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ijca,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ijbc,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('kjca,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kjac,ck,ib->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kibc,ck,ja->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ikbc,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ljka,cl,ikbc->aibj', g_abab[oa, ob, oa, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljak,cl,ikbc->aibj', g_bbbb[ob, ob, vb, ob], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('libk,cl,kjca->aibj', g_aaaa[oa, oa, va, oa], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ilbk,cl,jkac->aibj', g_abab[oa, ob, va, ob], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ijcl,ck,klba->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ijlc,ck,lkba->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kdba,ck,ijcd->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('dkba,ck,ijdc->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('cjda,dk,kicb->aibj', g_abab[va, ob, va, vb], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('jcad,dk,ikbc->aibj', g_bbbb[ob, vb, vb, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('icbd,dk,kjca->aibj', g_aaaa[oa, va, va, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('icbd,dk,kjca->aibj', g_abab[oa, vb, va, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ilba,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ilba,cdkl,kjcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ilba,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ljba,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ljba,dclk,ikdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ljba,cdlk,ikcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijda,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijda,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijda,dclk,lkbc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijbd,cdkl,klca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijbd,cdlk,lkca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  0.50 * einsum('ijbd,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('lkba,dclk,ijdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('klba,dckl,ijdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('lkba,cdlk,ijcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('klba,cdkl,ijcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljda,dckl,kibc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljda,dclk,ikbc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljad,cdkl,kibc->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljad,dckl,ikbc->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('libd,dckl,kjca->aibj', g_aaaa[oa, oa, va, va], t2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('libd,dclk,kjac->aibj', g_aaaa[oa, oa, va, va], t2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ilbd,cdkl,kjca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ilbd,dckl,kjac->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('ijcd,cdkl,klba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('ijcd,cdlk,lkba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('ijdc,dckl,klba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.250 * einsum('ijdc,dclk,lkba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.50 * einsum('lkba,cl,dk,ijcd->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.50 * einsum('klba,cl,dk,ijdc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljda,cl,dk,kibc->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ljad,cl,dk,ikbc->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('libd,cl,dk,kjca->aibj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ilbd,cl,dk,kjac->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.50 * einsum('ijcd,cl,dk,lkba->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_baab += -0.50 * einsum('ijdc,cl,dk,klba->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ja,ib->aibj', f_bb[ob, vb], l1_aa)
    lhe1e1cc_baab += -1.00 * einsum('ib,ja->aibj', f_aa[oa, va], l1_bb)
    return lhe1e1cc_baab


def get_lhe1e1cc_bbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    g_abab = uhf_scf_data.g_abab
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
    
    lhe1e1cc_bbaa =  1.00 * einsum('jiba->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_bbaa += -1.00 * einsum('jika,kb->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_bbaa += -1.00 * einsum('jibk,ka->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_bbaa +=  1.00 * einsum('ciba,jc->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_bbaa +=  1.00 * einsum('jcba,ic->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_bbaa +=  0.50 * einsum('jikl,klba->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_bbaa +=  0.50 * einsum('jilk,lkba->aibj', g_abab[oa, ob, oa, ob], l2_abab)
    lhe1e1cc_bbaa += -1.00 * einsum('jcka,kibc->aibj', g_abab[oa, vb, oa, vb], l2_abab)
    lhe1e1cc_bbaa += -1.00 * einsum('cibk,jkca->aibj', g_abab[va, ob, va, ob], l2_abab)
    lhe1e1cc_bbaa +=  0.50 * einsum('dcba,jidc->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_bbaa +=  0.50 * einsum('cdba,jicd->aibj', g_abab[va, vb, va, vb], l2_abab)
    lhe1e1cc_bbaa += -1.00 * einsum('kiba,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jkba,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jica,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jibc,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('jlka,cl,kibc->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('libk,cl,jkca->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('jicl,ck,klba->aibj', g_abab[oa, ob, va, ob], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('jilc,ck,lkba->aibj', g_abab[oa, ob, oa, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('kdba,ck,jicd->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('dkba,ck,jidc->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jcda,dk,kibc->aibj', g_abab[oa, vb, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('cibd,dk,jkca->aibj', g_abab[va, ob, va, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('liba,dckl,kjdc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('liba,dclk,jkdc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('liba,cdlk,jkcd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jlba,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jlba,cdkl,kicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jlba,dckl,kidc->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jida,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jida,dckl,klbc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jida,dclk,lkbc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jibd,cdkl,klca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jibd,cdlk,lkca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -0.50 * einsum('jibd,dckl,klac->aibj', g_abab[oa, ob, va, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('lkba,dclk,jidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('klba,dckl,jidc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('lkba,cdlk,jicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('klba,cdkl,jicd->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('jlda,dckl,kibc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('libd,cdlk,jkca->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('jicd,cdkl,klba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('jicd,cdlk,lkba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('jidc,dckl,klba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.250 * einsum('jidc,dclk,lkba->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.50 * einsum('lkba,cl,dk,jicd->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.50 * einsum('klba,cl,dk,jidc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('jlda,cl,dk,kibc->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lhe1e1cc_bbaa +=  1.00 * einsum('libd,cl,dk,jkca->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lhe1e1cc_bbaa +=  0.50 * einsum('jicd,cl,dk,lkba->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_bbaa +=  0.50 * einsum('jidc,cl,dk,klba->aibj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return lhe1e1cc_bbaa


def get_lhe1e1cc_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    """ The matrices h_aa and h_bb should be the matrix elements of the operator
    in question, e.g. mu_x_a and mu_x_b. """
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
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
    
    lhe1e1cc_bbbb = -1.00 * einsum('jiab->aibj', g_bbbb[ob, ob, vb, vb])
    contracted_intermediate =  1.00 * einsum('jiak,kb->aibj', g_bbbb[ob, ob, vb, ob], l1_bb)
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('icab,jc->aibj', g_bbbb[ob, vb, vb, vb], l1_bb)
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    lhe1e1cc_bbbb += -0.50 * einsum('jikl,klab->aibj', g_bbbb[ob, ob, ob, ob], l2_bbbb)
    lhe1e1cc_bbbb += -1.00 * einsum('cjka,kicb->aibj', g_abab[va, ob, oa, vb], l2_abab)
    lhe1e1cc_bbbb +=  1.00 * einsum('jcak,ikcb->aibj', g_bbbb[ob, vb, vb, ob], l2_bbbb)
    lhe1e1cc_bbbb += -1.00 * einsum('cikb,kjca->aibj', g_abab[va, ob, oa, vb], l2_abab)
    lhe1e1cc_bbbb +=  1.00 * einsum('icbk,jkca->aibj', g_bbbb[ob, vb, vb, ob], l2_bbbb)
    lhe1e1cc_bbbb += -0.50 * einsum('dcab,jidc->aibj', g_bbbb[vb, vb, vb, vb], l2_bbbb)
    contracted_intermediate =  1.00 * einsum('kiab,ck,jc->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,ck,kb->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_bbbb += -1.00 * einsum('kjca,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('kjac,ck,ib->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('kicb,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('kibc,ck,ja->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljka,cl,kicb->aibj', g_abab[oa, ob, oa, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('ljak,cl,ikbc->aibj', g_bbbb[ob, ob, vb, ob], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('likb,cl,kjca->aibj', g_abab[oa, ob, oa, vb], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('libk,cl,jkac->aibj', g_bbbb[ob, ob, vb, ob], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('jicl,ck,klab->aibj', g_bbbb[ob, ob, vb, ob], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('kdab,ck,jidc->aibj', g_bbbb[ob, vb, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('cjda,dk,kicb->aibj', g_abab[va, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('jcad,dk,kicb->aibj', g_bbbb[ob, vb, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('cidb,dk,kjca->aibj', g_abab[va, ob, va, vb], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('icbd,dk,kjca->aibj', g_bbbb[ob, vb, vb, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('liab,dckl,kjdc->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('liab,cdkl,kjcd->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('liab,dckl,kjdc->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,cdkl,klcb->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,cdlk,lkcb->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,dckl,klbc->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_bbbb += -0.250 * einsum('lkab,dclk,jidc->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljda,dckl,kicb->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljda,dclk,kibc->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljad,cdkl,kicb->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljad,dckl,kibc->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('lidb,dckl,kjca->aibj', g_abab[oa, ob, va, vb], t2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('lidb,dclk,kjac->aibj', g_abab[oa, ob, va, vb], t2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('libd,cdkl,kjca->aibj', g_bbbb[ob, ob, vb, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('libd,dckl,kjac->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -0.250 * einsum('jicd,cdkl,klab->aibj', g_bbbb[ob, ob, vb, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  0.50 * einsum('lkab,cl,dk,jidc->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljda,cl,dk,kicb->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('ljad,cl,dk,kibc->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('lidb,cl,dk,kjca->aibj', g_abab[oa, ob, va, vb], t1_aa, t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('libd,cl,dk,kjac->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  0.50 * einsum('jicd,cl,dk,klab->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('ja,ib->aibj', f_bb[ob, vb], l1_bb)
    lhe1e1cc_bbbb += -1.00 * einsum('ib,ja->aibj', f_bb[ob, vb], l1_bb)
    return lhe1e1cc_bbbb
