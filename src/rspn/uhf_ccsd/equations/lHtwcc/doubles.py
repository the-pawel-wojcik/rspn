from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.meta.spin_mbe import Spin_MBE, E1_spin, E2_spin


def get_lHtauwCC_doubles_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    contracted_intermediate = -1.00 * einsum('kjab,ck,ic->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,ck,kb->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,ck,ib->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jkac,ck,ib->abji', g_abab[oa, ob, va, vb], r1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljak,cl,ikbc->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlak,cl,ikbc->abji', g_abab[oa, ob, va, ob], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    lHtauwCC_doubles_aaaa +=  1.00 * einsum('ijck,cl,lkab->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ljck,cl,ikab->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jlkc,cl,ikab->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    lHtauwCC_doubles_aaaa +=  1.00 * einsum('kcab,dk,ijcd->abji', g_aaaa[oa, va, va, va], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jcad,dk,kicb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jcad,dk,ikbc->abji', g_abab[oa, vb, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kcad,dk,ijcb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ckad,dk,ijcb->abji', g_abab[va, ob, va, vb], r1_bb, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_aaaa += -1.00 * einsum('lkab,cl,dk,ijdc->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljad,cl,dk,kibc->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlad,cl,dk,ikbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljad,dk,cl,kibc->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlad,dk,cl,ikbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,cl,dk,ijbc->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,cl,dk,ijbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ck,ijbc->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klad,dl,ck,ijbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_aaaa += -1.00 * einsum('ijcd,dk,cl,klab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljcd,dk,cl,kiab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jldc,dk,cl,kiab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,ck,kiab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlcd,dl,ck,kiab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ck,ijbc->abji', f_aa[oa, va], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,ck,kiab->abji', f_aa[oa, va], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,cdlk,licd->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,cdkl,ilcd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,dckl,ildc->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    lHtauwCC_doubles_aaaa +=  0.250 * einsum('lkab,cdlk,ijcd->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ijac,cdlk,lkbd->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ijac,cdlk,lkbd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ijac,cdkl,klbd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,cdlk,libd->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,cdkl,ilbd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkac,dclk,libd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkac,cdlk,ilbd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('lkac,cdlk,ijbd->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('lkac,dclk,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('klac,dckl,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_aaaa +=  0.250 * einsum('ijcd,cdlk,lkab->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -0.50 * einsum('kjcd,cdlk,liab->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jkcd,cdlk,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jkdc,dclk,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    return lHtauwCC_doubles_aaaa


def get_lHtauwCC_doubles_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    lHtauwCC_doubles_abab =  1.00 * einsum('jkab,ck,ic->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kiab,ck,jc->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jiac,ck,kb->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jicb,ck,ka->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kjac,ck,ib->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jkac,ck,ib->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kicb,ck,ja->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kibc,ck,ja->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('ljak,cl,kicb->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jlak,cl,ikbc->abji', g_abab[oa, ob, va, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jlkb,cl,kiac->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('liak,cl,jkcb->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('likb,cl,jkac->abji', g_abab[oa, ob, oa, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('libk,cl,jkac->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jick,cl,lkab->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jikc,cl,klab->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ljck,cl,kiab->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jlkc,cl,kiab->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lick,cl,jkab->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lick,cl,jkab->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kcab,dk,jidc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ckab,dk,jicd->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jcad,dk,kicb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jcad,dk,kicb->abji', g_abab[oa, vb, va, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jcdb,dk,kiac->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ciad,dk,jkcb->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('cidb,dk,kjca->abji', g_abab[va, ob, va, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('icbd,dk,jkac->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kcad,dk,jicb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('ckad,dk,jicb->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kcdb,dk,jiac->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kcbd,dk,jiac->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('lkab,cl,dk,jicd->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('klab,cl,dk,jidc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('ljad,cl,dk,kicb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jlad,cl,dk,kibc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jldb,cl,dk,kiac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('liad,cl,dk,jkcb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('lidb,cl,dk,kjac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('libd,cl,dk,jkac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('ljad,dk,cl,kicb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jlad,dk,cl,kibc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jldb,dk,cl,kiac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('liad,dk,cl,jkcb->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('lidb,dk,cl,kjac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('libd,dk,cl,jkac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lkad,cl,dk,jicb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lkad,cl,dk,jicb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kldb,cl,dk,jiac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lkbd,cl,dk,jiac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('lkad,dl,ck,jicb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('klad,dl,ck,jicb->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lkdb,dl,ck,jiac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('lkbd,dl,ck,jiac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jidc,dk,cl,klab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jicd,dk,cl,lkab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ljcd,dk,cl,kiab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jldc,dk,cl,kiab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('licd,dk,cl,jkab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('licd,dk,cl,jkab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('ljcd,dl,ck,kiab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jlcd,dl,ck,kiab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('lidc,dl,ck,jkab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('licd,dl,ck,jkab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ka,ck,jicb->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('kb,ck,jiac->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('jc,ck,kiab->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  1.00 * einsum('ic,ck,jkab->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jkab,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jkab,dclk,lidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jkab,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('kiab,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('kiab,cdkl,jlcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('kiab,dckl,jldc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('lkab,cdlk,jicd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('klab,cdkl,jicd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('lkab,dclk,jidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('klab,dckl,jidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jiac,dclk,lkdb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jiac,dckl,kldb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jiac,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jicb,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jicb,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jicb,cdkl,klad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kjac,cdlk,lidb->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kjac,cdkl,libd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jkac,dclk,lidb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jkac,cdlk,libd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('jkcb,cdlk,liad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kiac,dckl,jldb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kicb,cdlk,ljad->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kicb,cdkl,jlad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kibc,dclk,ljad->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -1.00 * einsum('kibc,cdlk,jlad->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.50 * einsum('lkac,cdlk,jidb->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('lkac,dclk,jidb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('klac,dckl,jidb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('lkcb,cdlk,jiad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('klcb,cdkl,jiad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.50 * einsum('lkbc,cdlk,jiad->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('jicd,cdlk,lkab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('jicd,cdkl,klab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('jidc,dclk,lkab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.250 * einsum('jidc,dckl,klab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.50 * einsum('kjcd,cdlk,liab->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jkcd,cdlk,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('jkdc,dclk,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('kicd,cdkl,jlab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab +=  0.50 * einsum('kidc,dckl,jlab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abab += -0.50 * einsum('kicd,cdlk,jlab->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return lHtauwCC_doubles_abab


def get_lHtauwCC_doubles_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    lHtauwCC_doubles_abba = -1.00 * einsum('kjab,ck,ic->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ikab,ck,jc->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ijac,ck,kb->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ijcb,ck,ka->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjcb,ck,ia->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kjbc,ck,ia->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kiac,ck,jb->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ikac,ck,jb->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljak,cl,ikcb->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljkb,cl,ikac->abji', g_abab[oa, ob, oa, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljbk,cl,ikac->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('liak,cl,kjcb->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ilak,cl,jkbc->abji', g_abab[oa, ob, va, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ilkb,cl,kjac->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ijck,cl,lkab->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ijkc,cl,klab->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljck,cl,ikab->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljck,cl,ikab->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('lick,cl,kjab->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ilkc,cl,kjab->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kcab,dk,ijdc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ckab,dk,ijcd->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('cjad,dk,ikcb->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('cjdb,dk,kica->abji', g_abab[va, ob, va, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('jcbd,dk,ikac->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('icad,dk,kjcb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('icad,dk,kjcb->abji', g_abab[oa, vb, va, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('icdb,dk,kjac->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kcad,dk,ijcb->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ckad,dk,ijcb->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kcdb,dk,ijac->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kcbd,dk,ijac->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('lkab,cl,dk,ijcd->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('klab,cl,dk,ijdc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljad,cl,dk,ikcb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljdb,cl,dk,kiac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljbd,cl,dk,ikac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('liad,cl,dk,kjcb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ilad,cl,dk,kjbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ildb,cl,dk,kjac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljad,dk,cl,ikcb->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljdb,dk,cl,kiac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljbd,dk,cl,ikac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('liad,dk,cl,kjcb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ilad,dk,cl,kjbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ildb,dk,cl,kjac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('lkad,cl,dk,ijcb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('lkad,cl,dk,ijcb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kldb,cl,dk,ijac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('lkbd,cl,dk,ijac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('lkad,dl,ck,ijcb->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('klad,dl,ck,ijcb->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('lkdb,dl,ck,ijac->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('lkbd,dl,ck,ijac->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ijdc,dk,cl,klab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ijcd,dk,cl,lkab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljcd,dk,cl,ikab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljcd,dk,cl,ikab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('licd,dk,cl,kjab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ildc,dk,cl,kjab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ljdc,dl,ck,ikab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ljcd,dl,ck,ikab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('licd,dl,ck,kjab->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ilcd,dl,ck,kjab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ka,ck,ijcb->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('kb,ck,ijac->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('jc,ck,ikab->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -1.00 * einsum('ic,ck,kjab->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('kjab,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('kjab,cdkl,ilcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('kjab,dckl,ildc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ikab,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ikab,dclk,ljdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ikab,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('lkab,cdlk,ijcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('klab,cdkl,ijcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('lkab,dclk,ijdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('klab,dckl,ijdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijac,dclk,lkdb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijac,dckl,kldb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijac,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijcb,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijcb,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ijcb,cdkl,klad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjac,dckl,ildb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjcb,cdlk,liad->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjcb,cdkl,ilad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjbc,dclk,liad->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kjbc,cdlk,ilad->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kiac,cdlk,ljdb->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('kiac,cdkl,ljbd->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ikac,dclk,ljdb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ikac,cdlk,ljbd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  1.00 * einsum('ikcb,cdlk,ljad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_abba +=  0.50 * einsum('lkac,cdlk,ijdb->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('lkac,dclk,ijdb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('klac,dckl,ijdb->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('lkcb,cdlk,ijad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('klcb,cdkl,ijad->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.50 * einsum('lkbc,cdlk,ijad->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('ijcd,cdlk,lkab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('ijcd,cdkl,klab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('ijdc,dclk,lkab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.250 * einsum('ijdc,dckl,klab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('kjcd,cdkl,ilab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('kjdc,dckl,ilab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.50 * einsum('kjcd,cdlk,ilab->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba +=  0.50 * einsum('kicd,cdlk,ljab->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ikcd,cdlk,ljab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_abba += -0.50 * einsum('ikdc,dclk,ljab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return lHtauwCC_doubles_abba


def get_lHtauwCC_doubles_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    lHtauwCC_doubles_baab = -1.00 * einsum('jkba,ck,ic->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kiba,ck,jc->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jica,ck,kb->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jibc,ck,ka->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kjbc,ck,ia->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jkbc,ck,ia->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kica,ck,jb->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kiac,ck,jb->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jlka,cl,kibc->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('ljbk,cl,kica->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jlbk,cl,ikac->abji', g_abab[oa, ob, va, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lika,cl,jkbc->abji', g_abab[oa, ob, oa, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('liak,cl,jkbc->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('libk,cl,jkca->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jick,cl,lkba->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jikc,cl,klba->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('ljck,cl,kiba->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jlkc,cl,kiba->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lick,cl,jkba->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lick,cl,jkba->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kcba,dk,jidc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('ckba,dk,jicd->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jcda,dk,kibc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jcbd,dk,kica->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jcbd,dk,kica->abji', g_abab[oa, vb, va, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('cida,dk,kjcb->abji', g_abab[va, ob, va, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('icad,dk,jkbc->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('cibd,dk,jkca->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kcda,dk,jibc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kcad,dk,jibc->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kcbd,dk,jica->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('ckbd,dk,jica->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('lkba,cl,dk,jicd->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('klba,cl,dk,jidc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jlda,cl,dk,kibc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('ljbd,cl,dk,kica->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jlbd,cl,dk,kiac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('lida,cl,dk,kjbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('liad,cl,dk,jkbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('libd,cl,dk,jkca->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jlda,dk,cl,kibc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('ljbd,dk,cl,kica->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jlbd,dk,cl,kiac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('lida,dk,cl,kjbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('liad,dk,cl,jkbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('libd,dk,cl,jkca->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('klda,cl,dk,jibc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lkad,cl,dk,jibc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lkbd,cl,dk,jica->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lkbd,cl,dk,jica->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lkda,dl,ck,jibc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('lkad,dl,ck,jibc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('lkbd,dl,ck,jica->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('klbd,dl,ck,jica->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jidc,dk,cl,klba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jicd,dk,cl,lkba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('ljcd,dk,cl,kiba->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jldc,dk,cl,kiba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('licd,dk,cl,jkba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('licd,dk,cl,jkba->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('ljcd,dl,ck,kiba->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jlcd,dl,ck,kiba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('lidc,dl,ck,jkba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('licd,dl,ck,jkba->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('ka,ck,jibc->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('kb,ck,jica->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('jc,ck,kiba->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -1.00 * einsum('ic,ck,jkba->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jkba,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jkba,dclk,lidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jkba,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('kiba,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('kiba,cdkl,jlcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('kiba,dckl,jldc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('lkba,cdlk,jicd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('klba,cdkl,jicd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('lkba,dclk,jidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('klba,dckl,jidc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jica,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jica,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jica,cdkl,klbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jibc,dclk,lkda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jibc,dckl,klda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jibc,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jkca,cdlk,libd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kjbc,cdlk,lida->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kjbc,cdkl,liad->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jkbc,dclk,lida->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('jkbc,cdlk,liad->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kica,cdlk,ljbd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kica,cdkl,jlbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kiac,dclk,ljbd->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kiac,cdlk,jlbd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  1.00 * einsum('kibc,dckl,jlda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('lkca,cdlk,jibd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('klca,cdkl,jibd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.50 * einsum('lkac,cdlk,jibd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.50 * einsum('lkbc,cdlk,jida->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('lkbc,dclk,jida->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('klbc,dckl,jida->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('jicd,cdlk,lkba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('jicd,cdkl,klba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('jidc,dclk,lkba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.250 * einsum('jidc,dckl,klba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.50 * einsum('kjcd,cdlk,liba->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jkcd,cdlk,liba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('jkdc,dclk,liba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('kicd,cdkl,jlba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab += -0.50 * einsum('kidc,dckl,jlba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baab +=  0.50 * einsum('kicd,cdlk,jlba->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return lHtauwCC_doubles_baab


def get_lHtauwCC_doubles_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    lHtauwCC_doubles_baba =  1.00 * einsum('kjba,ck,ic->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ikba,ck,jc->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ijca,ck,kb->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ijbc,ck,ka->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjca,ck,ib->abji', g_abab[oa, ob, va, vb], r1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kjac,ck,ib->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kibc,ck,ja->abji', g_aaaa[oa, oa, va, va], r1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ikbc,ck,ja->abji', g_abab[oa, ob, va, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljka,cl,ikbc->abji', g_abab[oa, ob, oa, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljak,cl,ikbc->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljbk,cl,ikca->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ilka,cl,kjbc->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('libk,cl,kjca->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ilbk,cl,jkac->abji', g_abab[oa, ob, va, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ijck,cl,lkba->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ijkc,cl,klba->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljck,cl,ikba->abji', g_abab[oa, ob, va, ob], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljck,cl,ikba->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('lick,cl,kjba->abji', g_aaaa[oa, oa, va, oa], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ilkc,cl,kjba->abji', g_abab[oa, ob, oa, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kcba,dk,ijdc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ckba,dk,ijcd->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('cjda,dk,kicb->abji', g_abab[va, ob, va, vb], r1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('jcad,dk,ikbc->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('cjbd,dk,ikca->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('icda,dk,kjbc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('icbd,dk,kjca->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('icbd,dk,kjca->abji', g_abab[oa, vb, va, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kcda,dk,ijbc->abji', g_abab[oa, vb, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kcad,dk,ijbc->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kcbd,dk,ijca->abji', g_aaaa[oa, va, va, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ckbd,dk,ijca->abji', g_abab[va, ob, va, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('lkba,cl,dk,ijcd->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('klba,cl,dk,ijdc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljda,cl,dk,kibc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljad,cl,dk,ikbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljbd,cl,dk,ikca->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ilda,cl,dk,kjbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('libd,cl,dk,kjca->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ilbd,cl,dk,kjac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljda,dk,cl,kibc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljad,dk,cl,ikbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljbd,dk,cl,ikca->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ilda,dk,cl,kjbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('libd,dk,cl,kjca->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ilbd,dk,cl,kjac->abji', g_abab[oa, ob, va, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('klda,cl,dk,ijbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('lkad,cl,dk,ijbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('lkbd,cl,dk,ijca->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('lkbd,cl,dk,ijca->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('lkda,dl,ck,ijbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('lkad,dl,ck,ijbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('lkbd,dl,ck,ijca->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('klbd,dl,ck,ijca->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ijdc,dk,cl,klba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ijcd,dk,cl,lkba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljcd,dk,cl,ikba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljcd,dk,cl,ikba->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('licd,dk,cl,kjba->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ildc,dk,cl,kjba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ljdc,dl,ck,ikba->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ljcd,dl,ck,ikba->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('licd,dl,ck,kjba->abji', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ilcd,dl,ck,kjba->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ka,ck,ijbc->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('kb,ck,ijca->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('jc,ck,ikba->abji', f_bb[ob, vb], r1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  1.00 * einsum('ic,ck,kjba->abji', f_aa[oa, va], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('kjba,cdlk,licd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('kjba,cdkl,ilcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('kjba,dckl,ildc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ikba,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ikba,dclk,ljdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ikba,cdlk,ljcd->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('lkba,cdlk,ijcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('klba,cdkl,ijcd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('lkba,dclk,ijdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('klba,dckl,ijdc->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijca,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijca,cdlk,lkbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijca,cdkl,klbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijbc,dclk,lkda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijbc,dckl,klda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ijbc,cdlk,lkad->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjca,cdlk,libd->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjca,cdkl,ilbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjac,dclk,libd->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjac,cdlk,ilbd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kjbc,dckl,ilda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ikca,cdlk,ljbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kibc,cdlk,ljda->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('kibc,cdkl,ljad->abji', g_aaaa[oa, oa, va, va], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ikbc,dclk,ljda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -1.00 * einsum('ikbc,cdlk,ljad->abji', g_abab[oa, ob, va, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('lkca,cdlk,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('klca,cdkl,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.50 * einsum('lkac,cdlk,ijbd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.50 * einsum('lkbc,cdlk,ijda->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('lkbc,dclk,ijda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('klbc,dckl,ijda->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('ijcd,cdlk,lkba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('ijcd,cdkl,klba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('ijdc,dclk,lkba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.250 * einsum('ijdc,dckl,klba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('kjcd,cdkl,ilba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('kjdc,dckl,ilba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.50 * einsum('kjcd,cdlk,ilba->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba += -0.50 * einsum('kicd,cdlk,ljba->abji', g_aaaa[oa, oa, va, va], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ikcd,cdlk,ljba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_baba +=  0.50 * einsum('ikdc,dclk,ljba->abji', g_abab[oa, ob, va, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return lHtauwCC_doubles_baba


def get_lHtauwCC_doubles_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
    r2_abba = vector.doubles[E2_spin.abba]
    r2_baab = vector.doubles[E2_spin.baab]
    r2_baba = vector.doubles[E2_spin.baba]
    r2_bbbb = vector.doubles[E2_spin.bbbb]
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
    
    contracted_intermediate = -1.00 * einsum('kjab,ck,ic->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ijac,ck,kb->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kjca,ck,ib->abji', g_abab[oa, ob, va, vb], r1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,ck,ib->abji', g_bbbb[ob, ob, vb, vb], r1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljka,cl,kicb->abji', g_abab[oa, ob, oa, vb], r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljak,cl,ikbc->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    lHtauwCC_doubles_bbbb +=  1.00 * einsum('ijck,cl,lkab->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ljck,cl,ikab->abji', g_abab[oa, ob, va, ob], r1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljck,cl,ikab->abji', g_bbbb[ob, ob, vb, ob], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    lHtauwCC_doubles_bbbb +=  1.00 * einsum('kcab,dk,ijcd->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('cjda,dk,kicb->abji', g_abab[va, ob, va, vb], r1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jcad,dk,kicb->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kcda,dk,ijcb->abji', g_abab[oa, vb, va, vb], r1_aa, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kcad,dk,ijcb->abji', g_bbbb[ob, vb, vb, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_bbbb += -1.00 * einsum('lkab,cl,dk,ijdc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljda,cl,dk,kicb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljad,cl,dk,kibc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljda,dk,cl,kicb->abji', g_abab[oa, ob, va, vb], t1_aa, r1_aa, l2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljad,dk,cl,kibc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('klda,cl,dk,ijbc->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkad,cl,dk,ijbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,dl,ck,ijbc->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkad,dl,ck,ijbc->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_bbbb += -1.00 * einsum('ijcd,dk,cl,klab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ljcd,dk,cl,kiab->abji', g_abab[oa, ob, va, vb], t1_bb, r1_aa, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljcd,dk,cl,kiab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ljdc,dl,ck,kiab->abji', g_abab[oa, ob, va, vb], t1_aa, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ljcd,dl,ck,kiab->abji', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ck,ijbc->abji', f_bb[ob, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,ck,kiab->abji', f_bb[ob, vb], r1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,cdlk,licd->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,dclk,lidc->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjab,cdlk,licd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    lHtauwCC_doubles_bbbb +=  0.250 * einsum('lkab,cdlk,ijcd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ijac,dclk,lkdb->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ijac,dckl,kldb->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ijac,cdlk,lkbd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjca,cdlk,lidb->abji', g_abab[oa, ob, va, vb], r2_aaaa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjca,cdkl,libd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,dclk,lidb->abji', g_bbbb[ob, ob, vb, vb], r2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kjac,cdlk,libd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('lkca,cdlk,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('klca,cdkl,ijbd->abji', g_abab[oa, ob, va, vb], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('lkac,cdlk,ijbd->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    lHtauwCC_doubles_bbbb +=  0.250 * einsum('ijcd,cdlk,lkab->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('kjcd,cdkl,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('kjdc,dckl,liab->abji', g_abab[oa, ob, va, vb], r2_abab, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('kjcd,cdlk,liab->abji', g_bbbb[ob, ob, vb, vb], r2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    lHtauwCC_doubles_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    return lHtauwCC_doubles_bbbb
