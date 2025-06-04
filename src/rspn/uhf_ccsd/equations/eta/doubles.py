from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_eta_aaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    contracted_intermediate = -1.00 * einsum('ja,ib->abji', h_aa[oa, va], l1_aa)
    eta_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,ikab->abji', h_aa[oa, oa], l2_aaaa)
    eta_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ca,ijcb->abji', h_aa[va, va], l2_aaaa)
    eta_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ck,ijbc->abji', h_aa[oa, va], t1_aa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,ck,kiab->abji', h_aa[oa, va], t1_aa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    return eta_aaaa


def get_eta_abab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_abab = -1.00 * einsum('ja,ib->abji', h_aa[oa, va], l1_bb)
    eta_abab += -1.00 * einsum('ib,ja->abji', h_bb[ob, vb], l1_aa)
    eta_abab +=  1.00 * einsum('jk,kiab->abji', h_aa[oa, oa], l2_abab)
    eta_abab +=  1.00 * einsum('ik,jkab->abji', h_bb[ob, ob], l2_abab)
    eta_abab += -1.00 * einsum('ca,jicb->abji', h_aa[va, va], l2_abab)
    eta_abab += -1.00 * einsum('cb,jiac->abji', h_bb[vb, vb], l2_abab)
    eta_abab +=  1.00 * einsum('ka,ck,jicb->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_abab +=  1.00 * einsum('kb,ck,jiac->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_abab +=  1.00 * einsum('jc,ck,kiab->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_abab +=  1.00 * einsum('ic,ck,jkab->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return eta_abab


def get_eta_abba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_abba =  1.00 * einsum('jb,ia->abji', h_bb[ob, vb], l1_aa)
    eta_abba +=  1.00 * einsum('ia,jb->abji', h_aa[oa, va], l1_bb)
    eta_abba += -1.00 * einsum('jk,ikab->abji', h_bb[ob, ob], l2_abab)
    eta_abba += -1.00 * einsum('ik,kjab->abji', h_aa[oa, oa], l2_abab)
    eta_abba +=  1.00 * einsum('ca,ijcb->abji', h_aa[va, va], l2_abab)
    eta_abba +=  1.00 * einsum('cb,ijac->abji', h_bb[vb, vb], l2_abab)
    eta_abba += -1.00 * einsum('ka,ck,ijcb->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_abba += -1.00 * einsum('kb,ck,ijac->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_abba += -1.00 * einsum('jc,ck,ikab->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_abba += -1.00 * einsum('ic,ck,kjab->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    return eta_abba


def get_eta_baab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_baab =  1.00 * einsum('jb,ia->abji', h_aa[oa, va], l1_bb)
    eta_baab +=  1.00 * einsum('ia,jb->abji', h_bb[ob, vb], l1_aa)
    eta_baab += -1.00 * einsum('jk,kiba->abji', h_aa[oa, oa], l2_abab)
    eta_baab += -1.00 * einsum('ik,jkba->abji', h_bb[ob, ob], l2_abab)
    eta_baab +=  1.00 * einsum('ca,jibc->abji', h_bb[vb, vb], l2_abab)
    eta_baab +=  1.00 * einsum('cb,jica->abji', h_aa[va, va], l2_abab)
    eta_baab += -1.00 * einsum('ka,ck,jibc->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_baab += -1.00 * einsum('kb,ck,jica->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_baab += -1.00 * einsum('jc,ck,kiba->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_baab += -1.00 * einsum('ic,ck,jkba->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    return eta_baab


def get_eta_baba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_baba = -1.00 * einsum('ja,ib->abji', h_bb[ob, vb], l1_aa)
    eta_baba += -1.00 * einsum('ib,ja->abji', h_aa[oa, va], l1_bb)
    eta_baba +=  1.00 * einsum('jk,ikba->abji', h_bb[ob, ob], l2_abab)
    eta_baba +=  1.00 * einsum('ik,kjba->abji', h_aa[oa, oa], l2_abab)
    eta_baba += -1.00 * einsum('ca,ijbc->abji', h_bb[vb, vb], l2_abab)
    eta_baba += -1.00 * einsum('cb,ijca->abji', h_aa[va, va], l2_abab)
    eta_baba +=  1.00 * einsum('ka,ck,ijbc->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_baba +=  1.00 * einsum('kb,ck,ijca->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_baba +=  1.00 * einsum('jc,ck,ikba->abji', h_bb[ob, vb], t1_bb, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_baba +=  1.00 * einsum('ic,ck,kjba->abji', h_aa[oa, va], t1_aa, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    return eta_baba


def get_eta_bbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    contracted_intermediate = -1.00 * einsum('ja,ib->abji', h_bb[ob, vb], l1_bb)
    eta_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate)  + -1.00000 * einsum('abji->baji', contracted_intermediate)  +  1.00000 * einsum('abji->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,ikab->abji', h_bb[ob, ob], l2_bbbb)
    eta_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ca,ijcb->abji', h_bb[vb, vb], l2_bbbb)
    eta_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ka,ck,ijbc->abji', h_bb[ob, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jc,ck,kiab->abji', h_bb[ob, vb], t1_bb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abji->abij', contracted_intermediate) 
    return eta_bbbb
