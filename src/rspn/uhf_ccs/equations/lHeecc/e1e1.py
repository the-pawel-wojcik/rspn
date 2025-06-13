from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.uhf_ccs import UHF_CCS_Data, UHF_CCS_Lambda_Data


def get_lhe1e1cc_aaaa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_aaaa = -1.00 * einsum('jiab->aibj', g_aaaa[oa, oa, va, va])
    contracted_intermediate =  1.00 * einsum('jiak,kb->aibj', g_aaaa[oa, oa, va, oa], l1_aa)
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('icab,jc->aibj', g_aaaa[oa, va, va, va], l1_aa)
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kiab,ck,jc->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,ck,kb->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_aaaa +=  1.00 * einsum('kjac,ck,ib->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('jkac,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa +=  1.00 * einsum('kibc,ck,ja->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('ikbc,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aaaa += -1.00 * einsum('ja,ib->aibj', f_aa[oa, va], l1_aa)
    lhe1e1cc_aaaa += -1.00 * einsum('ib,ja->aibj', f_aa[oa, va], l1_aa)
    return lhe1e1cc_aaaa


def get_lhe1e1cc_aabb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_aabb =  1.00 * einsum('ijab->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_aabb += -1.00 * einsum('ijak,kb->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_aabb += -1.00 * einsum('ijkb,ka->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_aabb +=  1.00 * einsum('icab,jc->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_aabb +=  1.00 * einsum('cjab,ic->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_aabb += -1.00 * einsum('ikab,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('kjab,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('ijac,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_aabb += -1.00 * einsum('ijcb,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    return lhe1e1cc_aabb


def get_lhe1e1cc_abba(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_abba = -1.00 * einsum('jiab->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_abba +=  1.00 * einsum('jiak,kb->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_abba +=  1.00 * einsum('jikb,ka->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_abba += -1.00 * einsum('ciab,jc->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_abba += -1.00 * einsum('jcab,ic->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_abba +=  1.00 * einsum('kiab,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jkab,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jiac,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('jicb,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('kjac,ck,ib->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('jkac,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('kicb,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba +=  1.00 * einsum('kibc,ck,ja->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_abba += -1.00 * einsum('ja,ib->aibj', f_aa[oa, va], l1_bb)
    lhe1e1cc_abba += -1.00 * einsum('ib,ja->aibj', f_bb[ob, vb], l1_aa)
    return lhe1e1cc_abba


def get_lhe1e1cc_baab(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_baab = -1.00 * einsum('ijba->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_baab +=  1.00 * einsum('ijka,kb->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_baab +=  1.00 * einsum('ijbk,ka->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_baab += -1.00 * einsum('icba,jc->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_baab += -1.00 * einsum('cjba,ic->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_baab +=  1.00 * einsum('ikba,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kjba,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ijca,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('ijbc,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('kjca,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kjac,ck,ib->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab +=  1.00 * einsum('kibc,ck,ja->aibj', g_aaaa[oa, oa, va, va], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ikbc,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_baab += -1.00 * einsum('ja,ib->aibj', f_bb[ob, vb], l1_aa)
    lhe1e1cc_baab += -1.00 * einsum('ib,ja->aibj', f_aa[oa, va], l1_bb)
    return lhe1e1cc_baab


def get_lhe1e1cc_bbaa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_bbaa =  1.00 * einsum('jiba->aibj', g_abab[oa, ob, va, vb])
    lhe1e1cc_bbaa += -1.00 * einsum('jika,kb->aibj', g_abab[oa, ob, oa, vb], l1_aa)
    lhe1e1cc_bbaa += -1.00 * einsum('jibk,ka->aibj', g_abab[oa, ob, va, ob], l1_bb)
    lhe1e1cc_bbaa +=  1.00 * einsum('ciba,jc->aibj', g_abab[va, ob, va, vb], l1_aa)
    lhe1e1cc_bbaa +=  1.00 * einsum('jcba,ic->aibj', g_abab[oa, vb, va, vb], l1_bb)
    lhe1e1cc_bbaa += -1.00 * einsum('kiba,ck,jc->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jkba,ck,ic->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jica,ck,kb->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbaa += -1.00 * einsum('jibc,ck,ka->aibj', g_abab[oa, ob, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    return lhe1e1cc_bbaa


def get_lhe1e1cc_bbbb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    kd_aa =  uhf_data.identity_aa
    kd_bb =  uhf_data.identity_bb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    lhe1e1cc_bbbb = -1.00 * einsum('jiab->aibj', g_bbbb[ob, ob, vb, vb])
    contracted_intermediate =  1.00 * einsum('jiak,kb->aibj', g_bbbb[ob, ob, vb, ob], l1_bb)
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('icab,jc->aibj', g_bbbb[ob, vb, vb, vb], l1_bb)
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kiab,ck,jc->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,ck,kb->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc_bbbb += -1.00 * einsum('kjca,ck,ib->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('kjac,ck,ib->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('kicb,ck,ja->aibj', g_abab[oa, ob, va, vb], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb +=  1.00 * einsum('kibc,ck,ja->aibj', g_bbbb[ob, ob, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc_bbbb += -1.00 * einsum('ja,ib->aibj', f_bb[ob, vb], l1_bb)
    lhe1e1cc_bbbb += -1.00 * einsum('ib,ja->aibj', f_bb[ob, vb], l1_bb)
    return lhe1e1cc_bbbb
