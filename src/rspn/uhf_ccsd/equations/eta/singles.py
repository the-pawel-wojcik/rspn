from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_eta_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
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
    
    eta_aa =  1.00 * einsum('ia->ai', h_aa[oa, va])
    eta_aa += -1.00 * einsum('ij,ja->ai', h_aa[oa, oa], l1_aa)
    eta_aa +=  1.00 * einsum('ba,ib->ai', h_aa[va, va], l1_aa)
    eta_aa += -1.00 * einsum('ja,bj,ib->ai', h_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aa += -1.00 * einsum('ib,bj,ja->ai', h_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_aa += -0.50 * einsum('ka,cbjk,jicb->ai', h_aa[oa, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aa += -0.50 * einsum('ka,cbkj,ijcb->ai', h_aa[oa, va], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aa += -0.50 * einsum('ka,bckj,ijbc->ai', h_aa[oa, va], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aa += -0.50 * einsum('ic,cbjk,jkab->ai', h_aa[oa, va], t2_aaaa, l2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_aa += -0.50 * einsum('ic,cbjk,jkab->ai', h_aa[oa, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_aa += -0.50 * einsum('ic,cbkj,kjab->ai', h_aa[oa, va], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    return eta_aa


def get_eta_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
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
    
    eta_bb =  1.00 * einsum('ia->ai', h_bb[ob, vb])
    eta_bb += -1.00 * einsum('ij,ja->ai', h_bb[ob, ob], l1_bb)
    eta_bb +=  1.00 * einsum('ba,ib->ai', h_bb[vb, vb], l1_bb)
    eta_bb += -1.00 * einsum('ja,bj,ib->ai', h_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bb += -1.00 * einsum('ib,bj,ja->ai', h_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_bb += -0.50 * einsum('ka,cbjk,jicb->ai', h_bb[ob, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bb += -0.50 * einsum('ka,bcjk,jibc->ai', h_bb[ob, vb], t2_abab, l2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bb += -0.50 * einsum('ka,cbjk,jicb->ai', h_bb[ob, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bb += -0.50 * einsum('ic,bcjk,jkba->ai', h_bb[ob, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_bb += -0.50 * einsum('ic,bckj,kjba->ai', h_bb[ob, vb], t2_abab, l2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_bb += -0.50 * einsum('ic,cbjk,jkab->ai', h_bb[ob, vb], t2_bbbb, l2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    return eta_bb
