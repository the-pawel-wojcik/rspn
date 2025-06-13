from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.uhf_ccs import UHF_CCS_Data, UHF_CCS_Lambda_Data


def get_eta_aa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_aa =  1.00 * einsum('ia->ai', h_aa[oa, va])
    eta_aa += -1.00 * einsum('ij,ja->ai', h_aa[oa, oa], l1_aa)
    eta_aa +=  1.00 * einsum('ba,ib->ai', h_aa[va, va], l1_aa)
    eta_aa += -1.00 * einsum('ja,bj,ib->ai', h_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    eta_aa += -1.00 * einsum('ib,bj,ja->ai', h_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    return eta_aa


def get_eta_bb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
    operator_aa: NDArray,
    operator_bb: NDArray,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    h_aa = operator_aa
    h_bb = operator_bb
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
    
    eta_bb =  1.00 * einsum('ia->ai', h_bb[ob, vb])
    eta_bb += -1.00 * einsum('ij,ja->ai', h_bb[ob, ob], l1_bb)
    eta_bb +=  1.00 * einsum('ba,ib->ai', h_bb[vb, vb], l1_bb)
    eta_bb += -1.00 * einsum('ja,bj,ib->ai', h_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    eta_bb += -1.00 * einsum('ib,bj,ja->ai', h_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    return eta_bb
