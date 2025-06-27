from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.ccsd.containers import Spin_MBE, E1_spin, E2_spin


def get_cc_j_w_singles_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
    E_CC: float,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
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
    
    cc_j_w_singles_aa =  1.00 * einsum('kjbc,caik,bj->ai', g_aaaa[oa, oa, va, va], r2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjcb,caik,bj->ai', g_abab[oa, ob, va, vb], r2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jkbc,acik,bj->ai', g_abab[oa, ob, va, vb], r2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,acik,bj->ai', g_bbbb[ob, ob, vb, vb], r2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('kjbc,cakj,bi->ai', g_aaaa[oa, oa, va, va], r2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,ackj,bi->ai', g_abab[oa, ob, va, vb], r2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkbc,acjk,bi->ai', g_abab[oa, ob, va, vb], r2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('kjbc,aj,bcik->ai', g_aaaa[oa, oa, va, va], r1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkbc,aj,bcik->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkcb,aj,cbik->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_aaaa[oa, oa, va, va], r1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkbc,ai,bcjk->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjcb,ai,cbkj->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkcb,ai,cbjk->ai', g_abab[oa, ob, va, vb], r1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_bbbb[ob, ob, vb, vb], r1_aa, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    # manually added diagonal part
    cc_j_w_singles_aa -= E_CC * r1_aa
    return cc_j_w_singles_aa


def get_cc_j_w_singles_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
    E_CC: float,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    r1_aa = vector.singles[E1_spin.aa]
    r1_bb = vector.singles[E1_spin.bb]
    r2_aaaa = vector.doubles[E2_spin.aaaa]
    r2_abab = vector.doubles[E2_spin.abab]
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
    
    cc_j_w_singles_bb = -1.00 * einsum('kjbc,caki,bj->ai', g_aaaa[oa, oa, va, va], r2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjcb,caki,bj->ai', g_abab[oa, ob, va, vb], r2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jkbc,caik,bj->ai', g_abab[oa, ob, va, vb], r2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjbc,caik,bj->ai', g_bbbb[ob, ob, vb, vb], r2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjcb,cakj,bi->ai', g_abab[oa, ob, va, vb], r2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('jkcb,cajk,bi->ai', g_abab[oa, ob, va, vb], r2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('kjbc,cakj,bi->ai', g_bbbb[ob, ob, vb, vb], r2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,aj,bcki->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjcb,aj,cbki->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('kjbc,aj,bcik->ai', g_bbbb[ob, ob, vb, vb], r1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_aaaa[oa, oa, va, va], r1_bb, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkbc,ai,bcjk->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjcb,ai,cbkj->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkcb,ai,cbjk->ai', g_abab[oa, ob, va, vb], r1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_bbbb[ob, ob, vb, vb], r1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    # manually added diagonal part
    cc_j_w_singles_bb -= E_CC * r1_bb
    return cc_j_w_singles_bb
