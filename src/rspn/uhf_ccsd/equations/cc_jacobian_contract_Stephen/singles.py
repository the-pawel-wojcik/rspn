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
    
    cc_j_w_singles_aa =  1.00 * einsum('jj,ai->ai', f_aa[oa, oa], r1_aa)
    cc_j_w_singles_aa +=  1.00 * einsum('jj,ai->ai', f_bb[ob, ob], r1_aa)
    cc_j_w_singles_aa += -1.00 * einsum('ji,aj->ai', f_aa[oa, oa], r1_aa)
    cc_j_w_singles_aa +=  1.00 * einsum('ab,bi->ai', f_aa[va, va], r1_aa)
    cc_j_w_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', f_aa[oa, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jb,ai,bj->ai', f_bb[ob, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jb,aj,bi->ai', f_aa[oa, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jb,bi,aj->ai', f_aa[oa, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jb,bj,ai->ai', f_aa[oa, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jb,bj,ai->ai', f_bb[ob, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjkj,ai->ai', g_aaaa[oa, oa, oa, oa], r1_aa)
    cc_j_w_singles_aa += -0.50 * einsum('kjkj,ai->ai', g_abab[oa, ob, oa, ob], r1_aa)
    cc_j_w_singles_aa += -0.50 * einsum('jkjk,ai->ai', g_abab[oa, ob, oa, ob], r1_aa)
    cc_j_w_singles_aa += -0.50 * einsum('kjkj,ai->ai', g_bbbb[ob, ob, ob, ob], r1_aa)
    cc_j_w_singles_aa +=  1.00 * einsum('jabi,bj->ai', g_aaaa[oa, va, va, oa], r1_aa)
    cc_j_w_singles_aa +=  1.00 * einsum('ajib,bj->ai', g_abab[va, ob, oa, vb], r1_bb)
    cc_j_w_singles_aa +=  1.00 * einsum('kjbi,ak,bj->ai', g_aaaa[oa, oa, va, oa], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjib,ak,bj->ai', g_abab[oa, ob, oa, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbi,bk,aj->ai', g_aaaa[oa, oa, va, oa], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jkib,bk,aj->ai', g_abab[oa, ob, oa, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jabc,ci,bj->ai', g_aaaa[oa, va, va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('ajcb,ci,bj->ai', g_abab[va, ob, va, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jabc,cj,bi->ai', g_aaaa[oa, va, va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('ajbc,cj,bi->ai', g_abab[va, ob, va, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('kjbc,caik,bj->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjcb,caik,bj->ai', g_abab[oa, ob, va, vb], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jkbc,acik,bj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,acik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('kjbc,cakj,bi->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,ackj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkbc,acjk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('kjbc,bcik,aj->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkbc,bcik,aj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkcb,cbik,aj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkbc,bcjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjcb,cbkj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkcb,cbjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,ai,ck,bj->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('kjcb,ai,ck,bj->ai', g_abab[oa, ob, va, vb], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('jkbc,ai,ck,bj->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,ai,ck,bj->ai', g_bbbb[ob, ob, vb, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('kjbc,ak,ci,bj->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjcb,ak,ci,bj->ai', g_abab[oa, ob, va, vb], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,ak,cj,bi->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,ak,cj,bi->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,bk,ci,aj->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jkcb,bk,ci,aj->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('jkbc,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.50 * einsum('kjcb,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jb,baij->ai', f_aa[oa, va], r2_aaaa)
    cc_j_w_singles_aa +=  1.00 * einsum('jb,abij->ai', f_bb[ob, vb], r2_abab)
    cc_j_w_singles_aa += -0.50 * einsum('kjbi,bakj->ai', g_aaaa[oa, oa, va, oa], r2_aaaa)
    cc_j_w_singles_aa += -0.50 * einsum('kjib,abkj->ai', g_abab[oa, ob, oa, vb], r2_abab)
    cc_j_w_singles_aa += -0.50 * einsum('jkib,abjk->ai', g_abab[oa, ob, oa, vb], r2_abab)
    cc_j_w_singles_aa += -0.50 * einsum('jabc,bcij->ai', g_aaaa[oa, va, va, va], r2_aaaa)
    cc_j_w_singles_aa +=  0.50 * einsum('ajbc,bcij->ai', g_abab[va, ob, va, vb], r2_abab)
    cc_j_w_singles_aa +=  0.50 * einsum('ajcb,cbij->ai', g_abab[va, ob, va, vb], r2_abab)
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkbc,ai,bcjk->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjcb,ai,cbkj->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('jkcb,ai,cbjk->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  0.250 * einsum('kjbc,ai,bckj->ai', g_bbbb[ob, ob, vb, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,ak,bcij->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,ak,bcij->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjcb,ak,cbij->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjbc,ci,bakj->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('kjcb,ci,abkj->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa += -0.50 * einsum('jkcb,ci,abjk->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('kjbc,ck,baij->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa +=  1.00 * einsum('kjcb,ck,abij->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('jkbc,ck,baij->ai', g_abab[oa, ob, va, vb], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_aa += -1.00 * einsum('kjbc,ck,abij->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    # Manually added diagonal term
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
    
    cc_j_w_singles_bb =  1.00 * einsum('jj,ai->ai', f_aa[oa, oa], r1_bb)
    cc_j_w_singles_bb +=  1.00 * einsum('jj,ai->ai', f_bb[ob, ob], r1_bb)
    cc_j_w_singles_bb += -1.00 * einsum('ji,aj->ai', f_bb[ob, ob], r1_bb)
    cc_j_w_singles_bb +=  1.00 * einsum('ab,bi->ai', f_bb[vb, vb], r1_bb)
    cc_j_w_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', f_aa[oa, va], t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jb,ai,bj->ai', f_bb[ob, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jb,aj,bi->ai', f_bb[ob, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jb,bi,aj->ai', f_bb[ob, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jb,bj,ai->ai', f_aa[oa, va], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jb,bj,ai->ai', f_bb[ob, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjkj,ai->ai', g_aaaa[oa, oa, oa, oa], r1_bb)
    cc_j_w_singles_bb += -0.50 * einsum('kjkj,ai->ai', g_abab[oa, ob, oa, ob], r1_bb)
    cc_j_w_singles_bb += -0.50 * einsum('jkjk,ai->ai', g_abab[oa, ob, oa, ob], r1_bb)
    cc_j_w_singles_bb += -0.50 * einsum('kjkj,ai->ai', g_bbbb[ob, ob, ob, ob], r1_bb)
    cc_j_w_singles_bb +=  1.00 * einsum('jabi,bj->ai', g_abab[oa, vb, va, ob], r1_aa)
    cc_j_w_singles_bb +=  1.00 * einsum('jabi,bj->ai', g_bbbb[ob, vb, vb, ob], r1_bb)
    cc_j_w_singles_bb += -1.00 * einsum('jkbi,ak,bj->ai', g_abab[oa, ob, va, ob], t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjbi,ak,bj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbi,bk,aj->ai', g_abab[oa, ob, va, ob], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbi,bk,aj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jabc,ci,bj->ai', g_abab[oa, vb, va, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jabc,ci,bj->ai', g_bbbb[ob, vb, vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jacb,cj,bi->ai', g_abab[oa, vb, va, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jabc,cj,bi->ai', g_bbbb[ob, vb, vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,caki,bj->ai', g_aaaa[oa, oa, va, va], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjcb,caki,bj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jkbc,caik,bj->ai', g_abab[oa, ob, va, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjbc,caik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjcb,cakj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('jkcb,cajk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('kjbc,cakj,bi->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,bcki,aj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjcb,cbki,aj->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('kjbc,bcik,aj->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkbc,bcjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjcb,cbkj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkcb,cbjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,ai,ck,bj->ai', g_aaaa[oa, oa, va, va], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjcb,ai,ck,bj->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jkbc,ai,ck,bj->ai', g_abab[oa, ob, va, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,ai,ck,bj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jkbc,ak,ci,bj->ai', g_abab[oa, ob, va, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjbc,ak,ci,bj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('jkcb,ak,cj,bi->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,ak,cj,bi->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,bk,ci,aj->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,bk,ci,aj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('jkbc,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.50 * einsum('kjcb,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jb,baji->ai', f_aa[oa, va], r2_abab)
    cc_j_w_singles_bb += -1.00 * einsum('jb,baij->ai', f_bb[ob, vb], r2_bbbb)
    cc_j_w_singles_bb += -0.50 * einsum('kjbi,bakj->ai', g_abab[oa, ob, va, ob], r2_abab)
    cc_j_w_singles_bb += -0.50 * einsum('jkbi,bajk->ai', g_abab[oa, ob, va, ob], r2_abab)
    cc_j_w_singles_bb += -0.50 * einsum('kjbi,bakj->ai', g_bbbb[ob, ob, vb, ob], r2_bbbb)
    cc_j_w_singles_bb +=  0.50 * einsum('jabc,bcji->ai', g_abab[oa, vb, va, vb], r2_abab)
    cc_j_w_singles_bb +=  0.50 * einsum('jacb,cbji->ai', g_abab[oa, vb, va, vb], r2_abab)
    cc_j_w_singles_bb += -0.50 * einsum('jabc,bcij->ai', g_bbbb[ob, vb, vb, vb], r2_bbbb)
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_aaaa[oa, oa, va, va], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkbc,ai,bcjk->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjcb,ai,cbkj->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('jkcb,ai,cbjk->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb +=  0.250 * einsum('kjbc,ai,bckj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('jkbc,ak,bcji->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('jkcb,ak,cbji->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,ak,bcij->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,ci,bakj->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('jkbc,ci,bajk->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -0.50 * einsum('kjbc,ci,bakj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjbc,ck,baji->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb += -1.00 * einsum('kjcb,ck,baij->ai', g_abab[oa, ob, va, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('jkbc,ck,baji->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles_bb +=  1.00 * einsum('kjbc,ck,baij->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    # Manually added diagonal term
    cc_j_w_singles_bb -= E_CC * r1_bb
    return cc_j_w_singles_bb
