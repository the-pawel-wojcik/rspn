from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.meta.spin_mbe import Spin_MBE, E1_spin, E2_spin


def get_diagonal_part_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
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
    
    diagonal_part_aa =  1.00 * einsum('ai->ai', r1_aa)
    return diagonal_part_aa


def get_diagonal_part_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
    vector: Spin_MBE,
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
    
    diagonal_part_bb =  1.00 * einsum('ai->ai', r1_bb)
    return diagonal_part_bb
