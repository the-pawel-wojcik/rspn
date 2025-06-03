from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_doubles_doubles_aaaaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmce,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmce,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmde,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmde,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaaaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aaaaaaaa


def get_doubles_doubles_aabaaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,alcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,blcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aabaaaba +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klie,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmie,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,alce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,blce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aabaaaba +=  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,klce,beij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,bemj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmce,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,klde,beij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,bemj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmde,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klfe,feij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,mlef,efmj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,mlfe,femj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_aabaaaba += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klfe,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aabaaaba


def get_doubles_doubles_aabaaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,akcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,bkcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aabaaaab += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkej,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkie,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,akce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,amce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,bkce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bmce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aabaaaab += -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaab += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,lkce,beij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmce,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,bemj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,nmce,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,mnce,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,beij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmde,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,bemj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,nmde,benm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,mnde,bemn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,lkef,efij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,lkfe,feij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,mkef,efmj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,mkfe,femj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_aabaaaab +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkfe,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabaaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aabaaaab


def get_doubles_doubles_aaabaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,kacj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aaabaaba += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klje,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klei,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kmje,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,amce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bmce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aaabaaba += -1.00 * einsum('il,mkcd,abjm->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaba += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,klce,beji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bemi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kmce,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,nmce,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,mnce,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,beji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bemi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,kmde,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,nmde,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,mnde,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,klef,efji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,klfe,feji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,kmef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,kmfe,fejm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_aaabaaba +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kmfe,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aaabaaba


def get_doubles_doubles_aaabaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,akci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,bkci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aaabaaab +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkje,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkei,ej->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,akce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,bkce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_aaabaaab +=  1.00 * einsum('ik,mlcd,abjm->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,lkce,beji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lmce,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bemi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,beji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmde,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bemi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,lkef,efji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,lkfe,feji->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,lmef,efjm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,lmfe,fejm->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_aaabaaab += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkef,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmfe,em,fj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aaabaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aaabaaab


def get_doubles_doubles_aabbaabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,alcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,blcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[va, va, va, va], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,akce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,bkce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bemi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bemi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bemi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bemi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_aabbaabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aabbaabb


def get_doubles_doubles_aaaaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_aaaaabba += -1.00 * einsum('bc,ik,aljd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabba +=  1.00 * einsum('bc,jk,alid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabba +=  1.00 * einsum('ac,ik,bljd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabba += -1.00 * einsum('ac,jk,blid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ac,ik,mljd,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlid,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,aled,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,bled,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kled,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aaaaabba


def get_doubles_doubles_aaaaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_aaaaabab +=  1.00 * einsum('bc,il,akjd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabab += -1.00 * einsum('bc,jl,akid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabab += -1.00 * einsum('ac,il,bkjd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaaabab +=  1.00 * einsum('ac,jl,bkid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ac,il,mkjd,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkid,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,aked,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,bked,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lked,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaaabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aaaaabab


def get_doubles_doubles_aabaabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bc,jk,alid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,blid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlid,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_aabaabbb +=  1.00 * einsum('bc,jk,aled,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb += -1.00 * einsum('bc,jl,aked,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb += -1.00 * einsum('ac,jk,bled,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00 * einsum('ac,jl,bked,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb += -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,klde,beij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabaabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aabaabbb


def get_doubles_doubles_aaababbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bc,ik,aljd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,bljd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mljd,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_aaababbb += -1.00 * einsum('bc,ik,aled,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00 * einsum('bc,il,aked,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00 * einsum('ac,ik,bled,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb += -1.00 * einsum('ac,il,bked,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00 * einsum('ik,mlcd,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaababbb += -1.00 * einsum('il,mkcd,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,klde,beji->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mled,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mlde,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,mked,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,mkde,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mled,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,mked,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaababbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aaababbb


def get_doubles_doubles_aaaababa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_aaaababa +=  1.00 * einsum('bd,ik,aljc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaababa += -1.00 * einsum('bd,jk,alic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaababa += -1.00 * einsum('ad,ik,bljc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaababa +=  1.00 * einsum('ad,jk,blic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,ik,mljc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlic,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,alec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,blec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mldc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,klec,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaababa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aaaababa


def get_doubles_doubles_aaaabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_aaaabaab += -1.00 * einsum('bd,il,akjc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaabaab +=  1.00 * einsum('bd,jl,akic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaabaab +=  1.00 * einsum('ad,il,bkjc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaaabaab += -1.00 * einsum('ad,jl,bkic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,il,mkjc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkic,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,akec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,bkec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,mkdc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,lkec,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,beim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaaabaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_aaaabaab


def get_doubles_doubles_aabababb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,jk,alic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,blic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlic,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_aabababb += -1.00 * einsum('bd,jk,alec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00 * einsum('bd,jl,akec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00 * einsum('ad,jk,blec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb += -1.00 * einsum('ad,jl,bkec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00 * einsum('jk,mldc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabababb += -1.00 * einsum('jl,mkdc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,klce,beij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,beim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aabababb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aabababb


def get_doubles_doubles_aaabbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,aljc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,bljc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mljc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_aaabbabb +=  1.00 * einsum('bd,ik,alec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb += -1.00 * einsum('bd,il,akec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb += -1.00 * einsum('ad,ik,blec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00 * einsum('ad,il,bkec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb += -1.00 * einsum('ik,mldc,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00 * einsum('il,mkdc,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,klce,beji->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mlec,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mlce,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,mkec,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,mkce,bejm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mlec,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,mkec,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_aaabbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_aaabbabb


def get_doubles_doubles_abbaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lmcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,lmdj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abbaaaaa += -1.00 * einsum('ik,mlcd,abmj->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('il,mkcd,abmj->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ad,ik,mlce,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ad,ik,lmce,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ad,il,mkce,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ad,il,kmce,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ac,ik,mlde,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ac,ik,lmde,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ac,il,mkde,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ac,il,kmde,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ad,ik,lmce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ad,il,kmce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaaaaa += -1.00 * einsum('ac,ik,lmde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaaaaa +=  1.00 * einsum('ac,il,kmde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    return doubles_doubles_abbaaaaa


def get_doubles_doubles_ababaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmdi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_ababaaaa +=  1.00 * einsum('jk,mlcd,abmi->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('jl,mkcd,abmi->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ad,klce,ebji->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ad,jk,mlce,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ad,jk,lmce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ad,jl,mkce,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ad,jl,kmce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ac,klde,ebji->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ac,jk,mlde,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ac,jk,lmde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ac,jl,mkde,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ac,jl,kmde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ad,jk,lmce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ad,jl,kmce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababaaaa +=  1.00 * einsum('ac,jk,lmde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababaaaa += -1.00 * einsum('ac,jl,kmde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    return doubles_doubles_ababaaaa


def get_doubles_doubles_abbbaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abbbaaba += -1.00 * einsum('ad,il,kmcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00 * einsum('ac,il,kmdj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00 * einsum('ad,jl,kmci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba += -1.00 * einsum('ac,jl,kmdi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,mkcd,abmi->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbaaba += -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbaaba +=  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abbbaaba


def get_doubles_doubles_abbbaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abbbaaab +=  1.00 * einsum('ad,ik,lmcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab += -1.00 * einsum('ac,ik,lmdj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab += -1.00 * einsum('ad,jk,lmci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00 * einsum('ac,jk,lmdi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,abmi->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbaaab +=  1.00 * einsum('ad,lkce,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbaaab += -1.00 * einsum('ac,lkde,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abbbaaab


def get_doubles_doubles_abaaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaaabaa +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,lbjd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lbid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmjd,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmid,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,amcd,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,mbcd,am->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lbed,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kbed,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,mbed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,kmcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,mncd,abmn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaaabaa +=  1.00 * einsum('bd,klce,eaij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lmce,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kmce,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaaabaa +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,mncd,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaaabaa += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abaaabaa


def get_doubles_doubles_abbaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,alcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,jl,kbid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,bd,klie,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,mkci,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,jl,kmid,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,kmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ik,jl,amcd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ik,jl,mbcd,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,alce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,lbde,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,jl,kbed,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,jl,mbed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ik,mlcd,abmj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('jl,kmcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ik,jl,mncd,abmn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,klce,aeij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,ik,mlce,aemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,mkce,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,kmce,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('bd,ik,jl,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('bd,ik,jl,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,kled,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,mled,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,mlde,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,jl,kmed,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ac,bd,klfe,feij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,bd,ik,mlef,efmj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,bd,ik,mlfe,femj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ik,jl,mncd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,ik,mlce,am,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,jl,kmed,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba +=  1.00 * einsum('ac,bd,klfe,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabba += -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abbaabba


def get_doubles_doubles_abbaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,jk,bd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,jk,mc,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,bd,lkij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,akcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,kbdj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,jk,lbid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,il,mkcj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,bd,lkej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,bd,lkie,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,jk,lmid,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('il,jk,amcd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('il,jk,mbcd,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,akce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,il,jk,amce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,jk,lbed,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,kbde,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,jk,mbed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,il,jk,mbde,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('jk,lmcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('il,mkcd,abmj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('il,jk,mncd,abmn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,lkce,aeij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,mlce,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,lmce,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,il,mkce,aemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('bd,il,jk,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('bd,il,jk,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('bd,il,jk,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,lked,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,jk,lmed,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,mked,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,mkde,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,il,jk,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,il,jk,mned,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('ac,bd,lkef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('ac,bd,lkfe,feij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,bd,il,mkef,efmj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  0.50 * einsum('ac,bd,il,mkfe,femj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('il,jk,mncd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('bd,il,mkce,am,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,jk,lmed,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,il,jk,mned,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab += -1.00 * einsum('ac,bd,lkfe,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbaabab +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abbaabab


def get_doubles_doubles_abababba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababba += -1.00 * einsum('ac,il,jk,bd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,il,jk,mc,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,bd,klji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,kacj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,jk,alci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,kbjd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,mkcj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,il,kmjd,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,bd,klje,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,bd,klei,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,kmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('il,jk,amcd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('il,jk,mbcd,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,jk,alce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,jk,amce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,kbed,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,il,jk,mbed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,jk,mbde,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('jk,mlcd,abmi->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('il,kmcd,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('il,jk,mncd,abmn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,klce,aeji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,jk,mlce,aemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,mkce,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,kmce,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('bd,il,jk,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('bd,il,jk,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('bd,il,jk,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,kled,ebji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,mled,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,il,kmed,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,il,jk,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,il,jk,mned,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('ac,bd,klef,efji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('ac,bd,klfe,feji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,bd,il,kmef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  0.50 * einsum('ac,bd,il,kmfe,fejm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('il,jk,mncd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('bd,il,mkce,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,il,kmed,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,jk,mned,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababba +=  1.00 * einsum('ac,bd,il,kmfe,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abababba


def get_doubles_doubles_abababab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abababab += -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,bd,lkji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab += -1.00 * einsum('bd,jl,akci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,lbjd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,kbdi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,ik,lmjd,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,bd,lkje,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,bd,lkei,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,jl,mkci,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ik,jl,amcd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ik,jl,mbcd,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('bd,jl,akce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,lbed,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,ik,jl,mbed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ik,lmcd,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('jl,mkcd,abmi->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ik,jl,mncd,abmn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('bd,lkce,aeji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,mlce,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,lmce,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,jl,mkce,aemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('bd,ik,jl,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('bd,ik,jl,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,lked,ebji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,ik,lmed,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,mked,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ac,bd,lkef,efji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ac,bd,lkfe,feji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,bd,ik,lmef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,bd,ik,lmfe,fejm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ik,jl,mncd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,ik,mlce,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,ik,lmed,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab +=  1.00 * einsum('ac,bd,lkef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,ik,lmfe,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abababab += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abababab


def get_doubles_doubles_abbbabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbabbb +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('bd,ik,alcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,amcd,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,mbcd,am->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,alce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,akce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,amce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,mbed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,abmi->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,mkcd,abmi->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,mncd,abmn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,aemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,aemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,nmce,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,mnce,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbabbb +=  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jk,mled,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mked,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbabbb +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,mncd,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbabbb += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abbbabbb


def get_doubles_doubles_abaabaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabaaa += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lbjc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,abdc->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lmjc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmic,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,amdc,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,mbdc,am->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,amde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kbec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmdc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,kmdc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,nmdc,abnm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,mndc,abmn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabaaa += -1.00 * einsum('bc,klde,eaij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lmde,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kmde,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabaaa += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,mndc,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabaaa +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abaabaaa


def get_doubles_doubles_abbababa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,ki->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,le,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,aldj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,kadi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,jl,kbic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa += -1.00 * einsum('ik,jl,abdc->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,bc,klie,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,jl,kmic,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,mkdi,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,kmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ik,jl,amdc,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ik,jl,mbdc,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,alde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,ik,jl,amde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,jl,kbec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ik,mldc,abmj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('jl,kmdc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ik,jl,nmdc,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ik,jl,mndc,abmn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,klec,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,mlec,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,mlce,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,jl,kmec,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,klde,aeij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,ik,mlde,aemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,mkde,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,kmde,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('bc,ik,jl,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('bc,ik,jl,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ad,bc,klfe,feij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,bc,ik,mlef,efmj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,bc,ik,mlfe,femj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ad,bc,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,bc,jl,kmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  0.50 * einsum('ad,bc,jl,kmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ik,jl,mndc,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,jl,kmec,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,ik,mlde,am,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa += -1.00 * einsum('ad,bc,klfe,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbababa +=  1.00 * einsum('ad,bc,jl,kmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abbababa


def get_doubles_doubles_abbabaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,kj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,il,jk,ad->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,jk,md,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,ke,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,bc,lkij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,akdj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,jk,lbic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab +=  1.00 * einsum('il,jk,abdc->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,il,mkdj,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,bc,lkej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,bc,lkie,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,jk,lmic,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,lmie,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('il,jk,amdc,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('il,jk,mbdc,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,akde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,jk,made,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,il,jk,amde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,jk,lbec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,jk,mbec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('jk,lmdc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('il,mkdc,abmj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('il,jk,nmdc,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('il,jk,mndc,abmn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,lkec,ebij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,jk,lmec,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,mkec,ebmj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,mkce,ebjm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,il,jk,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,il,jk,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,lkde,aeij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,mlde,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,lmde,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,il,mkde,aemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('bc,il,jk,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('bc,il,jk,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('bc,il,jk,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('ad,bc,lkef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('ad,bc,lkfe,feij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,bc,jk,lmef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,bc,jk,lmfe,feim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,bc,il,mkef,efmj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -0.50 * einsum('ad,bc,il,mkfe,femj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  0.50 * einsum('ad,bc,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('il,jk,mndc,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,jk,lmec,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,il,jk,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('bc,il,mkde,am,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab +=  1.00 * einsum('ad,bc,lkfe,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,jk,lmfe,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbabaab += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_abbabaab


def get_doubles_doubles_ababbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,kj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,jk,ad->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,il,jk,md,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,ke,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,bc,klji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,kadj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,jk,aldi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,kbjc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba +=  1.00 * einsum('il,jk,abdc->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,il,kmjc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,mkdj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,bc,klje,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,bc,klei,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,kmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('il,jk,amdc,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('il,jk,mbdc,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,jk,alde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,kade,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,il,jk,made,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,jk,amde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,kbec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,il,jk,mbec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('jk,mldc,abmi->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('il,kmdc,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('il,jk,nmdc,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('il,jk,mndc,abmn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,klec,ebji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,mlec,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,il,kmec,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,il,jk,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,il,jk,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,klde,aeji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,jk,mlde,aemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,mkde,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,kmde,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('bc,il,jk,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('bc,il,jk,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('bc,il,jk,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('ad,bc,klef,efji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('ad,bc,klfe,feji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,bc,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,bc,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  0.50 * einsum('ad,bc,il,mkef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,bc,il,kmef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -0.50 * einsum('ad,bc,il,kmfe,fejm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('il,jk,mndc,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,il,kmec,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,jk,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('bc,il,mkde,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaba += -1.00 * einsum('ad,bc,il,kmfe,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_ababbaba


def get_doubles_doubles_ababbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,ki->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,le,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,bc,lkji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,jl,akdi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,lbjc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab += -1.00 * einsum('ik,jl,abdc->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,ik,lmjc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,bc,lkje,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,bc,lkei,ej->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,lmje,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,jl,mkdi,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ik,jl,amdc,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ik,jl,mbdc,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,lade,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,jl,akde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,jl,amde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,lbec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ik,lmdc,abjm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('jl,mkdc,abmi->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ik,jl,nmdc,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ik,jl,mndc,abmn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,lkec,ebji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,ik,lmec,ebjm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,mkec,ebmi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,lkde,aeji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,mlde,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,lmde,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,jl,mkde,aemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('bc,ik,jl,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('bc,ik,jl,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ad,bc,lkef,efji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ad,bc,lkfe,feji->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ad,bc,ik,mlef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,bc,ik,lmef,efjm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,bc,ik,lmfe,fejm->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,bc,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  0.50 * einsum('ad,bc,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ik,jl,mndc,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,ik,lmec,bm,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,ik,mlde,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab += -1.00 * einsum('ad,bc,lkef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,ik,lmfe,em,fj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_ababbaab +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_ababbaab


def get_doubles_doubles_abbbbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbbabb += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('bc,ik,aldj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,aldi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,abdc->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,amdc,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,mbdc,am->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,alde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,akde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,amde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mldc,abmi->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkdc,abmi->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,nmdc,abnm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,mndc,abmn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbbabb += -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,mlec,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkec,ebmi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,aemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,aemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,nmde,aenm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,mnde,aemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbbabb += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,mlef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,mlfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,mkef,efmi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,mkfe,femi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,mndc,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abbbbabb +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_aa[va, va], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_abbbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abbbbabb


def get_doubles_doubles_abaabbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,aljc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,alic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abaabbba += -1.00 * einsum('bd,ik,mljc,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00 * einsum('bc,ik,mljd,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00 * einsum('bd,jk,mlic,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba += -1.00 * einsum('bc,jk,mlid,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jk,alec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabbba += -1.00 * einsum('bd,klec,eaij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bd,jk,mlec,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,mlce,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabbba +=  1.00 * einsum('bc,kled,eaij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bc,jk,mled,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,mlde,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,mlec,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,mled,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abaabbba


def get_doubles_doubles_abaabbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,il,akjc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,akic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abaabbab +=  1.00 * einsum('bd,il,mkjc,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab += -1.00 * einsum('bc,il,mkjd,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab += -1.00 * einsum('bd,jl,mkic,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00 * einsum('bc,jl,mkid,am->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bd,jl,akec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabbab +=  1.00 * einsum('bd,lkec,eaij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jl,mkec,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,mkce,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_abaabbab += -1.00 * einsum('bc,lked,eaij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bc,jl,mked,eaim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,mkde,aeim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,mkec,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,mked,am,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_abaabbab


def get_doubles_doubles_abbabbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,jk,alic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,mlic,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,mlid,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,alec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,akec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_abbabbbb += -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bd,klce,aeij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bd,jk,mlec,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bd,jk,mlce,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bd,jl,mkec,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bd,jl,mkce,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bc,klde,aeij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bc,jk,mled,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bc,jk,mlde,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bc,jl,mked,eaim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bc,jl,mkde,aeim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bd,jk,mlec,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bd,jl,mkec,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabbbb += -1.00 * einsum('bc,jk,mled,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_abbabbbb +=  1.00 * einsum('bc,jl,mked,am,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    return doubles_doubles_abbabbbb


def get_doubles_doubles_ababbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,aljc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,mljc,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,mljd,am->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,alec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,akec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_ababbbbb +=  1.00 * einsum('ik,mlcd,abjm->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('il,mkcd,abjm->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bd,klce,aeji->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bd,ik,mlec,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bd,ik,mlce,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bd,il,mkec,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bd,il,mkce,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bc,klde,aeji->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bc,ik,mled,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bc,ik,mlde,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bc,il,mked,eajm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bc,il,mkde,aejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bd,ik,mlec,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bd,il,mkec,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbbbb +=  1.00 * einsum('bc,ik,mled,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_ababbbbb += -1.00 * einsum('bc,il,mked,am,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    return doubles_doubles_ababbbbb


def get_doubles_doubles_babaaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,lmcj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,lmdj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_babaaaaa +=  1.00 * einsum('ik,mlcd,bamj->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('il,mkcd,bamj->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bd,klce,eaij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bd,ik,mlce,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bd,ik,lmce,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bd,il,mkce,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bd,il,kmce,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bc,klde,eaij->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bc,ik,mlde,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bc,ik,lmde,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bc,il,mkde,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bc,il,kmde,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bd,ik,lmce,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bd,il,kmce,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaaaaa +=  1.00 * einsum('bc,ik,lmde,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaaaaa += -1.00 * einsum('bc,il,kmde,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    return doubles_doubles_babaaaaa


def get_doubles_doubles_baabaaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,lmci,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,lmdi,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_baabaaaa += -1.00 * einsum('jk,mlcd,bami->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('jl,mkcd,bami->abjicdlk', kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bd,klce,eaji->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bd,jk,mlce,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bd,jk,lmce,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bd,jl,mkce,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bd,jl,kmce,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bc,klde,eaji->abjicdlk', kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bc,jk,mlde,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bc,jk,lmde,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bc,jl,mkde,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bc,jl,kmde,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bd,jk,lmce,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bd,jl,kmce,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabaaaa += -1.00 * einsum('bc,jk,lmde,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabaaaa +=  1.00 * einsum('bc,jl,kmde,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    return doubles_doubles_baabaaaa


def get_doubles_doubles_babbaaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,il,kacj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_babbaaba +=  1.00 * einsum('bd,il,kmcj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba += -1.00 * einsum('bc,il,kmdj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba += -1.00 * einsum('bd,jl,kmci,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00 * einsum('bc,jl,kmdi,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,bami->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbaaba +=  1.00 * einsum('bd,klce,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jl,mkce,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kmce,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbaaba += -1.00 * einsum('bc,klde,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bc,jl,mkde,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,kmde,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kmce,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,kmde,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbaaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_babbaaba


def get_doubles_doubles_babbaaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_babbaaab += -1.00 * einsum('bd,ik,lmcj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00 * einsum('bc,ik,lmdj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00 * einsum('bd,jk,lmci,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab += -1.00 * einsum('bc,jk,lmdi,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,bami->abjicdlk', kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbaaab += -1.00 * einsum('bd,lkce,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bd,jk,mlce,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,lmce,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbaaab +=  1.00 * einsum('bc,lkde,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bc,jk,mlde,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,lmde,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,lmce,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,lmde,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbaaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_babbaaab


def get_doubles_doubles_baaaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaaabaa += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('bc,ik,lajd->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,laid->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,bacd->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,lmjd,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lmid,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,bmcd,am->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jk,laed,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jl,kaed,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,maed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmcd,baim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,kmcd,baim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,nmcd,banm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,mncd,bamn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaaabaa += -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmce,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmce,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lmed,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kmed,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,nmed,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,mned,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaaabaa += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lmed,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kmed,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,mned,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaaabaa +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_baaaabaa


def get_doubles_doubles_babaabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,ki->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,jl,kaid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,blcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba += -1.00 * einsum('ik,jl,bacd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,bc,klie,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,jl,kmid,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,kmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ik,jl,bmcd,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,lade,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,jl,kaed,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,jl,maed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,blce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ik,mlcd,bamj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('jl,kmcd,baim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ik,jl,nmcd,banm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ik,jl,mncd,bamn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,klce,beij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,ik,mlce,bemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,kmce,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,kled,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,mled,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,mlde,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,jl,kmed,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('bc,ik,jl,nmed,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('bc,ik,jl,mned,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ad,bc,klfe,feij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,bc,ik,mlef,efmj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,bc,ik,mlfe,femj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ad,bc,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,bc,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  0.50 * einsum('ad,bc,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,ik,mlde,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('bc,jl,kmed,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,ik,jl,mned,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba += -1.00 * einsum('ad,bc,klfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabba +=  1.00 * einsum('ad,bc,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_babaabba


def get_doubles_doubles_babaabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,kj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,jk,ad->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,il,jk,md,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,bc,lkij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,kadj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,jk,laid->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,bkcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab +=  1.00 * einsum('il,jk,bacd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,mkdj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,bc,lkej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,bc,lkie,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,jk,lmid,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('il,jk,bmcd,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,jk,laed,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,kade,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,jk,maed,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,il,jk,made,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,bkce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,il,jk,bmce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('jk,lmcd,baim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('il,mkcd,bamj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('il,jk,nmcd,banm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('il,jk,mncd,bamn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,lkce,beij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,lmce,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,il,mkce,bemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,il,jk,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,il,jk,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,lked,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,jk,lmed,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,mked,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,mkde,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('bc,il,jk,nmed,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('bc,il,jk,mned,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('bc,il,jk,nmde,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('ad,bc,lkef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('ad,bc,lkfe,feij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,bc,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,bc,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,bc,il,mkef,efmj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -0.50 * einsum('ad,bc,il,mkfe,femj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  0.50 * einsum('ad,bc,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,jk,lmed,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('bc,il,mkde,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,il,jk,mned,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab +=  1.00 * einsum('ad,bc,lkfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babaabab += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_babaabab


def get_doubles_doubles_baababba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,kj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,il,jk,ad->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababba += -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,jk,md,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,bc,klji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,kajd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba += -1.00 * einsum('ad,jk,blci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba +=  1.00 * einsum('il,jk,bacd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,il,kmjd,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,bc,klje,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,bc,klei,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,kmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('il,jk,bmcd,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,kaed,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,il,jk,maed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,jk,made,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,jk,blce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,jk,bmce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('jk,mlcd,bami->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('il,kmcd,bajm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('il,jk,nmcd,banm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('il,jk,mncd,bamn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,klce,beji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,jk,mlce,bemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,mkce,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,kmce,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,il,jk,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,il,jk,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,kled,eaji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,mled,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,mlde,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,il,kmed,eajm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('bc,il,jk,nmed,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('bc,il,jk,mned,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('bc,il,jk,nmde,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('ad,bc,klef,efji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('ad,bc,klfe,feji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,bc,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,bc,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  0.50 * einsum('ad,bc,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,bc,il,kmef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -0.50 * einsum('ad,bc,il,kmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('bc,il,kmed,am,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,jk,mned,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('bc,il,jk,nmde,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababba += -1.00 * einsum('ad,bc,il,kmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_baababba


def get_doubles_doubles_baababab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,ki->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababab += -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,bc,lkji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,lajd->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,kadi->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,jl,bkci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab += -1.00 * einsum('ik,jl,bacd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,ik,lmjd,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,bc,lkje,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,bc,lkei,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,mkdi,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ik,jl,bmcd,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,laed,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,ik,jl,maed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,jl,bkce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ik,lmcd,bajm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('jl,mkcd,bami->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ik,jl,nmcd,banm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ik,jl,mncd,bamn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,lkce,beji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,mlce,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,lmce,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,jl,mkce,bemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,lked,eaji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,ik,lmed,eajm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,mked,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,mkde,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('bc,ik,jl,nmed,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('bc,ik,jl,mned,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ad,bc,lkef,efji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ad,bc,lkfe,feji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ad,bc,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,bc,ik,lmef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,bc,ik,lmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,bc,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  0.50 * einsum('ad,bc,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,ik,lmed,am,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,jl,mned,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab += -1.00 * einsum('ad,bc,lkef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,ik,lmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baababab +=  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_baababab


def get_doubles_doubles_babbabbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,ad->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,md,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbabbb += -1.00 * einsum('ad,bc,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,blcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,bacd->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,mldj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mldi,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,bmcd,am->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,ik,jl,maed,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,made,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,blce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,bkce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bmce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,bami->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,bami->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,nmcd,banm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ik,jl,mncd,bamn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmce,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnce,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbabbb += -1.00 * einsum('bc,klde,eaij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('bc,jk,mled,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mked,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,nmed,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,ik,jl,mned,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bc,ik,jl,nmde,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbabbb += -0.50 * einsum('ad,bc,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,bc,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,bc,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,mlde,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,mkde,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,mned,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,ik,jl,nmde,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbabbb +=  1.00 * einsum('ad,bc,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,bc,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,bc,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbabbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_babbabbb


def get_doubles_doubles_baaabaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabaaa +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('bd,ik,lajc->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laic->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,badc->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lmjc,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lmic,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,madc,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,bmdc,am->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kaec,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bmde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmdc,baim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,kmdc,baim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmdc,banm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,mndc,bamn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lmec,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kmec,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,nmec,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,mnec,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabaaa +=  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmde,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmde,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabaaa +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,nmdc,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lmec,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kmec,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mnec,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabaaa += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baaabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_baaabaaa


def get_doubles_doubles_babababa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babababa += -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa += -1.00 * einsum('bd,jl,kaic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,bldj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,kbdi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa +=  1.00 * einsum('ik,jl,badc->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,bd,klie,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,jl,kmic,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,kmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ik,jl,madc,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ik,jl,bmdc,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('bd,jl,kaec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,blde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,ik,jl,bmde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ik,mldc,bamj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('jl,kmdc,baim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ik,jl,nmdc,banm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ik,jl,mndc,bamn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('bd,klec,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,mlec,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,mlce,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,jl,kmec,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('bd,ik,jl,nmec,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('bd,ik,jl,mnec,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,klde,beij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,ik,mlde,bemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,kmde,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ac,bd,klfe,feij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,bd,ik,mlef,efmj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,bd,ik,mlfe,femj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa += -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ik,jl,nmdc,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,ik,mlce,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('bd,jl,kmec,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('bd,ik,jl,mnec,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa +=  1.00 * einsum('ac,bd,klfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babababa += -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_babababa


def get_doubles_doubles_bababaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,il,jk,bd->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,il,jk,mc,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,bd,lkij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,kacj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,jk,laic->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,bkdj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab += -1.00 * einsum('il,jk,badc->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,mkcj,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,bd,lkej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,bd,lkie,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,jk,lmic,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('il,jk,madc,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('il,jk,bmdc,am->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,jk,laec,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,jk,maec,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,bkde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,jk,mbde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,il,jk,bmde,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('jk,lmdc,baim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('il,mkdc,bamj->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('il,jk,nmdc,banm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('il,jk,mndc,bamn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,lkec,eaij->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,jk,lmec,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,mkec,eamj->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,mkce,eajm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('bd,il,jk,nmec,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('bd,il,jk,mnec,eamn->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('bd,il,jk,nmce,eanm->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,lkde,beij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,lmde,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,il,mkde,bemj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,il,jk,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,il,jk,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('ac,bd,lkef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('ac,bd,lkfe,feij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,bd,il,mkef,efmj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  0.50 * einsum('ac,bd,il,mkfe,femj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('il,jk,nmdc,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,jk,lmec,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('bd,il,mkce,am,ej->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,il,jk,mnec,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab += -1.00 * einsum('ac,bd,lkfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bababaab +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_bababaab


def get_doubles_doubles_baabbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,jk,bd->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,jk,mc,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,bd,klji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,kajc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,kbdj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,jk,bldi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba += -1.00 * einsum('il,jk,badc->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,il,kmjc,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,bd,klje,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,bd,klei,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,kmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('il,jk,madc,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('il,jk,bmdc,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,kaec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,il,jk,maec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,jk,blde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,kbde,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,il,jk,mbde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,jk,bmde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('jk,mldc,bami->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('il,kmdc,bajm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('il,jk,nmdc,banm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('il,jk,mndc,bamn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,klec,eaji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,mlec,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,mlce,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,il,kmec,eajm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('bd,il,jk,nmec,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('bd,il,jk,mnec,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('bd,il,jk,nmce,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,klde,beji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,jk,mlde,bemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,mkde,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,kmde,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,il,jk,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,il,jk,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('ac,bd,klef,efji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('ac,bd,klfe,feji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,bd,il,kmef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  0.50 * einsum('ac,bd,il,kmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('il,jk,nmdc,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('bd,il,kmec,am,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,jk,mnec,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('bd,il,jk,nmce,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaba +=  1.00 * einsum('ac,bd,il,kmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_baabbaba


def get_doubles_doubles_baabbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,bd,lkji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,lajc->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,jl,bkdi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ik,jl,badc->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,ik,lmjc,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,bd,lkje,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,bd,lkei,ej->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,mkci,am->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ik,jl,madc,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ik,jl,bmdc,am->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,laec,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,lbde,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,jl,bkde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,jl,bmde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ik,lmdc,bajm->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('jl,mkdc,bami->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ik,jl,nmdc,banm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ik,jl,mndc,bamn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,lkec,eaji->abjicdlk', kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,ik,lmec,eajm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,mkec,eami->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,mkce,eaim->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('bd,ik,jl,nmec,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('bd,ik,jl,mnec,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,lkde,beji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,mlde,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,lmde,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,jl,mkde,bemi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ac,bd,lkef,efji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ac,bd,lkfe,feji->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,bd,ik,lmef,efjm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,bd,ik,lmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ik,jl,nmdc,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,ik,lmec,am,ej->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,jl,mnec,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab +=  1.00 * einsum('ac,bd,lkef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,ik,lmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_baabbaab += -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    return doubles_doubles_baabbaab


def get_doubles_doubles_babbbabb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bd->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[va, va], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mc,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbbabb +=  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,bldj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,bldi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,badc->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, vb, va, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,mlcj,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlci,am->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,madc,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,bmdc,am->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,blde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,bkde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mbde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, va, va, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,jl,bmde,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mldc,bami->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,mkdc,bami->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmdc,banm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,mndc,bamn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbbabb +=  1.00 * einsum('bd,klce,eaij->abjicdlk', kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('bd,jk,mlec,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkec,eami->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,eaim->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,nmec,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('bd,ik,jl,mnec,eamn->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bd,ik,jl,nmce,eanm->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bemi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmde,benm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mnde,bemn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbbabb +=  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,nmdc,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,mlce,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,mkce,am,ei->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mnec,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,nmce,an,em->abjicdlk', kd_aa[va, va], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 4), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_babbbabb += -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_aa[va, va], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_babbbabb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_babbbabb


def get_doubles_doubles_baaabbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ad,ik,bljc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,blic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_baaabbba +=  1.00 * einsum('ad,ik,mljc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba += -1.00 * einsum('ac,ik,mljd,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba += -1.00 * einsum('ad,jk,mlic,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00 * einsum('ac,jk,mlid,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,blec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,baim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabbba +=  1.00 * einsum('ad,klec,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabbba += -1.00 * einsum('ac,kled,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mled,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_baaabbba


def get_doubles_doubles_baaabbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,il,bkjc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,bkic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_baaabbab += -1.00 * einsum('ad,il,mkjc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00 * einsum('ac,il,mkjd,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00 * einsum('ad,jl,mkic,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab += -1.00 * einsum('ac,jl,mkid,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,jl,bkec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,mkcd,baim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabbab += -1.00 * einsum('ad,lkec,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    doubles_doubles_baaabbab +=  1.00 * einsum('ac,lked,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,beim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mked,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baaabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    return doubles_doubles_baaabbab


def get_doubles_doubles_bababbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ad,jk,blic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlic,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlid,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,blec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,bkec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bababbbb +=  1.00 * einsum('jk,mlcd,baim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('jl,mkcd,baim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ad,klce,beij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ad,jk,mlec,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ad,jk,mlce,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ad,jl,mkec,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ad,jl,mkce,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ac,klde,beij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ac,jk,mled,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ac,jk,mlde,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ac,jl,mked,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ac,jl,mkde,beim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ad,jk,mlec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ad,jl,mkec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababbbb +=  1.00 * einsum('ac,jk,mled,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bababbbb += -1.00 * einsum('ac,jl,mked,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    return doubles_doubles_bababbbb


def get_doubles_doubles_baabbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ad,ik,bljc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mljc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mljd,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,blec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,bkec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[va, ob, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_baabbbbb += -1.00 * einsum('ik,mlcd,bajm->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('il,mkcd,bajm->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ad,klce,beji->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ad,ik,mlec,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ad,ik,mlce,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ad,il,mkec,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ad,il,mkce,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ac,klde,beji->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ac,ik,mled,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ac,ik,mlde,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ac,il,mked,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ac,il,mkde,bejm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ad,ik,mlec,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ad,il,mkec,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbbbb += -1.00 * einsum('ac,ik,mled,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_baabbbbb +=  1.00 * einsum('ac,il,mked,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    return doubles_doubles_baabbbbb


def get_doubles_doubles_bbbaabaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lmcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_bbbaabaa +=  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa += -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa += -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa += -1.00 * einsum('ik,lmcd,abjm->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00 * einsum('il,kmcd,abjm->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,mlce,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lmce,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,mkce,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,kmce,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lmce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,kmce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbaabaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbbaabaa


def get_doubles_doubles_bbababaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_bbababaa += -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa += -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00 * einsum('jk,lmcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbababaa += -1.00 * einsum('jl,kmcd,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,klce,ebji->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbababaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbababaa


def get_doubles_doubles_bbbbabba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_bbbbabba += -1.00 * einsum('bd,il,kacj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabba +=  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabba +=  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabba += -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,il,kmcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jl,kmcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,mkce,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kmce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbabba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbbbabba


def get_doubles_doubles_bbbbabab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_bbbbabab +=  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabab += -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabab += -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbabab +=  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,ik,lmcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,lmcd,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,lkce,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,mlce,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lmce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbabab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbbbabab


def get_doubles_doubles_bbbabaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,lmdj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_bbbabaaa += -1.00 * einsum('bc,ik,lade,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00 * einsum('bc,il,kade,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00 * einsum('ac,ik,lbde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa += -1.00 * einsum('ac,il,kbde,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00 * einsum('ik,lmdc,abjm->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabaaa += -1.00 * einsum('il,kmdc,abjm->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,mlde,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,lmde,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,mkde,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,kmde,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,lmde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,kmde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbbabaaa


def get_doubles_doubles_bbabbaaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmdi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    doubles_doubles_bbabbaaa +=  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa += -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa += -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa += -1.00 * einsum('jk,lmdc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00 * einsum('jl,kmdc,abim->abjicdlk', kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ac,klde,ebji->abjicdlk', kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbabbaaa


def get_doubles_doubles_bbbbbaba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_bbbbbaba +=  1.00 * einsum('bc,il,kadj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaba += -1.00 * einsum('bc,jl,kadi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaba += -1.00 * einsum('ac,il,kbdj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaba +=  1.00 * einsum('ac,jl,kbdi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ac,il,kmdj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmdi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,jl,kade,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kbde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,kmdc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,mkde,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jl,kmde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbaba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbbbbaba


def get_doubles_doubles_bbbbbaab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    doubles_doubles_bbbbbaab += -1.00 * einsum('bc,ik,ladj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaab +=  1.00 * einsum('bc,jk,ladi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaab +=  1.00 * einsum('ac,ik,lbdj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbaab += -1.00 * einsum('ac,jk,lbdi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, ob], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ac,ik,lmdj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmdi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,jk,lade,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lbde,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lmdc,abim->abjicdlk', kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,mlde,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jk,lmde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbaab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbbbbaab


def get_doubles_doubles_bbaabbaa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_aaaa[oa, oa, oa, oa], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,lajc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lbjc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbic->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lmjc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmjd,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmic,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmid,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kaec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kbec,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbaabbaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbaabbaa


def get_doubles_doubles_bbbabbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kaic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kbic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbbabbba +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klie,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmic,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmid,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmie,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lace,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jl,kaec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jl,kbec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbbabbba +=  1.00 * einsum('ik,mlcd,abjm->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,klec,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlec,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kled,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mled,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klfe,feij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,mlef,efmj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,mlfe,femj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,kmfe,feim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_bbbabbba += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,ik,mlce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kmec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mlde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,kmed,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,kmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbbabbba


def get_doubles_doubles_bbbabbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,kacj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kbcj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbic->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbbabbab += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,il,mkcj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkdj,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkej,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkie,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmic,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmid,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmie,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jk,laec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,kace,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,maec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jk,lbec,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kbce,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,mbec,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbbabbab += -1.00 * einsum('il,mkcd,abjm->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbab += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,lkec,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkec,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lked,ebij->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mked,ebmj->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,ebjm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,lkef,efij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,lkfe,feij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,lmfe,feim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,mkef,efmj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,mkfe,femj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_bbbabbab +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,lmec,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,mkce,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,lmed,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,mkde,bm,ej->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,lkfe,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,lmfe,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbabbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbbabbab


def get_doubles_doubles_bbabbbba(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,ke,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,kajc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,kbjc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbabbbba += -1.00 * einsum('il,jk,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate =  1.00 * einsum('ad,il,kmjc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,kmjd,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klje,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klei,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kmje,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('il,jk,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,kaec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,il,jk,maec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,il,jk,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,kbec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,jk,mbec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbabbbba += -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbba += -0.50 * einsum('il,jk,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,klec,ebji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlec,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kmec,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,il,jk,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,il,jk,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,kled,ebji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mled,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,kmed,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,il,jk,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,il,jk,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,klef,efji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,klfe,feji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,il,mkef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,kmef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,il,kmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_bbabbbba +=  1.00 * einsum('il,jk,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,il,kmec,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,il,jk,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,il,kmed,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,il,jk,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,mkef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,il,kmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbba +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbabbbba


def get_doubles_doubles_bbabbbab(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, oa], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ki->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,le,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_aa[oa, va], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,lajc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kaci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lbjc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, oa, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbci->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbabbbab +=  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    contracted_intermediate = -1.00 * einsum('ad,ik,lmjc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmjd,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkje,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkei,ej->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, oa], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmje,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, oa, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkci,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkdi,bm->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,laec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,lbec,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate) 
    doubles_doubles_bbabbbab +=  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('ad,lkec,ebji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lmec,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkec,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lked,ebji->abjicdlk', kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmed,ebjm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mked,ebmi->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,lkef,efji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,lkfe,feji->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,ik,mlef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,lmef,efjm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,ik,lmfe,fejm->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    doubles_doubles_bbabbbab += -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('ad,ik,lmec,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,lmed,bm,ej->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,lkef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlef,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lmfe,em,fj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_aa[oa, oa], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbabbbab +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    return doubles_doubles_bbabbbab


def get_doubles_doubles_bbbbbbbb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
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
    
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,lj->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,li->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, ob], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,ac->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,bc->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[vb, vb], optimize=['einsum_path', (0, 1, 2, 3)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mc,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,md,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,le,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,ke,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], f_bb[ob, vb], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_bbbb[ob, ob, ob, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,lacj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,laci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,lbcj->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbci->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, ob], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abjidckl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,abcd->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[vb, vb, vb, vb], optimize=['einsum_path', (0, 1, 2)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,mlcj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,ik,mldj,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,klej,ei->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,ik,mlej,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlci,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mldi,bm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, ob], t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlei,em->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, ob], t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abjicdkl', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->bajicdkl', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,jl,macd,bm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,jk,lace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,jl,kace,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bd,ik,jl,maec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bd,ik,jl,mace,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,lbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,kbce,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,ik,jl,mbec,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, vb, va, vb], t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mbce,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, vb, vb, vb], t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->abjidclk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->abijdclk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jl,mkcd,abim->abjicdlk', kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,jl,nmcd,abnm->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,klce,ebij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlec,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkec,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,nmec,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ad,ik,jl,mnec,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ad,ik,jl,nmce,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,klde,ebij->abjicdlk', kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mled,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mked,ebmi->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,ebim->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,nmed,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,ik,jl,mned,ebmn->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,ik,jl,nmde,ebnm->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,klef,efij->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlef,efmi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jk,mlfe,femi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jk,mlef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkef,efmi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,bd,jl,mkfe,femi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ac,bd,jl,mkef,efim->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,jl,nmcd,am,bn->abjicdlk', kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,jk,mlce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ad,jl,mkce,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,mnec,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ad,ik,jl,nmce,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,jk,mlde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,jl,mkde,bm,ei->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,mned,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,ik,jl,nmde,bn,em->abjicdlk', kd_bb[vb, vb], kd_bb[ob, ob], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,klef,ej,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,bd,jk,mlef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,bd,jl,mkef,em,fi->abjicdlk', kd_bb[vb, vb], kd_bb[vb, vb], kd_bb[ob, ob], g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (1, 3), (0, 2), (0, 1)])
    doubles_doubles_bbbbbbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjicdlk->abijcdlk', contracted_intermediate)  + -1.00000 * einsum('abjicdlk->bajicdlk', contracted_intermediate)  +  1.00000 * einsum('abjicdlk->baijcdlk', contracted_intermediate) 
    return doubles_doubles_bbbbbbbb
