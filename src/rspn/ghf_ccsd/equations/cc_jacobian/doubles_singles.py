from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_doubles_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'b', 'j', 'i', 'c', 'k') """
    f = ghf_data.f
    g = ghf_data.g
    kd =  ghf_data.identity_singles
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('jk,lc,abil->abjick', kd[o, o], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles =  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,kd,dbij->abjick', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kaij->abjick', kd[v, v], g[o, v, o, o])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,abcj->abjick', kd[o, o], g[v, v, v, o])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkij,bl->abjick', kd[v, v], g[o, o, o, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ik,lacj,bl->abjick', kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('bc,kadj,di->abjick', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,abcd,di->abjick', kd[o, o], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcj,abil->abjick', g[o, o, v, o], t2)
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('ik,mlcj,abml->abjick', kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,dbil->abjick', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kacd,dbij->abjick', g[o, v, v, v], t2)
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,dbil->abjick', kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('bc,kade,deij->abjick', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,abil,dj->abjick', g[o, o, v, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,al,dbij->abjick', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,mlcd,abim,dl->abjick', kd[o, o], g[o, o, v, v], t2, t1, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('jk,mlcd,abml,di->abjick', kd[o, o], g[o, o, v, v], t2, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,dbim->abjick', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkde,ebij,dl->abjick', kd[v, v], g[o, o, v, v], t2, t1, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,ebil,dj->abjick', kd[v, v], g[o, o, v, v], t2, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ac,lkde,bl,deij->abjick', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ik,mlcj,al,bm->abjick', kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ac,lkdj,bl,di->abjick', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jk,lacd,bl,di->abjick', kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate)  + -1.00000 * einsum('abjick->bajick', contracted_intermediate)  +  1.00000 * einsum('abjick->baijck', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('bc,kade,dj,ei->abjick', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jk,mlcd,al,bm,di->abjick', kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->abijck', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,lkde,bl,dj,ei->abjick', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    cc_j_doubles_singles +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abjick->bajick', contracted_intermediate) 
    return cc_j_doubles_singles
