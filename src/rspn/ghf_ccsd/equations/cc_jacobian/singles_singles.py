from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_cc_j_singles_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f = ghf_data.f
    g = ghf_data.g
    kd =  ghf_data.identity_singles
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    cc_j_singles_singles = -1.00 * einsum('ab,ji->aibj', kd[v, v], f[o, o])
    cc_j_singles_singles +=  1.00 * einsum('ij,ab->aibj', kd[o, o], f[v, v])
    cc_j_singles_singles += -1.00 * einsum('ij,kb,ak->aibj', kd[o, o], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles += -1.00 * einsum('ab,jc,ci->aibj', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles +=  1.00 * einsum('jabi->aibj', g[o, v, v, o])
    cc_j_singles_singles +=  1.00 * einsum('kjbi,ak->aibj', g[o, o, v, o], t1)
    cc_j_singles_singles += -1.00 * einsum('ab,kjci,ck->aibj', kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles +=  1.00 * einsum('jabc,ci->aibj', g[o, v, v, v], t1)
    cc_j_singles_singles += -1.00 * einsum('ij,kabc,ck->aibj', kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles +=  1.00 * einsum('kjbc,caik->aibj', g[o, o, v, v], t2)
    cc_j_singles_singles +=  0.50 * einsum('ij,lkbc,calk->aibj', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles +=  0.50 * einsum('ab,kjcd,cdik->aibj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    cc_j_singles_singles +=  1.00 * einsum('kjbc,ak,ci->aibj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_singles_singles += -1.00 * einsum('ij,lkbc,al,ck->aibj', kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    cc_j_singles_singles += -1.00 * einsum('ab,kjcd,ck,di->aibj', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    return cc_j_singles_singles
