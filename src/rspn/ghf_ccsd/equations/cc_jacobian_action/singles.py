from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE


def get_cc_j_w_singles(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
    vector: GHF_CCSD_MBE,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i') """
    r1 = vector.singles
    r2 = vector.doubles
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2

    cc_j_w_singles = -1.00 * einsum('ji,aj->ai', f[o, o], r1)
    cc_j_w_singles +=  1.00 * einsum('ab,bi->ai', f[v, v], r1)
    cc_j_w_singles += -1.00 * einsum('jb,aj,bi->ai', f[o, v], t1, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('jb,bi,aj->ai', f[o, v], t1, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('jb,baij->ai', f[o, v], r2)
    cc_j_w_singles +=  1.00 * einsum('jabi,bj->ai', g[o, v, v, o], r1)
    cc_j_w_singles +=  1.00 * einsum('kjbi,ak,bj->ai', g[o, o, v, o], t1, r1, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('kjbi,bk,aj->ai', g[o, o, v, o], t1, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles +=  1.00 * einsum('jabc,ci,bj->ai', g[o, v, v, v], t1, r1, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('jabc,cj,bi->ai', g[o, v, v, v], t1, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles +=  1.00 * einsum('kjbc,caik,bj->ai', g[o, o, v, v], t2, r1, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles +=  0.50 * einsum('kjbc,cakj,bi->ai', g[o, o, v, v], t2, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles +=  0.50 * einsum('kjbc,bcik,aj->ai', g[o, o, v, v], t2, r1, optimize=['einsum_path', (0, 1), (0, 1)])
    cc_j_w_singles +=  1.00 * einsum('kjbc,ak,ci,bj->ai', g[o, o, v, v], t1, t1, r1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('kjbc,ak,cj,bi->ai', g[o, o, v, v], t1, t1, r1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    cc_j_w_singles += -1.00 * einsum('kjbc,bk,ci,aj->ai', g[o, o, v, v], t1, t1, r1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    cc_j_w_singles += -0.50 * einsum('kjbi,bakj->ai', g[o, o, v, o], r2)
    cc_j_w_singles += -0.50 * einsum('jabc,bcij->ai', g[o, v, v, v], r2)
    cc_j_w_singles += -0.50 * einsum('kjbc,ak,bcij->ai', g[o, o, v, v], t1, r2, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles += -0.50 * einsum('kjbc,ci,bakj->ai', g[o, o, v, v], t1, r2, optimize=['einsum_path', (0, 2), (0, 1)])
    cc_j_w_singles +=  1.00 * einsum('kjbc,ck,baij->ai', g[o, o, v, v], t1, r2, optimize=['einsum_path', (0, 1), (0, 1)])
    return cc_j_w_singles
