from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_lhe1e1cc(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('a', 'i', 'b', 'j') """
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    lhe1e1cc = -1.00 * einsum('jiab->aibj', g[o, o, v, v])
    contracted_intermediate =  1.00 * einsum('jiak,kb->aibj', g[o, o, v, o], l1)
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('icab,jc->aibj', g[o, v, v, v], l1)
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    lhe1e1cc += -0.50 * einsum('jikl,klab->aibj', g[o, o, o, o], l2)
    lhe1e1cc +=  1.00 * einsum('jcak,ikcb->aibj', g[o, v, v, o], l2)
    lhe1e1cc +=  1.00 * einsum('icbk,jkca->aibj', g[o, v, v, o], l2)
    lhe1e1cc += -0.50 * einsum('dcab,jidc->aibj', g[v, v, v, v], l2)
    contracted_intermediate =  1.00 * einsum('kiab,ck,jc->aibj', g[o, o, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jiac,ck,kb->aibj', g[o, o, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc +=  1.00 * einsum('kjac,ck,ib->aibj', g[o, o, v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc +=  1.00 * einsum('kibc,ck,ja->aibj', g[o, o, v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('ljak,cl,ikbc->aibj', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('libk,cl,jkac->aibj', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('jicl,ck,klab->aibj', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('kdab,ck,jidc->aibj', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('jcad,dk,kicb->aibj', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('icbd,dk,kjca->aibj', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  0.50 * einsum('liab,dckl,kjdc->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->ajbi', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('jiad,dckl,klbc->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lhe1e1cc +=  1.00000 * contracted_intermediate + -1.00000 * einsum('aibj->biaj', contracted_intermediate) 
    lhe1e1cc += -0.250 * einsum('lkab,dclk,jidc->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc +=  1.00 * einsum('ljad,dckl,kibc->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc +=  1.00 * einsum('libd,dckl,kjac->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc += -0.250 * einsum('jicd,cdkl,klab->aibj', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lhe1e1cc +=  0.50 * einsum('lkab,cl,dk,jidc->aibj', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc +=  1.00 * einsum('ljad,cl,dk,kibc->aibj', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc +=  1.00 * einsum('libd,cl,dk,kjac->aibj', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc +=  0.50 * einsum('jicd,cl,dk,klab->aibj', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lhe1e1cc += -1.00 * einsum('ja,ib->aibj', f[o, v], l1)
    lhe1e1cc += -1.00 * einsum('ib,ja->aibj', f[o, v], l1)
    return lhe1e1cc
