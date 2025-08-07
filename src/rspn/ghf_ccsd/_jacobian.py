from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd.equations.cc_jacobian.singles_singles import (
    get_cc_j_singles_singles,
)
from rspn.ghf_ccsd.equations.cc_jacobian.singles_doubles import (
    get_cc_j_singles_doubles,
)
from rspn.ghf_ccsd.equations.cc_jacobian.doubles_singles import (
    get_cc_j_doubles_singles
)
from rspn.ghf_ccsd.equations.cc_jacobian.doubles_doubles import (
    get_cc_j_doubles_doubles
)


def single_count_sd_and_ds(
    *,
    raw_sd: NDArray,
    raw_ds: NDArray,
    nv: int,
    no: int,
) -> tuple[NDArray, NDArray]:
    assert raw_sd.shape == (nv, no, nv, nv, no, no)
    assert raw_ds.shape == (nv, nv, no, no, nv, no)
    dim_s = nv * no
    dim_d = ((nv - 1) * nv // 2) * ((no - 1) * no // 2)
    sd = np.zeros(shape=(dim_s, dim_d))
    ds = np.zeros(shape=(dim_d, dim_s))
    abij = 0
    for a in range(0, nv):
        for b in range(a + 1, nv):
            for i in range(0, no):
                for j in range(i + 1, no):
                    ck = 0 
                    for c in range(0, nv):
                        for k in range(0, no):
                            sd[ck, abij] = raw_sd[c,k,a,b,i,j]
                            ds[abij, ck] = raw_ds[a,b,i,j,c,k]
                            ck += 1
                    abij += 1
    return sd, ds


def single_count_doubles(raw_dd: NDArray, *, nv: int, no: int) -> NDArray:
    assert raw_dd.shape == (nv, nv, no, no, nv, nv, no, no)
    dim = ((nv - 1) * nv // 2) * ((no - 1) * no // 2)
    dd = np.zeros(shape=(dim, dim))
    abij = 0
    for a in range(0, nv):
        for b in range(a + 1, nv):
            for i in range(0, no):
                for j in range(i + 1, no):
                    cdkl = 0 
                    for c in range(0, nv):
                        for d in range(c + 1, nv):
                            for k in range(0, no):
                                for l in range(k + 1, no):
                                    dd[abij,cdkl] = raw_dd[a,b,i,j,c,d,k,l]
                                    cdkl += 1
                    abij += 1
    return dd


def build_cc_jacobian(
    kwargs: GHF_Generators_Input,
):
    no = kwargs['ghf_data'].no
    nv = kwargs['ghf_data'].nv
    dim_s = nv * no
    singles_singles = get_cc_j_singles_singles(**kwargs).reshape(dim_s, dim_s)
    raw_sd = get_cc_j_singles_doubles(**kwargs)
    raw_ds = get_cc_j_doubles_singles(**kwargs)
    raw_dd = get_cc_j_doubles_doubles(**kwargs)

    # doubles_doubles = single_count_doubles(raw_dd, nv=nv, no=no)
    # singles_doubles, doubles_singles = single_count_sd_and_ds(
    #     raw_sd=raw_sd, raw_ds=raw_ds, nv=nv, no=no,
    # )
    dim_d = dim_s**2
    singles_doubles = raw_sd.reshape((dim_s, dim_d))
    doubles_singles = raw_ds.reshape((dim_d, dim_s))
    doubles_doubles = raw_dd.reshape((dim_d, dim_d))

    jacobian = np.block([
        [singles_singles, singles_doubles,],
        [doubles_singles, doubles_doubles,],
    ])
    return jacobian
