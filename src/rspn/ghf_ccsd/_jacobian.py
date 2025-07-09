from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from rspn.ghf_ccsd.equations.cc_jacobian.ref_singles import (
    get_cc_j_ref_singles,
)
from rspn.ghf_ccsd.equations.cc_jacobian.ref_doubles import (
    get_cc_j_ref_doubles,
)
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


def build_cc_jacobian(
    kwargs: GHF_Generators_Input,
):
    no = kwargs['ghf_data'].no
    nv = kwargs['ghf_data'].nv
    dim_r = 1
    dim_s = nv * no
    dim_d = nv * nv * no * no
    ref_ref = np.zeros(shape=(dim_r, dim_r))
    ref_singles = get_cc_j_ref_singles(**kwargs).reshape(dim_r, dim_s)
    ref_doubles = get_cc_j_ref_doubles(**kwargs).reshape(dim_r, dim_d)
    singles_ref = np.zeros(shape=(dim_s, dim_r))
    singles_singles = get_cc_j_singles_singles(**kwargs).reshape(dim_s, dim_s)
    singles_doubles = get_cc_j_singles_doubles(**kwargs).reshape(dim_s, dim_d)
    doubles_ref = np.zeros(shape=(dim_d, dim_r))
    doubles_singles = get_cc_j_doubles_singles(**kwargs).reshape(dim_d, dim_s)
    doubles_doubles = get_cc_j_doubles_doubles(**kwargs).reshape(dim_d, dim_d)

    jacobian = np.block([
        [ref_ref, ref_singles, ref_doubles,],
        [singles_ref, singles_singles, singles_doubles,],
        [doubles_ref, doubles_singles, doubles_doubles,],
    ])
    return jacobian
