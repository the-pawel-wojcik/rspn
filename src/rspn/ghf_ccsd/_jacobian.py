from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
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
    # TODO: it is likely here
    no = kwargs['ghf_data'].no
    nv = kwargs['ghf_data'].nv
    dim_s = nv * no
    dim_d = nv * nv * no * no
    singles_singles = get_cc_j_singles_singles(**kwargs).reshape(dim_s, dim_s)
    singles_doubles = get_cc_j_singles_doubles(**kwargs).reshape(dim_s, dim_d)
    doubles_singles = get_cc_j_doubles_singles(**kwargs).reshape(dim_d, dim_s)
    doubles_doubles = get_cc_j_doubles_doubles(**kwargs).reshape(dim_d, dim_d)

    jacobian = np.block([
        [singles_singles, singles_doubles,],
        [doubles_singles, doubles_doubles,],
    ])
    return jacobian
