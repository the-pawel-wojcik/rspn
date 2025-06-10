import pickle
import pytest

from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
import numpy as np


# takes a lot of resources and is not used in the rspn code
@pytest.mark.skip
def test_cc_diagonalization():
    with open('pickles/uhf_ccsd.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    cc_jacobian = build_cc_jacobian(
        kwargs=kwargs,
        dims=lr.assign_dims(),
    )
    evals, evecs = np.linalg.eig(cc_jacobian)
    # The CC Jacobian matrix is non-symmetric so a transpose, in general, does
    # not produce the inverse
    assert not np.allclose(evecs.T @ cc_jacobian @ evecs, np.diag(evals))
    # The inverse needs to be calculated
    inv_evecs = np.linalg.inv(evecs)
    assert np.allclose(inv_evecs @ cc_jacobian @ evecs, np.diag(evals))


if __name__ == "__main__":
    test_cc_diagonalization()
