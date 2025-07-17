from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from chem.ccsd.uhf_ccsd import UHF_CCSD
import numpy as np
import pytest
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


# Broken
@pytest.mark.skip
def test_cc_diagonalization(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    ccsd = uhf_ccsd_water_sto3g

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
    # with np.printoptions(precision=3, suppress=True):
    #     print((inv_evecs @ cc_jacobian @ evecs).shape)
    #     print(evals.shape)

    assert np.allclose(
        inv_evecs @ cc_jacobian @ evecs,
        np.diag(evals),
        atol=1e-2,
    )
