from chem.ccs.uhf_ccs import UHF_CCS
from chem.meta.coordinates import CARTESIAN
from rspn.uhf_ccs.uhf_ccs_lr import UHF_CCS_LR


def test_eta_works(uhf_ccs_water_ccpVDZ: UHF_CCS) -> None:
    ccs =  uhf_ccs_water_ccpVDZ
    lr = UHF_CCS_LR(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
        uhf_ccs_lambda_data=ccs.cc_lambda_data,
    )

    eta_mu = lr._find_eta_mu()
    assert set(eta_mu) == {coord for coord in CARTESIAN}
    for _, val in eta_mu.items():
        assert set(val) == {'aa', 'bb'}
        assert val['aa'].shape == (19, 5)
        assert val['bb'].shape == (19, 5)
