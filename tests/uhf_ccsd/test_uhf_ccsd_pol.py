from chem.ccsd.uhf_ccsd import UHF_CCSD
import pytest
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config


@pytest.mark.parametrize(
    argnames='lr_config',
    argvalues=(
        pytest.param(
            UHF_CCSD_LR_config(store_jacobian=True),
            id='store_jacobian',
        ),
        pytest.param(
            UHF_CCSD_LR_config(store_jacobian=False),
            id='no_store_jacobian',
        ),
    ),
)
def test_polarizabilities(
    lr_config: UHF_CCSD_LR_config,
    uhf_ccsd_water_sto3g: UHF_CCSD,
) -> None:
    ccsd = uhf_ccsd_water_sto3g
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    polarizability = lr.find_polarizabilities()
    fmt = '=^50'
    print(f'{' Polarizability ':{fmt}}')
    print(polarizability)
