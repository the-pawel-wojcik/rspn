import pickle
import pytest

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config


@pytest.mark.parametrize(
    argnames='lr_config',
    argvalues=(
        pytest.param(
            UHF_CCSD_LR_config(store_jacobian=True),
            id='store_jacobian',
        ),
        # pytest.param(
        #     UHF_CCSD_LR_config(store_jacobian=False),
        #     id='no_store_jacobian',
        # ),
    ),
)
def test_polarizabilities(lr_config: UHF_CCSD_LR_config):
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    polarizability = lr.find_polarizabilities()
    fmt = '=^50'
    print(f'{' Polarizability ':{fmt}}')
    print(polarizability)


if __name__ == "__main__":
    lr_config = UHF_CCSD_LR_config(store_jacobian=False)
    test_polarizabilities(lr_config)
