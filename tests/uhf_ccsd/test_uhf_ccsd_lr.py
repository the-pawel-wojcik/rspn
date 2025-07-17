from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from chem.ccsd.uhf_ccsd import UHF_CCSD


def test_UHF_CCSD_LR(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    ccsd = uhf_ccsd_water_sto3g
    lr_config = UHF_CCSD_LR_config(store_jacobian=True)
    UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
