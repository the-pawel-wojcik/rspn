from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config
from chem.ccsd.ghf_ccsd import GHF_CCSD


def test_constructor(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    GHF_CCSD_LR(
        ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_water_sto3g.data,
        lr_config
    )


def test_polarizabilities(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(
        ghf_ccsd_water_sto3g.ghf_data,
        ghf_ccsd_water_sto3g.data,
        lr_config
    )
    polarizability = lr.find_polarizabilities()
    fmt = '=^50'
    print(f'{'η^A X^B':{fmt}}')
    print(polarizability)
