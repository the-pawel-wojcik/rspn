from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import Descartes
from chem.meta.polarizability import Polarizability
from rspn.ghf_ccsd.ghf_ccsd_lr import GHF_CCSD_LR, GHF_CCSD_LR_config


def test_polarizabilities(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    lr_config = GHF_CCSD_LR_config(store_jacobian=True)
    lr = GHF_CCSD_LR(ccsd.ghf_data, ccsd.data, lr_config)
    pol_0_05_au = lr.find_dynamic_polarizabilities(omega=0.05)
    fmt = '=^50'
    print(f'{'ω = 0.05 a.u.':{fmt}}')
    print(pol_0_05_au)
    pol_0_05_au_psi4 = Polarizability({
        Descartes.x: {
            Descartes.x: -0.04636,
            Descartes.y: 0.0,
            Descartes.z: 0.0,
        },
        Descartes.y: {
            Descartes.x: 0.0,
            Descartes.y: -5.14862,
            Descartes.z: 0.0,
        },
        Descartes.z: {
            Descartes.x: 0.0,
            Descartes.y: 0.0,
            Descartes.z: -2.53719,
        },
    })
    assert pol_0_05_au.isclose(pol_0_05_au_psi4, 1e-5)

    pol_0_10_au = lr.find_dynamic_polarizabilities(omega=0.10)
    print(f'{'ω = 0.10 a.u.':{fmt}}')
    print(pol_0_10_au)
    pol_0_10_au_psi4 = Polarizability({
        Descartes.x: {
            Descartes.x: -0.04824,
            Descartes.y: 0.0,
            Descartes.z: 0.0,
        },
        Descartes.y: {
            Descartes.x: 0.0,
            Descartes.y: -5.21364,
            Descartes.z: 0.0,
        },
        Descartes.z: {
            Descartes.x: 0.0,
            Descartes.y: 0.0,
            Descartes.z: -2.56591,
        },
    })
    assert pol_0_10_au.isclose(pol_0_10_au_psi4, 1e-5)
