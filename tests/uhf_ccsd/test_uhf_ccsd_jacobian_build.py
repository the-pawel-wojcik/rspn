from chem.ccsd.uhf_ccsd import UHF_CCSD
import numpy as np
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from chem.ccsd.equations.util import GeneratorsInput
from rspn.uhf_ccsd._jacobian import (
    cc_jacobian_singles_singles,
    cc_jacobian_singles_doubles,
    cc_jacobian_doubles_singles,
    cc_jacobian_doubles_doubles,
)


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def test_cc_jacobian_build(uhf_ccsd_water_sto3g: UHF_CCSD) -> None:
    ccsd = uhf_ccsd_water_sto3g
    lr_config = UHF_CCSD_LR_config(store_jacobian=True)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    builders_input = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    singles_singles = cc_jacobian_singles_singles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    assert humanify(singles_singles.nbytes) == '3.12 KiB'
    assert singles_singles.shape == (20, 20)
    singles_doubles = cc_jacobian_singles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    assert humanify(singles_doubles.nbytes) == '93.75 KiB'
    assert singles_doubles.shape == (20, 600)
    doubles_singles = cc_jacobian_doubles_singles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    assert humanify(doubles_singles.nbytes) == '93.75 KiB'
    assert doubles_singles.shape == (600, 20)
    doubles_doubles = cc_jacobian_doubles_doubles(
        kwargs=builders_input,
        dims=lr.assign_dims(),
    )
    assert humanify(doubles_doubles.nbytes) == '2.75 MiB'
    assert doubles_doubles.shape == (600, 600)
    
    # The jacobian is not symmetric
    assert not np.allclose(
        np.zeros_like(singles_singles),
        singles_singles - singles_singles.T,
        atol=1e-6,
    )
