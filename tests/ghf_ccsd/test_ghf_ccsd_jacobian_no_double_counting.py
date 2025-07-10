import pickle

from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
import numpy as np
from rspn.ghf_ccsd.ghf_ccsd_lr import build_cc_jacobian


def test_cc_jacobian_spectrum():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: GHF_CCSD = pickle.load(bak_file)

    builders_input = GHF_Generators_Input(
        ghf_data=ccsd.ghf_data,
        ghf_ccsd_data=ccsd.data,
    )

    cc_jacobian = build_cc_jacobian(builders_input)
    print(f'{cc_jacobian.shape=}')
    eigenvalues = np.linalg.eigvals(cc_jacobian)
    for eval in sorted(eigenvalues, key=lambda z: abs(z)):
        print(f'{eval.real:+12.6f}', end='')
        if abs(eval.imag) > 1e-6:
            print(f'{eval.real:+12.6f}i')
        else:
            print('')
