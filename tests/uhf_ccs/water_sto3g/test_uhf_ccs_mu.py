import pickle

from chem.ccs.equations.util import UHF_CCS_InputPair
from chem.ccs.uhf_ccs import UHF_CCS
from chem.meta.coordinates import CARTESIAN, Descartes
import numpy as np
from rspn.uhf_ccs._nuOpCC import build_nu_bar_V_cc


def test_cc_mu():
    with open('../pickles/water_uhf_ccs_lambda_sto3g.pkl', 'rb') as bak_file:
        ccs: UHF_CCS = pickle.load(bak_file)
    input=UHF_CCS_InputPair(
        uhf_data=ccs.scf_data,
        uhf_ccs_data=ccs.data,
    )
    cc_interation_operator = build_nu_bar_V_cc(input=input)
    # TODO: Test the values
    assert set(cc_interation_operator) == {coord for coord in CARTESIAN}
    for _, val in cc_interation_operator.items():
        assert set(val) == {'aa', 'bb'}
        assert val['aa'].shape == (2, 5)
        assert val['bb'].shape == (2, 5)

    uhf_data = ccs.scf_data
    uhf_data.mua_x
    uhf_dipole_operator = {
        Descartes.x: {
            'aa': uhf_data.mua_x,
            'bb': uhf_data.mub_x,
        },
        Descartes.y: {
            'aa': uhf_data.mua_y,
            'bb': uhf_data.mub_y,
        },
        Descartes.z: {
            'aa': uhf_data.mua_z,
            'bb': uhf_data.mub_z,
        },
    }

    oa = uhf_data.oa
    va = uhf_data.va
    ob = uhf_data.ob
    vb = uhf_data.vb
    for cart, matrix in cc_interation_operator.items():
        assert np.allclose(
            matrix['aa'],
            uhf_dipole_operator[cart]['aa'][va, oa]
        )
        assert np.allclose(
            matrix['bb'],
            uhf_dipole_operator[cart]['bb'][vb, ob]
        )

    with np.printoptions(precision=3, suppress=True):
        for cart, matrix in cc_interation_operator.items():
            print(matrix['aa'])
            print(uhf_dipole_operator[cart]['aa'])


if __name__ == "__main__":
    test_cc_mu()
