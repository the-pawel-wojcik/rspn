import itertools
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import GHF_Data
from chem.meta.coordinates import Descartes
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc


def print_doubles_as_psi4(doubles: NDArray, ghf_data: GHF_Data) -> None:
    assert len(doubles.shape) == 4
    no = ghf_data.no
    nv = ghf_data.nv
    assert doubles.shape == (nv, nv, no, no)

    print(' ' * 6, end='')
    for a, b in itertools.product(range(0, nv, 2), range(0, nv, 2)):
        print(f'  ({a}, {b})', end='')
    print()
    fmt = ' 6.4f'
    for i, j in itertools.product(range(0, no, 2), range(0, no, 2)):
        print(f'({i}, {j}) ', end='')
        for a, b in itertools.product(range(0, nv, 2), range(0, nv, 2)):
            print(f'{doubles[a][b][i][j]:{fmt}} ', end='')
        print()


def test_cc_mu(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ghf_data = ghf_ccsd_water_sto3g.ghf_data
    input = GHF_Generators_Input(
        ghf_data=ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    assert set(cced_interation_op) == {coord for coord in Descartes}

    PSI4_MU_BAR_IA = {
        Descartes.x: np.array([
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.065762610303468, 0.0, 0.0, 0.0,],
            [0.0, 0.065762610303468, 0.0, 0.0,],
        ]),
        Descartes.y: np.array([
            [0.0, 0.0, -0.055511540012045, 0.0,],
            [0.0, 0.0, 0.0, -0.055511540012045,],
            [0.0, 0.0, -0.109470745197504, 0.0,],
            [0.0, 0.0, 0.0, -0.109470745197504,],
            [-0.756949029804676, 0.0, 0.0, 0.0,],
            [0.0, -0.756949029804676, 0.0, 0.0,],
            [0.0, 0.0, 0.656689990154577, 0.0,],
            [0.0, 0.0, 0.0, 0.656689990154577,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
        ]),
        Descartes.z: np.array([
            [-0.045301907858314, 0.0, 0.0, 0.0,],
            [0.0, -0.045301907858314, 0.0, 0.0,],
            [-0.051186365046473, 0.0, 0.0, 0.0,],
            [0.0, -0.051186365046473, 0.0, 0.0,],
            [0.0, 0.0, -0.580477576777052, 0.0,],
            [0.0, 0.0, 0.0, -0.580477576777052,],
            [0.481036934775297, 0.0, 0.0, 0.0,],
            [0.0, 0.481036934775297, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 0.0,],
        ]),
    }

    PSI4_MU_BAR_IjAb = {
        Descartes.z: np.array([
            [
                [[-0.00122875, -0.00000000], [-0.00000000, -0.00060678],],
                [[-0.00138308, -0.00000000],  [0.00000000,  0.00112794],],
                [[0.00000000, -0.00311126], [-0.00174706,  0.00000000],],
                [[0.00294655,  0.00000000], [-0.00000000,  0.00341753],],
                [[0.00000000,  0.00000000],  [0.00000000,  0.00000000],],
            ],
            [
                [[-0.00138308,  0.00000000], [-0.00000000,  0.00112794],],
                [[0.00567631, -0.00000000], [-0.00000000, -0.01656360],],
                [[-0.00000000,  0.01204824],  [0.00968822, -0.00000000],],
                [[-0.00494186,  0.00000000],  [0.00000000, -0.04751553],],
                [[0.00000000, -0.00000000], [-0.00000000, -0.00000000],],
            ],
            [
                [[0.00000000, -0.00174706], [-0.00311126,  0.00000000],],
                [[-0.00000000,  0.00968822],  [0.01204824, -0.00000000],],
                [[-0.04131878,  0.00000000],  [0.00000000, -0.03021427],],
                [[-0.00000000,  0.03074188],  [0.04162006,  0.00000000],],
                [[-0.00000000,  0.00000000],  [0.00000000, -0.00000000],],
            ],
            [
                [[0.00294655, -0.00000000],  [0.00000000,  0.00341753],],
                [[-0.00494186,  0.00000000],  [0.00000000, -0.04751553],],
                [[-0.00000000,  0.04162006],  [0.03074188,  0.00000000],],
                [[-0.06014308, -0.00000000], [-0.00000000, -0.07114661],],
                [[-0.00000000,  0.00000000],  [0.00000000, -0.00000000],],
            ],
            [
                [[0.00000000,  0.00000000],  [0.00000000,  0.00000000],],
                [[0.00000000, -0.00000000], [-0.00000000, -0.00000000],],
                [[-0.00000000,  0.00000000],  [0.00000000, -0.00000000],],
                [[-0.00000000,  0.00000000],  [0.00000000, -0.00000000],],
                [[-0.05169415, -0.00000000], [-0.00000000, -0.01553448],],
            ],
        ]),
    }
    for direction, mubar in cced_interation_op.items():
        assert set(mubar) == {'singles', 'doubles', }

        assert mubar['singles'].shape == (4, 10)
        np.set_printoptions(precision=6, suppress=True)
        print()
        print(direction)
        print(f'{mubar['singles'].T}')
        print(f'{mubar['singles'].T.shape=}')
        print(PSI4_MU_BAR_IA[direction])
        print(f'{PSI4_MU_BAR_IA[direction].shape=}')

        assert np.allclose(
            mubar['singles'], PSI4_MU_BAR_IA[direction].T, atol=1e-8,
        )

        assert mubar['doubles'].shape == (4, 4, 10, 10)
        print()
        print(direction)
        print_doubles_as_psi4(mubar['doubles'], ghf_data)
