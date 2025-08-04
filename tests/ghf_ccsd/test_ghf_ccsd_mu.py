import itertools
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.ccsd.ghf_ccsd import GHF_CCSD
from chem.hf.ghf_data import GHF_Data
from chem.meta.coordinates import Descartes
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc
from psi4_data import PSI4_MU_BAR_IjAb, PSI4_MU_BAR_IA, psi4_rhf_doubles_to_ghf


def print_ghf_doubles(doubles: NDArray, ghf_data: GHF_Data) -> None:
    assert len(doubles.shape) == 4
    no = ghf_data.no
    nv = ghf_data.nv
    assert doubles.shape == (nv, nv, no, no)

    pad = ' '
    fmt = ' 6.3f'
    print(r'[')  # ]
    for a, cube in enumerate(doubles):
        print(f'{pad}[ {a=}')  # ]
        for b, wall in enumerate(cube):
            print(f'{pad*2}[ {b=}')  # ]
            print(f'{pad*3}[  j=', end='')  # ]
            for j, _ in enumerate(wall[0]):
                print(f'{j:^6d}', end='')
            print(']')
            for i, row in enumerate(wall):
                print(f'{pad*3}[ {i=}', end='')  # ]
                for value in row:
                    print(f'{value:{fmt}}', end='')
                print('],')
            print(f'{pad*2}],')
        print(f'{pad}],')
    print(']')


def print_ghf_doubles_as_psi4(doubles: NDArray, ghf_data: GHF_Data) -> None:
    assert len(doubles.shape) == 4
    no = ghf_data.no
    nv = ghf_data.nv
    assert doubles.shape == (nv, nv, no, no)

    print(' ' * 6, end='')
    for a, b in itertools.product(range(0, nv), range(0, nv)):
        print(f'  ({a}, {b})', end='')
    print()
    fmt = ' 6.4f'
    for i, j in itertools.product(range(0, no), range(0, no)):
        print(f'({i}, {j}) ', end='')
        for a, b in itertools.product(range(0, nv), range(0, nv)):
            print(f'{doubles[a, b, i, j]:{fmt}} ', end='')
        print()


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
            print(f'{doubles[a, b, i, j]:{fmt}} ', end='')
        print()


def print_psi4_doubles(doubles: NDArray, ghf_data: GHF_Data) -> None:
    assert len(doubles.shape) == 4
    # Psi4's data comes from RHF so the GHF dimensions need to be cut in half
    no = ghf_data.no // 2
    nv = ghf_data.nv // 2
    assert doubles.shape == (no, no, nv, nv)

    print("Psi4 doubles")
    print(' ' * 6, end='')
    for a, b in itertools.product(range(nv), range(nv)):
        print(f'  ({a}, {b})', end='')
    print()
    fmt = ' 6.4f'
    for i, j in itertools.product(range(no), range(no)):
        print(f'({i}, {j}) ', end='')
        for a, b in itertools.product(range(nv), range(nv)):
            print(f'{doubles[i, j, a, b]:{fmt}} ', end='')
        print()


def test_cc_mu(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ghf_data = ghf_ccsd_water_sto3g.ghf_data
    input = GHF_Generators_Input(
        ghf_data=ghf_data,
        ghf_ccsd_data=ghf_ccsd_water_sto3g.data,
    )
    cced_interation_op = build_nu_bar_V_cc(input=input)
    assert set(cced_interation_op) == {coord for coord in Descartes}

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
        print(f"Psi4's doubles {direction}")
        print_psi4_doubles(PSI4_MU_BAR_IjAb[direction], ghf_data)
        print(f"Paweł's doubles {direction}")
        print_doubles_as_psi4(mubar['doubles'], ghf_data)
        print_ghf_doubles_as_psi4(mubar['doubles'], ghf_data)
        print(f"Paweł's doubles {direction}")
        print_ghf_doubles(mubar['doubles'], ghf_data)
        ghf_psi4_mu = psi4_rhf_doubles_to_ghf(
            PSI4_MU_BAR_IjAb[direction], ghf_data
        )
        # print_doubles_as_psi4(ghf_psi4_mu, ghf_data)
        # TODO: here is another mismatch
        assert not np.allclose(ghf_psi4_mu,  mubar['doubles'], atol=1e-8)
        break
