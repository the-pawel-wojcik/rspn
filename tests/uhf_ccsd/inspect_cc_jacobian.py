import pickle

from chem.ccsd.equations.util import GeneratorsInput
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR, UHF_CCSD_LR_config
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
import numpy as np


def jacobian_factory():
    with open('pickles/water_sto3g@HF.pkl', 'rb') as bak_file:
        ccsd: UHF_CCSD = pickle.load(bak_file)

    kwargs = GeneratorsInput(
        uhf_scf_data=ccsd.scf_data,
        uhf_ccsd_data=ccsd.data,
    )
    lr_config = UHF_CCSD_LR_config(BUILD_JACOBIAN=True)
    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data, lr_config)
    cc_jacobian = build_cc_jacobian(kwargs=kwargs, dims=lr.assign_dims())
    return cc_jacobian


def view_the_jacobian():
    cc_jacobian = jacobian_factory()

    # for id, eval in enumerate(np.sort(cc_jacobian.diagonal())):
    #     print(f'{id:>3}: {eval:.4f}')

    # off_diag = cc_jacobian - np.diag(cc_jacobian.diagonal())
    # for id, eval in enumerate(np.sort(off_diag.flatten())):
    #     print(f'{id:>3}: {eval:.4f}')

    print("Fifth column of the CC Jacobian:")
    single_column = cc_jacobian[:, 4]
    for id, eval in enumerate(single_column):
        val = f'{eval:-8.3f}' if abs(eval) > 1e-3 else ' ' * 8
        print(f'{id:>3}: {val}')


def look_at_the_eigensystem():
    cc_jacobian = jacobian_factory()
    evals, evecs = np.linalg.eig(cc_jacobian)

    sorting_indices = np.argsort(evals)
    evals = evals[sorting_indices]
    evecs = evecs[:, sorting_indices]

    index = 0
    vector = evecs[:, index]
    print(f'{evals[index]=}')
    before = np.linalg.norm(vector)
    print(f'{before=}')
    after = np.linalg.norm(cc_jacobian @ vector)
    print(f'{after=}')
    print(f'{vector.T @ cc_jacobian @ vector=}')


def save_cc_jacobian():
    cc_jacobian = jacobian_factory()
    with open('pickles/cc_jacobian_water_sto3G@HF.pkl', 'wb') as bak_file:
        pickle.dump(cc_jacobian, bak_file)


def show_part_of_jacobian():
    with open('pickles/cc_jacobian_water_sto3G@HF.pkl', 'rb') as bak_file:
        jacobian = pickle.load(bak_file)

    with np.printoptions(precision=3, suppress=True):

        print("aa - aa")
        aa_aa = jacobian[0:10, 0:10]
        print(aa_aa)

        print("aa - bb")
        aa_bb = jacobian[0:10, 10:20]
        print(aa_bb)

        for idx in range(10):
            print(f"aa - aaaa part {idx}")
            print(jacobian[0:10, 20 + 10 * idx: 20 + 10 * (idx + 1)])

        for idx in range(10):
            print(f"aa - aaaa part {idx}")
            print(jacobian[0:10, 20 + 10 * idx: 20 + 10 * (idx + 1)])


if __name__ == "__main__":
    # view_the_jacobian()
    show_part_of_jacobian()
    # save_cc_jacobian()
