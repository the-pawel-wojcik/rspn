import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR
import numpy as np

def view_the_jacobian():
    with open('pickles/uhf_ccsd.pkl','rb') as bak_file:
       ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()

    # for id, eval in enumerate(np.sort(cc_jacobian.diagonal())):
    #     print(f'{id:>3}: {eval:.4f}')

    # off_diag = cc_jacobian - np.diag(cc_jacobian.diagonal())
    # for id, eval in enumerate(np.sort(off_diag.flatten())):
    #     print(f'{id:>3}: {eval:.4f}')

    single_column = cc_jacobian[:, 4]
    for id, eval in enumerate(single_column):
        val = f'{eval:-8.3f}' if abs(eval) > 1e-3 else ' '*8
        print(f'{id:>3}: {val}')


def look_at_the_eigensystem():
    with open('pickles/uhf_ccsd.pkl','rb') as bak_file:
       ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
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
    with open('pickles/uhf_ccsd.pkl','rb') as bak_file:
       ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    with open('pickles/cc_jacobian.pkl','wb') as bak_file:
        pickle.dump(cc_jacobian, bak_file)


if __name__ == "__main__":
    view_the_jacobian()
