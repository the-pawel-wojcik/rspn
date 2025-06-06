import pickle
from chem.ccsd.uhf_ccsd import UHF_CCSD
from rspn.uhf_ccsd.uhf_ccsd_lr import UHF_CCSD_LR


def humanify(size_bytes: float) -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti']:
        if abs(size_bytes) < 1024.0:
            return f'{size_bytes:.2f} {unit}B'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PiB'


def test_cc_jacobian_build():
    with open('pickles/uhf_ccsd_lambda.pkl','rb') as bak_file:
       ccsd: UHF_CCSD = pickle.load(bak_file)

    lr = UHF_CCSD_LR(ccsd.data, ccsd.scf_data)
    cc_jacobian = lr.build_the_cc_jacobian()
    print(f'CC Jacobian size = {humanify(cc_jacobian.nbytes)}.')
    print(f'{cc_jacobian.shape=}')


if __name__ == "__main__":
    test_cc_jacobian_build()
