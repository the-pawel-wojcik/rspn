""" Transform the HF dipole operator matrix elements to the CC dipole operators
matrix elements. """
import pdaggerq
from rspn.uhf_ccsd.equations.printer import (
    DefineSections,
    print_to_numpy,
    print_imports,
)
from chem.meta.coordinates import CARTESIAN


def singles():
    print_imports()
    for component in CARTESIAN:
        pq = pdaggerq.pq_helper('fermi')
        pq.add_st_operator(1.0, ['a*(i)', 'a(a)', 'h'], ['t1', 't2'])
        pq.simplify()
        extra_definitions = (
            f'h_aa = uhf_scf_data.mua_{component}',
            f'h_bb = uhf_scf_data.mub_{component}',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}',
            tensor_subscripts=('a', 'i'),
            defines_exclude={
                DefineSections.IDENTITY,
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
                DefineSections.LAMBDA_AMPS,
            },
            extra_definitions=extra_definitions,
        )


def doubles():
    print_imports()
    for component in CARTESIAN:
        pq = pdaggerq.pq_helper('fermi')
        pq.add_st_operator(1.0, ['e2(i,j,b,a)', 'h'], ['t1', 't2'])
        pq.simplify()
        extra_definitions = (
            f'h_aa = uhf_scf_data.mua_{component}',
            f'h_bb = uhf_scf_data.mub_{component}',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}',
            tensor_subscripts=('a', 'b', 'j', 'i'),
            defines_exclude={
                DefineSections.IDENTITY,
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
                DefineSections.LAMBDA_AMPS,
            },
            extra_definitions=extra_definitions,
        )


if __name__ == "__main__":
    singles_time = False
    doubles_time = True

    if singles_time:
        singles()

    elif doubles_time:
        doubles()
