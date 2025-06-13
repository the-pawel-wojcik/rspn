r""" The perturbation operator needed as given by Eq. (60) from Ref. [1]
$$
\xi _{\nu} (\omega) 
=
\bra{0} \tau _{\nu} e ^{-T} V ^{\omega} e ^{T} \ket{0}
$$

The version of this prorgam at the time of writing this comment works only with
the electric dipole moment operator.

Refs: 
[1] H. Koch and P. Jørgensen, Coupled cluster response functions, The Journal of
Chemical Physics 93, 3333 (1990).
"""
import pdaggerq
from rspn.uhf_ccs.equations.printer import (
    DefineSections,
    print_to_numpy,
    print_imports,
)
from chem.meta.coordinates import CARTESIAN


def singles():
    defines_exclude={
        DefineSections.IDENTITY,
        DefineSections.FOCK,
        DefineSections.FLUCTUATION,
        DefineSections.LAMBDA_AMPS,
    }
    print_imports(defines_exclude=defines_exclude)
    for component in CARTESIAN:
        pq = pdaggerq.pq_helper('fermi')
        pq.add_st_operator(1.0, ['a*(i)', 'a(a)', 'h'], ['t1'])
        pq.simplify()
        extra_definitions = (
            f'h_aa = uhf_data.mua_{component}',
            f'h_bb = uhf_data.mub_{component}',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}',
            tensor_subscripts=('a', 'i'),
            defines_exclude=defines_exclude,
            extra_definitions=extra_definitions,
        )


if __name__ == "__main__":
    singles_time = True

    if singles_time:
        singles()
