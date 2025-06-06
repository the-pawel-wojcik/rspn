from chem.ccsd.equations.util import GeneratorsInput
import numpy as np
from numpy.typing import NDArray
from rspn.uhf_ccsd.equations.cc_jacobian.singles_singles import (
    get_cc_j_singles_singles_aaaa,
    get_cc_j_singles_singles_aabb,
    get_cc_j_singles_singles_bbaa,
    get_cc_j_singles_singles_bbbb,
)
from rspn.uhf_ccsd.equations.cc_jacobian.singles_doubles import (
    get_cc_j_singles_doubles_aaaaaa,
    get_cc_j_singles_doubles_aaabab,
    get_cc_j_singles_doubles_aaabba,
    get_cc_j_singles_doubles_aabaab,
    get_cc_j_singles_doubles_aababa,
    get_cc_j_singles_doubles_bbabab,
    get_cc_j_singles_doubles_bbabba,
    get_cc_j_singles_doubles_bbbaab,
    get_cc_j_singles_doubles_bbbaba,
    get_cc_j_singles_doubles_bbbbbb,
)
from rspn.uhf_ccsd.equations.cc_jacobian.doubles_singles import (
    get_cc_j_doubles_singles_aaaaaa,
    get_cc_j_doubles_singles_ababaa,
    get_cc_j_doubles_singles_abbaaa,
    get_cc_j_doubles_singles_baabaa,
    get_cc_j_doubles_singles_babaaa,
    get_cc_j_doubles_singles_bbbbaa,
    get_cc_j_doubles_singles_aaaabb,
    get_cc_j_doubles_singles_ababbb,
    get_cc_j_doubles_singles_abbabb,
    get_cc_j_doubles_singles_baabbb,
    get_cc_j_doubles_singles_bababb,
    get_cc_j_doubles_singles_bbbbbb,
)
from rspn.uhf_ccsd.equations.cc_jacobian.doubles_doubles import (
    get_cc_j_doubles_doubles_aaaaaaaa,
    get_cc_j_doubles_doubles_aaaaabab,
    get_cc_j_doubles_doubles_aaaaabba,
    get_cc_j_doubles_doubles_aaaabaab,
    get_cc_j_doubles_doubles_aaaababa,
    get_cc_j_doubles_doubles_ababaaaa,
    get_cc_j_doubles_doubles_abababab,
    get_cc_j_doubles_doubles_abababba,
    get_cc_j_doubles_doubles_ababbaab,
    get_cc_j_doubles_doubles_ababbaba,
    get_cc_j_doubles_doubles_ababbbbb,
    get_cc_j_doubles_doubles_abbaaaaa,
    get_cc_j_doubles_doubles_abbaabab,
    get_cc_j_doubles_doubles_abbaabba,
    get_cc_j_doubles_doubles_abbabaab,
    get_cc_j_doubles_doubles_abbababa,
    get_cc_j_doubles_doubles_abbabbbb,
    get_cc_j_doubles_doubles_baabaaaa,
    get_cc_j_doubles_doubles_baababab,
    get_cc_j_doubles_doubles_baababba,
    get_cc_j_doubles_doubles_baabbaab,
    get_cc_j_doubles_doubles_baabbaba,
    get_cc_j_doubles_doubles_baabbbbb,
    get_cc_j_doubles_doubles_babaaaaa,
    get_cc_j_doubles_doubles_babaabab,
    get_cc_j_doubles_doubles_babaabba,
    get_cc_j_doubles_doubles_bababaab,
    get_cc_j_doubles_doubles_babababa,
    get_cc_j_doubles_doubles_bababbbb,
    get_cc_j_doubles_doubles_bbbbabab,
    get_cc_j_doubles_doubles_bbbbabba,
    get_cc_j_doubles_doubles_bbbbbaab,
    get_cc_j_doubles_doubles_bbbbbaba,
    get_cc_j_doubles_doubles_bbbbbbbb,
)


def build_cc_jacobian(
    kwargs: GeneratorsInput,
    dims: dict[str, int],
):
    singles_singles = cc_jacobian_singles_singles(kwargs=kwargs, dims=dims)
    singles_doubles = cc_jacobian_singles_doubles(kwargs=kwargs, dims=dims)
    doubles_singles = cc_jacobian_doubles_singles(kwargs=kwargs, dims=dims)
    doubles_doubles = cc_jacobian_doubles_doubles(kwargs=kwargs, dims=dims)

    jacobian = np.block([
        singles_singles, singles_doubles,
        doubles_singles, doubles_doubles,
    ])
    return jacobian


def cc_jacobian_singles_singles(
    kwargs: GeneratorsInput,
    dims: dict[str, int],
) -> NDArray:
    dim_aa = dims['aa']
    dim_bb = dims['bb']
    aa_aa = get_cc_j_singles_singles_aaaa(**kwargs).reshape(dim_aa, dim_aa)
    aa_bb = get_cc_j_singles_singles_aabb(**kwargs).reshape(dim_aa, dim_bb)
    bb_aa = get_cc_j_singles_singles_bbaa(**kwargs).reshape(dim_bb, dim_aa)
    bb_bb = get_cc_j_singles_singles_bbbb(**kwargs).reshape(dim_bb, dim_bb)

    jacobian_singles_singles = np.block([
        [aa_aa, aa_bb,],
        [bb_aa, bb_bb,],
    ])

    return jacobian_singles_singles


def cc_jacobian_singles_doubles(
    kwargs: GeneratorsInput,
    dims: dict[str, int],
) -> NDArray:

    aa_aaaa = get_cc_j_singles_doubles_aaaaaa(**kwargs).reshape(dims['aa'], dims['aaaa'])
    aa_abab = get_cc_j_singles_doubles_aaabab(**kwargs).reshape(dims['aa'], dims['abab'])
    aa_abba = get_cc_j_singles_doubles_aaabba(**kwargs).reshape(dims['aa'], dims['abba'])
    aa_baab = get_cc_j_singles_doubles_aabaab(**kwargs).reshape(dims['aa'], dims['baab'])
    aa_baba = get_cc_j_singles_doubles_aababa(**kwargs).reshape(dims['aa'], dims['baba'])
    aa_bbbb = np.zeros(shape=(dims['aa'], dims['bbbb']))
    bb_aaaa = np.zeros(shape=(dims['bb'], dims['aaaa']))
    bb_abab = get_cc_j_singles_doubles_bbabab(**kwargs).reshape(dims['bb'], dims['abab'])
    bb_abba = get_cc_j_singles_doubles_bbabba(**kwargs).reshape(dims['bb'], dims['abba'])
    bb_baab = get_cc_j_singles_doubles_bbbaab(**kwargs).reshape(dims['bb'], dims['baab'])
    bb_baba = get_cc_j_singles_doubles_bbbaba(**kwargs).reshape(dims['bb'], dims['baba'])
    bb_bbbb = get_cc_j_singles_doubles_bbbbbb(**kwargs).reshape(dims['bb'], dims['bbbb'])

    singles_doubles = np.block([
        [aa_aaaa, aa_abab, aa_abba, aa_baab, aa_baba, aa_bbbb,],
        [bb_aaaa, bb_abab, bb_abba, bb_baab, bb_baba, bb_bbbb,],
    ])

    return singles_doubles


def cc_jacobian_doubles_singles(
    kwargs: GeneratorsInput,
    dims: dict[str, int],
) -> NDArray:
    aaaa_aa = get_cc_j_doubles_singles_aaaaaa(**kwargs).reshape(dims['aaaa'], dims['aa'])
    abab_aa = get_cc_j_doubles_singles_ababaa(**kwargs).reshape(dims['abab'], dims['aa'])
    abba_aa = get_cc_j_doubles_singles_abbaaa(**kwargs).reshape(dims['abba'], dims['aa'])
    baab_aa = get_cc_j_doubles_singles_baabaa(**kwargs).reshape(dims['baab'], dims['aa'])
    baba_aa = get_cc_j_doubles_singles_babaaa(**kwargs).reshape(dims['baba'], dims['aa'])
    bbbb_aa = get_cc_j_doubles_singles_bbbbaa(**kwargs).reshape(dims['bbbb'], dims['aa'])

    aaaa_bb = get_cc_j_doubles_singles_aaaabb(**kwargs).reshape(dims['aaaa'], dims['bb'])
    abab_bb = get_cc_j_doubles_singles_ababbb(**kwargs).reshape(dims['abab'], dims['bb'])
    abba_bb = get_cc_j_doubles_singles_abbabb(**kwargs).reshape(dims['abba'], dims['bb'])
    baab_bb = get_cc_j_doubles_singles_baabbb(**kwargs).reshape(dims['baab'], dims['bb'])
    baba_bb = get_cc_j_doubles_singles_bababb(**kwargs).reshape(dims['baba'], dims['bb'])
    bbbb_bb = get_cc_j_doubles_singles_bbbbbb(**kwargs).reshape(dims['bbbb'], dims['bb'])

    doubles_singles = np.block([
        [aaaa_aa, aaaa_bb],
        [abab_aa, abab_bb],
        [abba_aa, abba_bb],
        [baab_aa, baab_bb],
        [baba_aa, baba_bb],
        [bbbb_aa, bbbb_bb],
    ])
    return doubles_singles


def cc_jacobian_doubles_doubles(
    kwargs: GeneratorsInput,
    dims: dict[str, int],
) -> NDArray:
    aaaa_aaaa = get_cc_j_doubles_doubles_aaaaaaaa(**kwargs).reshape(dims['aaaa'], dims['aaaa'])
    aaaa_abab = get_cc_j_doubles_doubles_aaaaabab(**kwargs).reshape(dims['aaaa'], dims['abab'])
    aaaa_abba = get_cc_j_doubles_doubles_aaaaabba(**kwargs).reshape(dims['aaaa'], dims['abba'])
    aaaa_baab = get_cc_j_doubles_doubles_aaaabaab(**kwargs).reshape(dims['aaaa'], dims['baab'])
    aaaa_baba = get_cc_j_doubles_doubles_aaaababa(**kwargs).reshape(dims['aaaa'], dims['baba'])
    aaaa_bbbb = np.zeros(shape=(dims['aaaa'], dims['bbbb']))
    abab_aaaa = get_cc_j_doubles_doubles_ababaaaa(**kwargs).reshape(dims['abab'], dims['aaaa'])
    abab_abab = get_cc_j_doubles_doubles_abababab(**kwargs).reshape(dims['abab'], dims['abab'])
    abab_abba = get_cc_j_doubles_doubles_abababba(**kwargs).reshape(dims['abab'], dims['abba'])
    abab_baab = get_cc_j_doubles_doubles_ababbaab(**kwargs).reshape(dims['abab'], dims['baab'])
    abab_baba = get_cc_j_doubles_doubles_ababbaba(**kwargs).reshape(dims['abab'], dims['baba'])
    abab_bbbb = get_cc_j_doubles_doubles_ababbbbb(**kwargs).reshape(dims['abab'], dims['bbbb'])
    abba_aaaa = get_cc_j_doubles_doubles_abbaaaaa(**kwargs).reshape(dims['abba'], dims['aaaa'])
    abba_abab = get_cc_j_doubles_doubles_abbaabab(**kwargs).reshape(dims['abba'], dims['abab'])
    abba_abba = get_cc_j_doubles_doubles_abbaabba(**kwargs).reshape(dims['abba'], dims['abba'])
    abba_baab = get_cc_j_doubles_doubles_abbabaab(**kwargs).reshape(dims['abba'], dims['baab'])
    abba_baba = get_cc_j_doubles_doubles_abbababa(**kwargs).reshape(dims['abba'], dims['baba'])
    abba_bbbb = get_cc_j_doubles_doubles_abbabbbb(**kwargs).reshape(dims['abba'], dims['bbbb'])
    baab_aaaa = get_cc_j_doubles_doubles_baabaaaa(**kwargs).reshape(dims['baab'], dims['aaaa'])
    baab_abab = get_cc_j_doubles_doubles_baababab(**kwargs).reshape(dims['baab'], dims['abab'])
    baab_abba = get_cc_j_doubles_doubles_baababba(**kwargs).reshape(dims['baab'], dims['abba'])
    baab_baab = get_cc_j_doubles_doubles_baabbaab(**kwargs).reshape(dims['baab'], dims['baab'])
    baab_baba = get_cc_j_doubles_doubles_baabbaba(**kwargs).reshape(dims['baab'], dims['baba'])
    baab_bbbb = get_cc_j_doubles_doubles_baabbbbb(**kwargs).reshape(dims['baab'], dims['bbbb'])
    baba_aaaa = get_cc_j_doubles_doubles_babaaaaa(**kwargs).reshape(dims['baba'], dims['aaaa'])
    baba_abab = get_cc_j_doubles_doubles_babaabab(**kwargs).reshape(dims['baba'], dims['abab'])
    baba_abba = get_cc_j_doubles_doubles_babaabba(**kwargs).reshape(dims['baba'], dims['abba'])
    baba_baab = get_cc_j_doubles_doubles_bababaab(**kwargs).reshape(dims['baba'], dims['baab'])
    baba_baba = get_cc_j_doubles_doubles_babababa(**kwargs).reshape(dims['baba'], dims['baba'])
    baba_bbbb = get_cc_j_doubles_doubles_bababbbb(**kwargs).reshape(dims['baba'], dims['bbbb'])
    bbbb_aaaa = np.zeros(shape=(dims['bbbb'], dims['aaaa']))
    bbbb_abab = get_cc_j_doubles_doubles_bbbbabab(**kwargs).reshape(dims['bbbb'], dims['abab'])
    bbbb_abba = get_cc_j_doubles_doubles_bbbbabba(**kwargs).reshape(dims['bbbb'], dims['abba'])
    bbbb_baab = get_cc_j_doubles_doubles_bbbbbaab(**kwargs).reshape(dims['bbbb'], dims['baab'])
    bbbb_baba = get_cc_j_doubles_doubles_bbbbbaba(**kwargs).reshape(dims['bbbb'], dims['baba'])
    bbbb_bbbb = get_cc_j_doubles_doubles_bbbbbbbb(**kwargs).reshape(dims['bbbb'], dims['bbbb'])

    doubles_doubles = np.block([
        [aaaa_aaaa, aaaa_abab, aaaa_abba, aaaa_baab, aaaa_baba, aaaa_bbbb,],
        [abab_aaaa, abab_abab, abab_abba, abab_baab, abab_baba, abab_bbbb,],
        [abba_aaaa, abba_abab, abba_abba, abba_baab, abba_baba, abba_bbbb,],
        [baab_aaaa, baab_abab, baab_abba, baab_baab, baab_baba, baab_bbbb,],
        [baba_aaaa, baba_abab, baba_abba, baba_baab, baba_baba, baba_bbbb,],
        [bbbb_aaaa, bbbb_abab, bbbb_abba, bbbb_baab, bbbb_baba, bbbb_bbbb,],
    ])

    return doubles_doubles
