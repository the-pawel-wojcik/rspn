from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import Descartes
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from rspn.uhf_ccsd.equations.lHeecc.e1e1 import (
    get_lhe1e1cc_aaaa,
    get_lhe1e1cc_aabb,
    get_lhe1e1cc_bbaa,
    get_lhe1e1cc_bbbb,
)
from rspn.uhf_ccsd.equations.lHeecc.e1e2 import (
    get_lhe1e2cc_aaaaaa,
    get_lhe1e2cc_aaabab,
    get_lhe1e2cc_aaabba,
    get_lhe1e2cc_aabaab,
    get_lhe1e2cc_aababa,
    get_lhe1e2cc_aabbbb,
    get_lhe1e2cc_bbaaaa,
    get_lhe1e2cc_bbabab,
    get_lhe1e2cc_bbabba,
    get_lhe1e2cc_bbbaab,
    get_lhe1e2cc_bbbaba,
    get_lhe1e2cc_bbbbbb,
)
from rspn.uhf_ccsd.equations.lHeecc.e2e1 import (
    get_lhe2e1cc_aaaaaa,
    get_lhe2e1cc_aaaabb,
    get_lhe2e1cc_ababaa,
    get_lhe2e1cc_ababbb,
    get_lhe2e1cc_abbaaa,
    get_lhe2e1cc_abbabb,
    get_lhe2e1cc_baabaa,
    get_lhe2e1cc_baabbb,
    get_lhe2e1cc_babaaa,
    get_lhe2e1cc_bababb,
    get_lhe2e1cc_bbbbaa,
    get_lhe2e1cc_bbbbbb,
)
from rspn.uhf_ccsd.equations.lHeecc.e2e2 import (
    get_lhe2e2cc_aaaaaaaa,
    get_lhe2e2cc_aaaaabab,
    get_lhe2e2cc_aaaaabba,
    get_lhe2e2cc_aaaabaab,
    get_lhe2e2cc_aaaababa,
    get_lhe2e2cc_aaaabbbb,
    get_lhe2e2cc_ababaaaa,
    get_lhe2e2cc_abababab,
    get_lhe2e2cc_abababba,
    get_lhe2e2cc_ababbaab,
    get_lhe2e2cc_ababbaba,
    get_lhe2e2cc_ababbbbb,
    get_lhe2e2cc_abbaaaaa,
    get_lhe2e2cc_abbaabab,
    get_lhe2e2cc_abbaabba,
    get_lhe2e2cc_abbabaab,
    get_lhe2e2cc_abbababa,
    get_lhe2e2cc_abbabbbb,
    get_lhe2e2cc_baabaaaa,
    get_lhe2e2cc_baababab,
    get_lhe2e2cc_baababba,
    get_lhe2e2cc_baabbaab,
    get_lhe2e2cc_baabbaba,
    get_lhe2e2cc_baabbbbb,
    get_lhe2e2cc_babaaaaa,
    get_lhe2e2cc_babaabab,
    get_lhe2e2cc_babaabba,
    get_lhe2e2cc_bababaab,
    get_lhe2e2cc_babababa,
    get_lhe2e2cc_bababbbb,
    get_lhe2e2cc_bbbbaaaa,
    get_lhe2e2cc_bbbbabab,
    get_lhe2e2cc_bbbbabba,
    get_lhe2e2cc_bbbbbaab,
    get_lhe2e2cc_bbbbbaba,
    get_lhe2e2cc_bbbbbbbb,
)


def build_pol_xA_F_xB(
    kwargs: GeneratorsInput,
    t_res_B: dict[Descartes, dict[str, NDArray]],
    t_res_A: dict[Descartes, dict[str, NDArray]],
) -> Polarizability:
    f_aa_aa = get_lhe1e1cc_aaaa(**kwargs)
    f_aa_bb = get_lhe1e1cc_aabb(**kwargs)
    f_bb_aa = get_lhe1e1cc_bbaa(**kwargs)
    f_bb_bb = get_lhe1e1cc_bbbb(**kwargs)

    f_aa_aaaa = get_lhe1e2cc_aaaaaa(**kwargs)
    f_aa_abab = get_lhe1e2cc_aaabab(**kwargs)
    f_aa_abba = get_lhe1e2cc_aaabba(**kwargs)
    f_aa_baab = get_lhe1e2cc_aabaab(**kwargs)
    f_aa_baba = get_lhe1e2cc_aababa(**kwargs)
    f_aa_bbbb = get_lhe1e2cc_aabbbb(**kwargs)
    f_bb_aaaa = get_lhe1e2cc_bbaaaa(**kwargs)
    f_bb_abab = get_lhe1e2cc_bbabab(**kwargs)
    f_bb_abba = get_lhe1e2cc_bbabba(**kwargs)
    f_bb_baab = get_lhe1e2cc_bbbaab(**kwargs)
    f_bb_baba = get_lhe1e2cc_bbbaba(**kwargs)
    f_bb_bbbb = get_lhe1e2cc_bbbbbb(**kwargs)

    f_aaaa_aa = get_lhe2e1cc_aaaaaa(**kwargs)
    f_aaaa_bb = get_lhe2e1cc_aaaabb(**kwargs)
    f_abab_aa = get_lhe2e1cc_ababaa(**kwargs)
    f_abab_bb = get_lhe2e1cc_ababbb(**kwargs)
    f_abba_aa = get_lhe2e1cc_abbaaa(**kwargs)
    f_abba_bb = get_lhe2e1cc_abbabb(**kwargs)
    f_baab_aa = get_lhe2e1cc_baabaa(**kwargs)
    f_baab_bb = get_lhe2e1cc_baabbb(**kwargs)
    f_baba_aa = get_lhe2e1cc_babaaa(**kwargs)
    f_baba_bb = get_lhe2e1cc_bababb(**kwargs)
    f_bbbb_aa = get_lhe2e1cc_bbbbaa(**kwargs)
    f_bbbb_bb = get_lhe2e1cc_bbbbbb(**kwargs)

    f_aaaa_aaaa = get_lhe2e2cc_aaaaaaaa(**kwargs)
    f_aaaa_abab = get_lhe2e2cc_aaaaabab(**kwargs)
    f_aaaa_abba = get_lhe2e2cc_aaaaabba(**kwargs)
    f_aaaa_baab = get_lhe2e2cc_aaaabaab(**kwargs)
    f_aaaa_baba = get_lhe2e2cc_aaaababa(**kwargs)
    f_aaaa_bbbb = get_lhe2e2cc_aaaabbbb(**kwargs)
    f_abab_aaaa = get_lhe2e2cc_ababaaaa(**kwargs)
    f_abab_abab = get_lhe2e2cc_abababab(**kwargs)
    f_abab_abba = get_lhe2e2cc_abababba(**kwargs)
    f_abab_baab = get_lhe2e2cc_ababbaab(**kwargs)
    f_abab_baba = get_lhe2e2cc_ababbaba(**kwargs)
    f_abab_bbbb = get_lhe2e2cc_ababbbbb(**kwargs)
    f_abba_aaaa = get_lhe2e2cc_abbaaaaa(**kwargs)
    f_abba_abab = get_lhe2e2cc_abbaabab(**kwargs)
    f_abba_abba = get_lhe2e2cc_abbaabba(**kwargs)
    f_abba_baab = get_lhe2e2cc_abbabaab(**kwargs)
    f_abba_baba = get_lhe2e2cc_abbababa(**kwargs)
    f_abba_bbbb = get_lhe2e2cc_abbabbbb(**kwargs)
    f_baab_aaaa = get_lhe2e2cc_baabaaaa(**kwargs)
    f_baab_abab = get_lhe2e2cc_baababab(**kwargs)
    f_baab_abba = get_lhe2e2cc_baababba(**kwargs)
    f_baab_baab = get_lhe2e2cc_baabbaab(**kwargs)
    f_baab_baba = get_lhe2e2cc_baabbaba(**kwargs)
    f_baab_bbbb = get_lhe2e2cc_baabbbbb(**kwargs)
    f_baba_aaaa = get_lhe2e2cc_babaaaaa(**kwargs)
    f_baba_abab = get_lhe2e2cc_babaabab(**kwargs)
    f_baba_abba = get_lhe2e2cc_babaabba(**kwargs)
    f_baba_baab = get_lhe2e2cc_bababaab(**kwargs)
    f_baba_baba = get_lhe2e2cc_babababa(**kwargs)
    f_baba_bbbb = get_lhe2e2cc_bababbbb(**kwargs)
    f_bbbb_aaaa = get_lhe2e2cc_bbbbaaaa(**kwargs)
    f_bbbb_abab = get_lhe2e2cc_bbbbabab(**kwargs)
    f_bbbb_abba = get_lhe2e2cc_bbbbabba(**kwargs)
    f_bbbb_baab = get_lhe2e2cc_bbbbbaab(**kwargs)
    f_bbbb_baba = get_lhe2e2cc_bbbbbaba(**kwargs)
    f_bbbb_bbbb = get_lhe2e2cc_bbbbbbbb(**kwargs)

    return Polarizability.from_builder(
        builder=lambda first, second: (
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['aa'],
                f_aa_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['aa'],
                f_aa_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['bb'],
                f_bb_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'ai,aibj,bj->',
                t_res_A[first]['bb'],
                f_bb_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['aa'],
                f_aa_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'ai,aibckj,bckj->',
                t_res_A[first]['bb'],
                f_bb_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['aaaa'],
                f_aaaa_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['aaaa'],
                f_aaaa_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['abab'],
                f_abab_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['abab'],
                f_abab_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['abba'],
                f_abba_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['abba'],
                f_abba_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['baab'],
                f_baab_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['baab'],
                f_baab_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['baba'],
                f_baba_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['baba'],
                f_baba_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['bbbb'],
                f_bbbb_aa,
                t_res_B[second]['aa'],
            )
            +
            np.einsum(
                'abji,abjick,ck->',
                t_res_A[first]['bbbb'],
                f_bbbb_bb,
                t_res_B[second]['bb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['aaaa'],
                f_aaaa_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abab'],
                f_abab_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['abba'],
                f_abba_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baab'],
                f_baab_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['baba'],
                f_baba_bbbb,
                t_res_B[second]['bbbb'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_aaaa,
                t_res_B[second]['aaaa'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_abab,
                t_res_B[second]['abab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_abba,
                t_res_B[second]['abba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_baab,
                t_res_B[second]['baab'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_baba,
                t_res_B[second]['baba'],
            )
            +
            np.einsum(
                'abji,abjicdlk,cdlk->',
                t_res_A[first]['bbbb'],
                f_bbbb_bbbb,
                t_res_B[second]['bbbb'],
            )
        )
    )
