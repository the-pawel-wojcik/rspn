from dataclasses import dataclass
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import Descartes, CARTESIAN
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres
from rspn.uhf_ccsd.equations.dipole.singles import (
    get_mux_aa,
    get_mux_bb,
    get_muy_aa,
    get_muy_bb,
    get_muz_aa,
    get_muz_bb,
)
from rspn.uhf_ccsd.equations.dipole.doubles import (
    get_mux_aaaa,
    get_mux_abab,
    get_mux_abba,
    get_mux_baab,
    get_mux_baba,
    get_mux_bbbb,
    get_muy_aaaa,
    get_muy_abab,
    get_muy_abba,
    get_muy_baab,
    get_muy_baba,
    get_muy_bbbb,
    get_muz_aaaa,
    get_muz_abab,
    get_muz_abba,
    get_muz_baab,
    get_muz_baba,
    get_muz_bbbb,
)
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
import rspn.uhf_ccsd.equations.eta.singles as eta_singles
import rspn.uhf_ccsd.equations.eta.doubles as eta_doubles
from rspn.uhf_ccsd.equations.lHeecc.e1e1 import (
    get_lhe1e1cc_aaaa,
    get_lhe1e1cc_aabb,
    get_lhe1e1cc_bbaa,
    get_lhe1e1cc_bbbb,
)

@dataclass
class UHF_CCSD_LR:
    uhf_ccsd_data: UHF_CCSD_Data
    uhf_scf_data: Intermediates

    def find_polarizabilities(self) -> Polarizability:
        cc_jacobian = self.build_the_cc_jacobian()
        cc_electric_dipole = self.build_cc_electric_dipole_singles()
        t_response = self.find_t_response(cc_jacobian, cc_electric_dipole)
        eta_mu = self._find_eta_mu()

        # TODO: all operators work only for the electric dipole operator
        pol_etaA_xB = self._build_pol_eta_X(eta_mu, t_response)
        pol_xA_F_xB = self._build_pol_xA_F_xB(
            t_res_A=t_response, t_res_B=t_response,
        )
        # when there is only one operator this term is the same as the first one
        # pol_etaB_xA = self._build_pol_eta_X(eta_mu, t_response)
        pol_etaB_xA = pol_etaA_xB

        return pol_etaA_xB + pol_xA_F_xB + pol_etaB_xA

    def _build_pol_xA_F_xB(self, t_res_B, t_res_A) -> Polarizability:
        kwargs = GeneratorsInput(
            uhf_scf_data=self.uhf_scf_data,
            uhf_ccsd_data=self.uhf_ccsd_data,
        )
        f_aaaa = get_lhe1e1cc_aaaa(**kwargs)
        f_aabb = get_lhe1e1cc_aabb(**kwargs)
        f_bbaa = get_lhe1e1cc_bbaa(**kwargs)
        f_bbbb = get_lhe1e1cc_bbbb(**kwargs)

        return Polarizability.from_builder(
            builder=lambda first, second: (
                np.einsum(
                    'ai,aibj,bj->',
                    t_res_A[first]['aa'],
                    f_aaaa,
                    t_res_B[second]['aa'],
                )
                +
                np.einsum(
                    'ai,aibj,bj->',
                    t_res_A[first]['aa'],
                    f_aabb,
                    t_res_B[second]['bb'],
                )
                +
                np.einsum(
                    'ai,aibj,bj->',
                    t_res_A[first]['bb'],
                    f_bbaa,
                    t_res_B[second]['aa'],
                )
                +
                np.einsum(
                    'ai,aibj,bj->',
                    t_res_A[first]['bb'],
                    f_bbbb,
                    t_res_B[second]['bb'],
                )
            )
        )

    def _build_pol_eta_X(self, eta, t_response) -> Polarizability:
        r"""
        Calculates 
        sum _mu \eta _\mu X _\mu
        """
        pol = Polarizability.from_builder(
            builder=lambda first, second: sum(
                float(
                    np.sum(eta[first][spin] * t_response[second][spin])
                )
                for spin in ['aa', 'bb']
            ),
        )
        return pol
    
    def _find_eta_mu(self) -> dict[Descartes, dict[str, NDArray]]:
        """ mu stands for the electric dipole moment """
        operators = {
            Descartes.x: dict(
                h_aa=self.uhf_scf_data.mua_x,
                h_bb=self.uhf_scf_data.mub_x,
            ),
            Descartes.y: dict(
                h_aa=self.uhf_scf_data.mua_y,
                h_bb=self.uhf_scf_data.mub_y,
            ),
            Descartes.z: dict(
                h_aa=self.uhf_scf_data.mua_z,
                h_bb=self.uhf_scf_data.mub_z,
            ),
        }
        etas = {}
        for coord in CARTESIAN:
            etas[coord] = dict()
            etas[coord]['aa'] = eta_singles.get_eta_aa(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['bb'] = eta_singles.get_eta_bb(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
        return etas

    def find_t_response(
        self,
        cc_jacobian: NDArray,
        cc_mu: dict[Descartes, dict[str, NDArray]],
    ) -> dict[Descartes, dict[str, NDArray]]:
        dims = self.assign_dims()
        t_response_mu = {}
        for coord in CARTESIAN:
            mu = cc_mu[coord]
            rhs = np.vstack(
                (
                    mu['aa'].reshape(-1, 1),
                    mu['bb'].reshape(-1, 1),
                    mu['aaaa'].reshape(-1, 1),
                    mu['abab'].reshape(-1, 1),
                    mu['abba'].reshape(-1, 1),
                    mu['baab'].reshape(-1, 1),
                    mu['baba'].reshape(-1, 1),
                    mu['bbbb'].reshape(-1, 1),
                )
            )
            gmres_output = gmres(
                cc_jacobian,
                rhs,
                rtol=1e-7,
                atol=1e-7,
            )
            exit_code: int = gmres_output[1]
            if exit_code != 0:
                msg = f'gmres didn\'t find the response vector for mu {coord}.'
                raise RuntimeError(msg)

            scf = self.uhf_scf_data
            nmo = scf.nmo
            noa = scf.noa
            nob = scf.nob
            nva = nmo - noa
            nvb = nmo - nob

            blocks = [
                'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
            ]

            slices = dict()
            current_size = 0
            for block in blocks:
                block_dim = dims[block]
                slices[block] = slice(current_size, current_size + block_dim)
                current_size += block_dim

            shapes = {
                'aa': (nva, noa),
                'bb': (nvb, nob),
                'aaaa': (nva, nva, noa, noa),
                'abab': (nva, nvb, noa, nob),
                'abba': (nva, nvb, nob, noa),
                'baab': (nvb, nva, noa, nob),
                'baba': (nvb, nva, nob, noa),
                'bbbb': (nvb, nvb, nob, nob),
            }

            response: NDArray = gmres_output[0]
            t_response_mu[coord] = {
                block: response[slices[block]].reshape(shapes[block])
                for block in blocks
            }
        return t_response_mu

    def assign_dims(self) -> dict[str, int]:
        """ TODO: make it work alright """
        scf = self.uhf_scf_data
        nmo = scf.nmo
        noa = scf.noa
        nva = nmo - noa
        nob = scf.nob
        nvb = nmo - nob
        self.dims = {}
        self.dims['aa'] = nva * noa
        self.dims['ab'] = nva * nob
        self.dims['ba'] = nvb * noa
        self.dims['bb'] = nvb * nob
        self.dims['aaaa'] = nva * nva * noa * noa
        self.dims['abab'] = nva * nvb * noa * nob
        self.dims['abba'] = nva * nvb * nob * noa
        self.dims['baab'] = nvb * nva * noa * nob
        self.dims['baba'] = nvb * nva * nob * noa
        self.dims['bbbb'] = nvb * nvb * nob * nob
        return self.dims

    def build_the_cc_jacobian(self):
        dims = self.assign_dims()
        dim_aa = self.dims['aa']
        dim_bb = self.dims['bb']
        aa_aa = get_cc_j_singles_singles_aaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_aa, dim_aa)
        aa_bb = get_cc_j_singles_singles_aabb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_aa, dim_bb)
        bb_aa = get_cc_j_singles_singles_bbaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_bb, dim_aa)
        bb_bb = get_cc_j_singles_singles_bbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_bb, dim_bb)

        aa_aaaa = get_cc_j_singles_doubles_aaaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aa'], dims['aaaa'])
        aa_abab = get_cc_j_singles_doubles_aaabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aa'], dims['abab'])
        aa_abba = get_cc_j_singles_doubles_aaabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aa'], dims['abba'])
        aa_baab = get_cc_j_singles_doubles_aabaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aa'], dims['baab'])
        aa_baba = get_cc_j_singles_doubles_aababa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aa'], dims['baba'])
        aa_bbbb = np.zeros(shape=(dims['aa'], dims['bbbb']))
        bb_aaaa = np.zeros(shape=(dims['bb'], dims['aaaa']))
        bb_abab = get_cc_j_singles_doubles_bbabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bb'], dims['abab'])
        bb_abba = get_cc_j_singles_doubles_bbabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bb'], dims['abba'])
        bb_baab = get_cc_j_singles_doubles_bbbaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bb'], dims['baab'])
        bb_baba = get_cc_j_singles_doubles_bbbaba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bb'], dims['baba'])
        bb_bbbb = get_cc_j_singles_doubles_bbbbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bb'], dims['bbbb'])

        aaaa_aa = get_cc_j_doubles_singles_aaaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['aa'])
        abab_aa = get_cc_j_doubles_singles_ababaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['aa'])
        abba_aa = get_cc_j_doubles_singles_abbaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['aa'])
        baab_aa = get_cc_j_doubles_singles_baabaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['aa'])
        baba_aa = get_cc_j_doubles_singles_babaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['aa'])
        bbbb_aa = get_cc_j_doubles_singles_bbbbaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['aa'])

        aaaa_bb = get_cc_j_doubles_singles_aaaabb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['bb'])
        abab_bb = get_cc_j_doubles_singles_ababbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['bb'])
        abba_bb = get_cc_j_doubles_singles_abbabb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['bb'])
        baab_bb = get_cc_j_doubles_singles_baabbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['bb'])
        baba_bb = get_cc_j_doubles_singles_bababb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['bb'])
        bbbb_bb = get_cc_j_doubles_singles_bbbbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['bb'])

        aaaa_aaaa = get_cc_j_doubles_doubles_aaaaaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['aaaa'])

        aaaa_abab = get_cc_j_doubles_doubles_aaaaabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['abab'])

        aaaa_abba = get_cc_j_doubles_doubles_aaaaabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['abba'])

        aaaa_baab = get_cc_j_doubles_doubles_aaaabaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['baab'])

        aaaa_baba = get_cc_j_doubles_doubles_aaaababa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['aaaa'], dims['baba'])

        aaaa_bbbb = np.zeros(shape=(dims['aaaa'], dims['bbbb']))

        abab_aaaa = get_cc_j_doubles_doubles_ababaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['aaaa'])

        abab_abab = get_cc_j_doubles_doubles_abababab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['abab'])

        abab_abba = get_cc_j_doubles_doubles_abababba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['abba'])

        abab_baab = get_cc_j_doubles_doubles_ababbaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['baab'])

        abab_baba = get_cc_j_doubles_doubles_ababbaba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['baba'])

        abab_bbbb = get_cc_j_doubles_doubles_ababbbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abab'], dims['bbbb'])

        abba_aaaa = get_cc_j_doubles_doubles_abbaaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['aaaa'])

        abba_abab = get_cc_j_doubles_doubles_abbaabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['abab'])

        abba_abba = get_cc_j_doubles_doubles_abbaabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['abba'])

        abba_baab = get_cc_j_doubles_doubles_abbabaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['baab'])

        abba_baba = get_cc_j_doubles_doubles_abbababa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['baba'])

        abba_bbbb = get_cc_j_doubles_doubles_abbabbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['abba'], dims['bbbb'])

        baab_aaaa = get_cc_j_doubles_doubles_baabaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['aaaa'])

        baab_abab = get_cc_j_doubles_doubles_baababab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['abab'])

        baab_abba = get_cc_j_doubles_doubles_baababba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['abba'])

        baab_baab = get_cc_j_doubles_doubles_baabbaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['baab'])

        baab_baba = get_cc_j_doubles_doubles_baabbaba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['baba'])

        baab_bbbb = get_cc_j_doubles_doubles_baabbbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baab'], dims['bbbb'])

        baba_aaaa = get_cc_j_doubles_doubles_babaaaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['aaaa'])

        baba_abab = get_cc_j_doubles_doubles_babaabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['abab'])

        baba_abba = get_cc_j_doubles_doubles_babaabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['abba'])

        baba_baab = get_cc_j_doubles_doubles_bababaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['baab'])

        baba_baba = get_cc_j_doubles_doubles_babababa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['baba'])

        baba_bbbb = get_cc_j_doubles_doubles_bababbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['baba'], dims['bbbb'])

        bbbb_aaaa = np.zeros(shape=(dims['bbbb'], dims['aaaa']))

        bbbb_abab = get_cc_j_doubles_doubles_bbbbabab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['abab'])

        bbbb_abba = get_cc_j_doubles_doubles_bbbbabba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['abba'])

        bbbb_baab = get_cc_j_doubles_doubles_bbbbbaab(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['baab'])

        bbbb_baba = get_cc_j_doubles_doubles_bbbbbaba(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['baba'])

        bbbb_bbbb = get_cc_j_doubles_doubles_bbbbbbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['bbbb'], dims['bbbb'])

        jacobian = np.block([
            [aa_aa, aa_bb, aa_aaaa, aa_abab, aa_abba, aa_baab, aa_baba, aa_bbbb,],
            [bb_aa, bb_bb, bb_aaaa, bb_abab, bb_abba, bb_baab, bb_baba, bb_bbbb,],
            [aaaa_aa, aaaa_bb, aaaa_aaaa, aaaa_abab, aaaa_abba, aaaa_baab, aaaa_baba, aaaa_bbbb,],
            [abab_aa, abab_bb, abab_aaaa, abab_abab, abab_abba, abab_baab, abab_baba, abab_bbbb,],
            [abba_aa, abba_bb, abba_aaaa, abba_abab, abba_abba, abba_baab, abba_baba, abba_bbbb,],
            [baab_aa, baab_bb, baab_aaaa, baab_abab, baab_abba, baab_baab, baab_baba, baab_bbbb,],
            [baba_aa, baba_bb, baba_aaaa, baba_abab, baba_abba, baba_baab, baba_baba, baba_bbbb,],
            [bbbb_aa, bbbb_bb, bbbb_aaaa, bbbb_abab, bbbb_abba, bbbb_baab, bbbb_baba, bbbb_bbbb,],
        ])
        return jacobian

    def build_cc_electric_dipole_singles(
        self
    ) -> dict[Descartes, dict[str, NDArray]]:
        r"""
        Builds the matrices:

        matrix[x][a, i] = <HF| e^{-T} a*(i) a(a) \hat{mu} _x E^{T} |HF>

        the flipped order of `a` and `i` appears because the pair (a, i) is an
        index of a single substitution $\tau _mu ^\dagger$. The theory from
        Ref. [1] works on the substitution indices.

        [1] H. Koch and P. Jørgensen, Coupled cluster response functions, The
        Journal of Chemical Physics 93, 3333 (1990).
        """
        cc_mu = {}
        for coord in CARTESIAN:
            cc_mu[coord] = self._build_the_cc_dipole_helper(coord)
        return cc_mu

    def _build_the_cc_dipole_helper(
        self, coord: Descartes,
    ) -> dict[str, NDArray]:
        scf = self.uhf_scf_data
        ccsd = self.uhf_ccsd_data

        if coord == Descartes.x:
            return {
                'aa': get_mux_aa(scf, ccsd),
                'bb': get_mux_bb(scf, ccsd),
                'aaaa': get_mux_aaaa(scf, ccsd),
                'abab': get_mux_abab(scf, ccsd),
                'abba': get_mux_abba(scf, ccsd),
                'baab': get_mux_baab(scf, ccsd),
                'baba': get_mux_baba(scf, ccsd),
                'bbbb': get_mux_bbbb(scf, ccsd),
            }

        elif coord == Descartes.y:
            return {
                'aa': get_muy_aa(scf, ccsd),
                'bb': get_muy_bb(scf, ccsd),
                'aaaa': get_muy_aaaa(scf, ccsd),
                'abab': get_muy_abab(scf, ccsd),
                'abba': get_muy_abba(scf, ccsd),
                'baab': get_muy_baab(scf, ccsd),
                'baba': get_muy_baba(scf, ccsd),
                'bbbb': get_muy_bbbb(scf, ccsd),
            }

        elif coord == Descartes.z:
            return {
                'aa': get_muz_aa(scf, ccsd),
                'bb': get_muz_bb(scf, ccsd),
                'aaaa': get_muz_aaaa(scf, ccsd),
                'abab': get_muz_abab(scf, ccsd),
                'abba': get_muz_abba(scf, ccsd),
                'baab': get_muz_baab(scf, ccsd),
                'baba': get_muz_baba(scf, ccsd),
                'bbbb': get_muz_bbbb(scf, ccsd),
            }

        else:
            raise ValueError(f"Unknown cartesian coordinate: {coord}.")
