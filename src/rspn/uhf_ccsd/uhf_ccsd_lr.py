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
import rspn.uhf_ccsd.equations.eta.singles as eta_singles
import rspn.uhf_ccsd.equations.eta.doubles as eta_doubles
from rspn.uhf_ccsd._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccsd._jacobian import build_cc_jacobian

@dataclass
class UHF_CCSD_LR:
    uhf_ccsd_data: UHF_CCSD_Data
    uhf_scf_data: Intermediates
    SPIN_BLOCKS: list[str]  = [
        'aa', 'bb',
        'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
    ]

    def find_polarizabilities(self) -> Polarizability:
        builders_input = GeneratorsInput(
            uhf_scf_data=self.uhf_scf_data,
            uhf_ccsd_data=self.uhf_ccsd_data,
        )
        cc_jacobian = build_cc_jacobian(
            kwargs=builders_input,
            dims=self.assign_dims(),
        )
        cc_electric_dipole = self.build_cc_electric_diple()
        t_response = self.find_t_response(cc_jacobian, cc_electric_dipole)
        eta_mu = self._find_eta_mu()

        # TODO: all operators work only for the electric dipole operator
        pol_etaA_xB = self._build_pol_eta_X(eta_mu, t_response)
        pol_xA_F_xB = build_pol_xA_F_xB(
            builders_input, t_res_A=t_response, t_res_B=t_response,
        )
        # when there is only one operator this term is the same as the first one
        # pol_etaB_xA = self._build_pol_eta_X(eta_mu, t_response)
        pol_etaB_xA = pol_etaA_xB

        return pol_etaA_xB + pol_xA_F_xB + pol_etaB_xA

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
                for spin in self.SPIN_BLOCKS
            ),
        )
        return pol
    
    def _find_eta_mu(self) -> dict[Descartes, dict[str, NDArray]]:
        """ mu stands for the electric dipole moment. """
        operators = {
            Descartes.x: dict(
                operator_aa=self.uhf_scf_data.mua_x,
                operator_bb=self.uhf_scf_data.mub_x,
            ),
            Descartes.y: dict(
                operator_aa=self.uhf_scf_data.mua_y,
                operator_bb=self.uhf_scf_data.mub_y,
            ),
            Descartes.z: dict(
                operator_aa=self.uhf_scf_data.mua_z,
                operator_bb=self.uhf_scf_data.mub_z,
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
            etas[coord]['aaaa'] = eta_doubles.get_eta_aaaa(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['abab'] = eta_doubles.get_eta_abab(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['abba'] = eta_doubles.get_eta_abba(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['baab'] = eta_doubles.get_eta_baab(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['baba'] = eta_doubles.get_eta_baba(
                self.uhf_scf_data,
                self.uhf_ccsd_data,
                **operators[coord],
            )
            etas[coord]['bbbb'] = eta_doubles.get_eta_bbbb(
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
                tuple(mu[block].reshape(-1, 1) for block in self.SPIN_BLOCKS)
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

            slices = dict()
            current_size = 0
            for block in self.SPIN_BLOCKS:
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
                for block in self.SPIN_BLOCKS
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

    def build_cc_electric_diple(
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
