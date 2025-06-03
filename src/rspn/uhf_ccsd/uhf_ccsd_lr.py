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
from rspn.uhf_ccsd.equations.cc_jacobian.singles_singles import (
    get_singles_singles_aaaa,
    get_singles_singles_aabb,
    get_singles_singles_bbaa,
    get_singles_singles_bbbb,
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
            response: NDArray = gmres_output[0]
            scf = self.uhf_scf_data
            nmo = scf.nmo
            noa = scf.noa
            nva = nmo - noa
            nob = scf.nob
            nvb = nmo - nob
            t_response_mu[coord] = {
                'aa': response[:dims['aa']].reshape((nva, noa)),
                'bb': response[dims['aa']:].reshape((nvb, nob)),
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
        return self.dims

    def build_the_cc_jacobian(self):
        self.assign_dims()
        dim_aa = self.dims['aa']
        dim_bb = self.dims['bb']
        aaaa = get_singles_singles_aaaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_aa, dim_aa)
        aabb = get_singles_singles_aabb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_aa, dim_bb)
        bbaa = get_singles_singles_bbaa(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_bb, dim_aa)
        bbbb = get_singles_singles_bbbb(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dim_bb, dim_bb)
        jacobian = np.block([[aaaa, aabb], [bbaa, bbbb]])
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
            }

        elif coord == Descartes.y:
            return {
                'aa': get_muy_aa(scf, ccsd),
                'bb': get_muy_bb(scf, ccsd),
            }

        elif coord == Descartes.z:
            return {
                'aa': get_muz_aa(scf, ccsd),
                'bb': get_muz_bb(scf, ccsd),
            }

        else:
            raise ValueError(f"Unknown cartesian coordinate: {coord}.")
