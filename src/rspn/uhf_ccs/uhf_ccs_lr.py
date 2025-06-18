from dataclasses import dataclass
from chem.ccs.containers import UHF_CCS_Data, UHF_CCS_Lambda_Data
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.equations.util import UHF_CCS_InputPair, UHF_CCS_InputTriple
from chem.meta.coordinates import Descartes, CARTESIAN
from chem.meta.polarizability import Polarizability
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres
import rspn.uhf_ccs.equations.eta.singles as eta_singles
from rspn.uhf_ccs._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccs._jacobian import build_cc_jacobian
from rspn.uhf_ccs._nuOpCC import build_nu_bar_V_cc


@dataclass
class UHF_CCS_LR:
    uhf_data: Intermediates
    uhf_ccs_data: UHF_CCS_Data
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data
    SPIN_BLOCKS: tuple[str, ...] = ('aa', 'bb',)

    def find_polarizabilities(self) -> Polarizability:
        input_pair = UHF_CCS_InputPair(
            uhf_data=self.uhf_data,
            uhf_ccs_data=self.uhf_ccs_data,
        )
        cc_jacobian = build_cc_jacobian(
            kwargs=input_pair,
            dims=self.assign_dims(),
        )
        cc_interaction_operator = build_nu_bar_V_cc(input=input_pair)
        t_response = self.find_t_response(cc_jacobian, cc_interaction_operator)

        eta_mu = self._find_eta_mu()
        # TODO: all operators work only for the electric dipole operator
        pol_etaA_xB = self._build_pol_eta_X(eta_mu, t_response)
        input_triple = UHF_CCS_InputTriple(
            uhf_data=self.uhf_data,
            uhf_ccs_data=self.uhf_ccs_data,
            uhf_ccs_lambda_data=self.uhf_ccs_lambda_data,
        )
        pol_xA_F_xB = build_pol_xA_F_xB(
            input_triple, t_res_A=t_response, t_res_B=t_response,
        )
        # when there is only one operator this term is the same as the first
        # one pol_etaB_xA = self._build_pol_eta_X(eta_mu, t_response)
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
        """mu stands for the electric dipole moment. It is a special case of a
        general perturbation operator. See the comment in
        rspn.uhf_ccs.equations.eta.generate.py for details. For CCS with T=0
        and L = 0, the eta reduces a lot."""
        operators = {
            Descartes.x: dict(
                operator_aa=self.uhf_data.mua_x,
                operator_bb=self.uhf_data.mub_x,
            ),
            Descartes.y: dict(
                operator_aa=self.uhf_data.mua_y,
                operator_bb=self.uhf_data.mub_y,
            ),
            Descartes.z: dict(
                operator_aa=self.uhf_data.mua_z,
                operator_bb=self.uhf_data.mub_z,
            ),
        }
        input_triple = UHF_CCS_InputTriple(
            uhf_data=self.uhf_data,
            uhf_ccs_data=self.uhf_ccs_data,
            uhf_ccs_lambda_data=self.uhf_ccs_lambda_data,
        )
        etas = {}
        for coord in CARTESIAN:
            etas[coord] = dict()
            etas[coord]['aa'] = eta_singles.get_eta_aa(
                **input_triple,
                **operators[coord],
            )
            etas[coord]['bb'] = eta_singles.get_eta_bb(
                **input_triple,
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
                -cc_jacobian,
                rhs,
                rtol=1e-10,
                atol=1e-10,
            )
            exit_code: int = gmres_output[1]
            if exit_code != 0:
                msg = f'gmres didn\'t find the response vector for mu {coord}.'
                raise RuntimeError(msg)

            scf = self.uhf_data
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
            }

            response: NDArray = gmres_output[0]
            t_response_mu[coord] = {
                block: response[slices[block]].reshape(shapes[block])
                for block in self.SPIN_BLOCKS
            }
        return t_response_mu

    def assign_dims(self) -> dict[str, int]:
        """ TODO: make it work alright """
        scf = self.uhf_data
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
