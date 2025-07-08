from dataclasses import dataclass, field

from chem.ccsd.containers import GHF_CCSD_Data
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.hf.ghf_data import GHF_Data, ghf_data_to_GHF_ov_data
from chem.meta.coordinates import Descartes
from chem.meta.polarizability import Polarizability
from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE
import numpy as np
from numpy.typing import NDArray
from rspn.ghf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.ghf_ccsd._jacobian import build_cc_jacobian
from rspn.ghf_ccsd._lheecc import build_pol_xA_F_xB
import rspn.ghf_ccsd.equations.eta.singles as eta_singles
import rspn.ghf_ccsd.equations.eta.doubles as eta_doubles
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass
class GHF_CCSD_LR_config:
    """ 
    store_jacobian: One implementation builds the whole CC Jacobian matrix and
    uses it to solve the equation 
    Jacobian @ response_vector = "external_field_operator"

    The other implementation does not store the whole matrix but only
    implements the operator that takes an input_vector and returns the vector
    Jacobian @ input_vector. The second approach saves a ton of memory.

    Set to False to save memory.
    """
    gmres_threshold: float = 1e-5
    store_jacobian: bool = False
    store_lHeecc: bool = True
    verbose: int = 1

    def __str__(self) -> str:
        msg = "GHF-CCSD-LR config\n"
        msg += f"  Response threshold: {self.gmres_threshold}\n"
        msg += f"  Store CC Jacobian: {self.store_jacobian}\n"
        msg += f"  Store <ᴧ|[[H,τ_μ],τ_ν]|CC>: {self.store_lHeecc}\n"
        msg += f"  Verbose: {self.verbose}\n"
        return msg


@dataclass
class GHF_CCSD_LR:
    ghf_data: GHF_Data
    ghf_ccsd_data: GHF_CCSD_Data
    CONFIG: GHF_CCSD_LR_config = field(default_factory=GHF_CCSD_LR_config)

    def find_polarizabilities(self) -> Polarizability:
        if self.CONFIG.verbose > 0:
            print("Finding UHF-CCSD-LR polarizabilities.")
            print("Configuration:")
            print(self.CONFIG)

        builders_input = GHF_Generators_Input(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.ghf_ccsd_data,
        )

        cc_electric_dipole = build_nu_bar_V_cc(input=builders_input)

        if self.CONFIG.store_jacobian:
            cc_jacobian = build_cc_jacobian(
                kwargs=builders_input,
            )
            t_response = self.find_t_response(
                minus_cc_jacobian=-cc_jacobian,
                cc_mu=cc_electric_dipole,
            )
        else:
            raise NotImplementedError("GHF-CCSD Jacobian action.")
        eta_mu = self._find_eta_mu()
        # TODO: generalize to operators other than electric dipole
        pol_etaA_xB = self._build_pol_eta_X(eta_mu, t_response)
        # when there is only one operator this term is the same as the first
        # one pol_etaB_xA = self._build_pol_eta_X(eta_mu, t_response)
        pol_etaB_xA = pol_etaA_xB

        if self.CONFIG.store_lHeecc is True:
            pol_xA_F_xB = build_pol_xA_F_xB(
                builders_input, t_res_A=t_response, t_res_B=t_response,
            )
        else:
            raise NotImplementedError("On-the-fly contraction F response.")

        return pol_etaA_xB + pol_xA_F_xB + pol_etaB_xA

    def find_t_response(
        self,
        minus_cc_jacobian: NDArray | LinearOperator,
        cc_mu: dict[Descartes, dict[str, NDArray]],
    ) -> dict[Descartes, dict[str, NDArray]]:
        t_response_mu = {}
        for coord in Descartes:
            mu = cc_mu[coord]
            rhs_mbe = GHF_CCSD_MBE(
                singles=mu['singles'],
                doubles=mu['doubles'],
            )
            rhs = rhs_mbe.flatten()
            gmres_output = gmres(
                minus_cc_jacobian,
                rhs,
                atol=self.CONFIG.gmres_threshold,
            )
            exit_code: int = gmres_output[1]
            if exit_code != 0:
                msg = f'gmres didn\'t find the response vector for mu {coord}.'
                raise RuntimeError(msg)

            response: NDArray = gmres_output[0]
            ghf_ov_data = ghf_data_to_GHF_ov_data(self.ghf_data)
            response_mbe = GHF_CCSD_MBE.from_NDArray(response, ghf_ov_data)
            t_response_mu[coord] = {
                'singles': response_mbe.singles,
                'doubles': response_mbe.doubles,
            }
        return t_response_mu

    def _find_eta_mu(self) -> dict[Descartes, dict[str, NDArray]]:
        """ mu stands for the electric dipole moment. It is a special case of a
        general perturbation operator. """
        operators = {
            direction: self.ghf_data.mu[direction]
            for direction in Descartes
        }

        etas = {}
        for direction in Descartes:
            etas[direction] = dict()
            etas[direction]['singles'] = eta_singles.get_eta(
                self.ghf_data,
                self.ghf_ccsd_data,
                operators[direction],
            )
            etas[direction]['doubles'] = eta_doubles.get_eta(
                self.ghf_data,
                self.ghf_ccsd_data,
                operators[direction],
            )
        return etas

    def _build_pol_eta_X(
        self,
        eta: dict[Descartes, dict[str, NDArray]],
        t_response: dict[Descartes, dict[str, NDArray]],
    ) -> Polarizability:
        r"""
        Calculates
        sum _mu \eta _\mu X _\mu
        """
        pol = Polarizability.from_builder(
            builder=lambda first, second: float(
                np.einsum(
                    'ai,ai->',
                    eta[first]['singles'],
                    t_response[second]['singles'],
                )
                +
                np.einsum(
                    'abij,abij->',
                    eta[first]['doubles'],
                    t_response[second]['doubles'],
                )
            ),
        )
        return pol
