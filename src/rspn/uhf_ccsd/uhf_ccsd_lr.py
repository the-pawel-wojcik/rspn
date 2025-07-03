from dataclasses import dataclass, field
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.hf.intermediates_builders import Intermediates
from chem.hf.util import E1_spin, E2_spin
from chem.ccsd.equations.util import GeneratorsInput
from chem.meta.coordinates import Descartes, CARTESIAN
from chem.meta.polarizability import Polarizability
from chem.meta.spin_mbe import Spin_MBE
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres
import rspn.uhf_ccsd.equations.eta.singles as eta_singles
import rspn.uhf_ccsd.equations.eta.doubles as eta_doubles
from rspn.uhf_ccsd._lheecc import build_pol_xA_F_xB
from rspn.uhf_ccsd._lhtwcc import build_pol_xA_F_xB as build_pol_xA_F_xB_wo_store
from rspn.uhf_ccsd._jacobian import build_cc_jacobian
from rspn.uhf_ccsd._nuOpCC import build_nu_bar_V_cc
from rspn.uhf_ccsd._jacobian_action import Minus_UHF_CCSD_Jacobian_action
# from rspn.uhf_ccsd._jacobian_action_Stephen import (
#     Minus_UHF_CCSD_Jacobian_action
# )


@dataclass
class UHF_CCSD_LR_config:
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
    store_lHeecc: bool = False
    verbose: int = 1
    SPIN_BLOCKS: tuple[str, ...] = (
        'aa', 'bb',
        'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb',
    )  # TODO: these should be replaced with E1_Spin and E2_Spin


    def __str__(self) -> str:
        msg = ""
        msg += f"  Response threshold: {self.gmres_threshold}\n"
        msg += f"  Store CC Jacobian: {self.store_jacobian}\n"
        msg += f"  Store <ᴧ|[[H,τ_μ],τ_ν]|CC>: {self.store_lHeecc}\n"
        msg += f"  Verbose: {self.verbose}\n"
        return msg



@dataclass
class UHF_CCSD_LR:
    uhf_ccsd_data: UHF_CCSD_Data
    uhf_scf_data: Intermediates
    CONFIG: UHF_CCSD_LR_config = field(default_factory=UHF_CCSD_LR_config)

    def find_polarizabilities(self) -> Polarizability:
        if self.CONFIG.verbose > 0:
            print("Finding UHF-CCSD-LR polarizabilities.")
            print("Configuration:")
            print(self.CONFIG)

        builders_input = GeneratorsInput(
            uhf_scf_data=self.uhf_scf_data,
            uhf_ccsd_data=self.uhf_ccsd_data,
        )

        cc_electric_dipole = build_nu_bar_V_cc(input=builders_input)
        if self.CONFIG.store_jacobian:
            cc_jacobian = build_cc_jacobian(
                kwargs=builders_input,
                dims=self.assign_dims(),
            )
            t_response = self.find_t_response(
                minus_cc_jacobian=-cc_jacobian,
                cc_mu=cc_electric_dipole,
            )
        else:
            # Rain's approach
            minus_jacobian_op = Minus_UHF_CCSD_Jacobian_action(
                uhf_hf_data=self.uhf_scf_data,
                uhf_ccsd_data=self.uhf_ccsd_data,
            )

            # # Stephen's approach
            # ccsd = UHF_CCSD(scf_data=self.uhf_scf_data)
            # ccsd.data = self.uhf_ccsd_data
            # uhf_ccsd_energy = ccsd.get_energy()
            #
            # minus_jacobian_op = Minus_UHF_CCSD_Jacobian_action(
            #     uhf_hf_data=self.uhf_scf_data,
            #     uhf_ccsd_data=self.uhf_ccsd_data,
            #     cc_energy=uhf_ccsd_energy,
            # )

            t_response = self.find_t_response(
                minus_cc_jacobian=minus_jacobian_op,
                cc_mu=cc_electric_dipole
            )

        eta_mu = self._find_eta_mu()

        # TODO: all operators work only for the electric dipole operator
        pol_etaA_xB = self._build_pol_eta_X(eta_mu, t_response)
        if self.CONFIG.store_lHeecc is True:
            pol_xA_F_xB = build_pol_xA_F_xB(
                builders_input, t_res_A=t_response, t_res_B=t_response,
            )
        else:
            t_response_mbe = t_response_to_Spin_MBE(t_response)
            pol_xA_F_xB = build_pol_xA_F_xB_wo_store(
                builders_input, t_res_A=t_response_mbe, t_res_B=t_response_mbe,
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
                for spin in self.CONFIG.SPIN_BLOCKS
            ),
        )
        return pol

    def _find_eta_mu(self) -> dict[Descartes, dict[str, NDArray]]:
        """ mu stands for the electric dipole moment. It is a special case of a
        general perturbation operator. """
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
        minus_cc_jacobian: NDArray | LinearOperator,
        cc_mu: dict[Descartes, dict[str, NDArray]],
    ) -> dict[Descartes, dict[str, NDArray]]:
        dims = self.assign_dims()
        t_response_mu = {}
        for coord in CARTESIAN:
            mu = cc_mu[coord]
            rhs = np.vstack(
                tuple(
                    mu[block].reshape(-1, 1)
                    for block in self.CONFIG.SPIN_BLOCKS
                )
            )
            gmres_output = gmres(
                minus_cc_jacobian,
                rhs,
                atol=self.CONFIG.gmres_threshold,
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
            for block in self.CONFIG.SPIN_BLOCKS:
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
                for block in self.CONFIG.SPIN_BLOCKS
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


def tmp_helper(untyped_spin_mbe: dict[str, NDArray]) -> Spin_MBE:
    vector_mbe = Spin_MBE()
    vector_mbe.singles[E1_spin.aa] = untyped_spin_mbe['aa']
    vector_mbe.singles[E1_spin.bb] = untyped_spin_mbe['bb']

    vector_mbe.doubles[E2_spin.aaaa] = untyped_spin_mbe['aaaa']
    vector_mbe.doubles[E2_spin.abab] = untyped_spin_mbe['abab']
    vector_mbe.doubles[E2_spin.abba] = untyped_spin_mbe['abba']
    vector_mbe.doubles[E2_spin.baab] = untyped_spin_mbe['baab']
    vector_mbe.doubles[E2_spin.baba] = untyped_spin_mbe['baba']
    vector_mbe.doubles[E2_spin.bbbb] = untyped_spin_mbe['bbbb']

    return vector_mbe

def t_response_to_Spin_MBE(
    t_response: dict[Descartes, dict[str, NDArray]]
) -> dict[Descartes, Spin_MBE]:
    mbe_version = dict()
    for key, value in t_response.items():
        mbe_version[key] = tmp_helper(value)

    return mbe_version
