from dataclasses import dataclass
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data
from chem.hf.intermediates_builders import Intermediates
from chem.meta.coordinates import Descartes, CARTESIAN
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
from rspn.uhf_ccsd.equations.singles_singles import (
    get_singles_singles_aaaa,
    get_singles_singles_aabb,
    get_singles_singles_bbaa,
    get_singles_singles_bbbb,
)

@dataclass
class UHF_CCSD_LR:
    uhf_ccsd_data: UHF_CCSD_Data
    uhf_scf_data: Intermediates

    def find_polarizabilities(self):
        cc_jacobian = self.build_the_cc_jacobian()
        cc_electric_dipole = self.build_cc_electric_dipole_singles()
        t_response = self.find_t_response(cc_jacobian, cc_electric_dipole)
    
    def find_t_response(
        self,
        cc_jacobian: NDArray,
        cc_mu: dict[Descartes, NDArray],
    ) -> dict[Descartes, NDArray]:

        t_response_mu = {}
        for coord in CARTESIAN:
            response, exit_code = gmres(
                cc_jacobian,
                cc_mu[coord],
                rtol=1e-7,
                atol=1e-7,
            )
            if exit_code != 0:
                msg = f'gmres didn\'t find the response vector for mu {coord}'
                raise RuntimeError(msg)
            t_response_mu[coord] = response
        return t_response_mu

    def assign_dims(self):
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

    def build_cc_electric_dipole_singles(self) -> dict[Descartes, NDArray]:
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

    def _build_the_cc_dipole_helper(self, coord: Descartes):
        scf = self.uhf_scf_data
        ccsd = self.uhf_ccsd_data
        self.assign_dims()
        dim_aa = self.dims['aa']
        dim_bb = self.dims['bb']

        if coord == Descartes.x:
            aa = get_mux_aa(scf, ccsd).reshape(dim_aa)
            bb = get_mux_bb(scf, ccsd).reshape(dim_bb)
            return np.block([aa, bb])

        elif coord == Descartes.y:
            aa = get_muy_aa(scf, ccsd).reshape(dim_aa)
            bb = get_muy_bb(scf, ccsd).reshape(dim_bb)
            return np.block([aa, bb])

        elif coord == Descartes.z:
            aa = get_muz_aa(scf, ccsd).reshape(dim_aa)
            bb = get_muz_bb(scf, ccsd).reshape(dim_bb)
            return np.block([aa, bb])

        else:
            raise ValueError(f"Unknown cartesian coordinate: {coord}.")
