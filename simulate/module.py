from __future__ import annotations

import logging

import numpy as np

from .cellgrid import CellGrid
from .interpolant import InterpolationKind

logger = logging.getLogger(__name__)


class Module:
    def __init__(
        self,
        n_parallel: int,
        n_series: int,
        interpolants: dict[str, str | InterpolationKind] | None = None,
        interpolants_options: dict[str, dict] | None = None,
        cell_parameters: dict | None = None,
        bussbar_parameters: dict | None = None,
    ):

        self._cellgrid = CellGrid(
            n_parallel=n_parallel,
            n_series=n_series,
            interpolants=interpolants,
            interpolants_options=interpolants_options,
            parameters=cell_parameters,
        )
        if bussbar_parameters is None:
            bussbar_parameters = {}
        bussbar_parameters.setdefault("Positive terminal resistance / Ohm", 1e-3)
        bussbar_parameters.setdefault("Negative terminal resistance / Ohm", 1e-3)
        bussbar_parameters.setdefault("Positive terminal relative width / 1", 0)
        bussbar_parameters.setdefault("Negative terminal relative width / 1", 0)
        bussbar_parameters.setdefault("Mean series resistance / Ohm", 1e-2)
        bussbar_parameters.setdefault("Std series resistance / Ohm", 0)
        bussbar_parameters.setdefault("Mean parallel resistance / Ohm", 1e-2)
        bussbar_parameters.setdefault("Std parallel resistance / Ohm", 0)

        self._parameters = bussbar_parameters

        w = np.linspace(-1, 1, n_parallel)
        w_p = w * bussbar_parameters["Positive terminal relative width / 1"]
        w_n = w * bussbar_parameters["Negative terminal relative width / 1"]
        d_p = np.sqrt(1**2 + w_p**2)
        d_n = np.sqrt(1**2 + w_n**2)

        R_pos = bussbar_parameters["Positive terminal resistance / Ohm"] * d_p
        R_neg = bussbar_parameters["Negative terminal resistance / Ohm"] * d_n

        R_ser = np.random.lognormal(
            mean=np.log(bussbar_parameters["Mean series resistance / Ohm"]),
            sigma=bussbar_parameters["Std series resistance / Ohm"],
            size=(n_series, n_parallel),
        )

        R_par = np.random.lognormal(
            mean=np.log(bussbar_parameters["Mean parallel resistance / Ohm"]),
            sigma=bussbar_parameters["Std parallel resistance / Ohm"],
            size=(n_series + 1, n_parallel - 1),
        )

        self._series_resistance = np.vstack([R_neg[None, :], R_ser, R_pos[None, :]])
        self._parallel_resistance = R_par

        logger.debug(
            f"Initialized Module with n_parallel={n_parallel}, n_series={n_series}, bussbar_parameters={bussbar_parameters}"
        )

        logger.debug(f"Series resistance: {self._series_resistance}")
        logger.debug(f"Parallel resistance: {self._parallel_resistance}")
        return

    @property
    def cellgrid(self) -> CellGrid:
        return self._cellgrid

    @property
    def n_parallel(self) -> int:
        return self.cellgrid.n_parallel

    @property
    def n_series(self) -> int:
        return self.cellgrid.n_series

    @property
    def nominal_capacity(self) -> np.ndarray:
        return self.cellgrid.nominal_capacity * self.n_parallel

    @property
    def cell_parameters(self) -> dict:
        return self.cellgrid.parameters

    @property
    def bussbar_parameters(self) -> dict:
        return self._parameters

    @property
    def series_resistance(self) -> np.ndarray:
        return self._series_resistance

    @property
    def parallel_resistance(self) -> np.ndarray:
        return self._parallel_resistance

    @property
    def x0(self) -> np.ndarray:
        return self.cellgrid.x0

    @property
    def z0(self) -> np.ndarray:
        """
        Initial state vector for the bussbar.
            J_series ...
            J_parallel ...
            U_electric ...

        """
        # Series current
        z_s = np.zeros((self.n_series + 2) * self.n_parallel)

        # Parallel current
        z_p = np.zeros((self.n_series + 1) * (self.n_parallel - 1))

        # Electric potential
        z_e = np.zeros(2 + (self.n_series + 1) * self.n_parallel)
        return np.hstack([z_s, z_p, z_e])

    def split_x(self, x: np.ndarray) -> dict:
        return self.cellgrid.split_x(x)

    def split_z(self, z: np.ndarray) -> dict:
        # Number of series currents
        n_s = (self.n_series + 2) * self.n_parallel

        # Number of parallel currents
        n_p = (self.n_series + 1) * (self.n_parallel - 1)

        # Number of node potentials
        n_e = 2 + (self.n_series + 1) * self.n_parallel

        # Series currents
        z_s = z[:n_s]

        # Parallel currents
        z_p = z[n_s : n_s + n_p]

        # Node potentials
        z_e = z[n_s + n_p :]

        return {
            "Negative terminal potential / V": z_e[0],
            "Positive terminal potential / V": z_e[-1],
            "Inner node potential / V": z_e[1:-1].reshape(
                (self.n_series + 1, self.n_parallel)
            ),
            "Series current / A": z_s.reshape((self.n_series + 2, self.n_parallel)),
            "Parallel current / A": z_p.reshape(
                (self.n_series + 1, self.n_parallel - 1)
            ),
            "Cell current / A": z_s.reshape((self.n_series + 2, self.n_parallel))[1:-1],
        }

    def ode(
        self, states: np.ndarray, variables: np.ndarray, current: np.ndarray
    ) -> np.ndarray:

        z = self.split_z(variables)
        dxdt = self.cellgrid.ode(states, z["Cell current / A"])
        return dxdt

    def alg(
        self, states: np.ndarray, variables: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates g in
            dx/dt = f(x, z, u)
            0 = g(x, z, u)
        states are known as x, variables are known as z, and current is known as u.
        """
        z = self.split_z(variables)

        J_s = z["Series current / A"]
        J_p = z["Parallel current / A"]
        J_c = z["Cell current / A"]
        V_s = self.series_resistance * J_s
        V_p = self.parallel_resistance * J_p
        V_c = self.cellgrid.voltage(states, J_c)

        U_i = z["Inner node potential / V"]
        U_n = z["Negative terminal potential / V"]
        U_p = z["Positive terminal potential / V"]

        alg = [
            # Negative terminal at zero volt
            U_n,
            # Potential difference equal voltage drop over bussbar
            (U_i[0, :] - U_n) - V_s[0, :],
            # Potential difference equal voltage drop over bussbar and cell current
            (U_i[1::, :] - U_i[0:-1, :]) - (V_s[1:-1, :] + V_c),
            # Potential difference equal voltage drop over bussbar
            (U_p - U_i[-1, :]) - V_s[-1, :],
            # Total current entering positive terminal equal total current
            np.sum(J_s[-1, :]) - current,
            # Parallel voltage drop. Positive direction left to right, top to bottom
            (U_i[:, 1::] - U_i[:, 0:-1]) - V_p,
        ]

        # Sum of currents equals zero at inner nodes
        alg_ = J_s[0:-1, :] - J_s[1::, :]
        alg_[:, 0:-1] -= J_p
        alg_[:, 1::] += J_p
        alg.append(alg_)
        residual = np.hstack([a.flatten() for a in alg])
        return residual

    def obs(
        self, states: np.ndarray, variables: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        z = self.split_z(variables)
        return self.cellgrid.voltage(states, z["Cell current / A"])
