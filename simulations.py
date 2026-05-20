from dataclasses import dataclass

import numpy as np

from simulate.module import Module
from simulate.solver import DAESolver


@dataclass
class SimulationResults:
    module: Module
    t: np.ndarray
    u: np.ndarray
    x: list[np.ndarray]
    z: list[np.ndarray]
    y: list[np.ndarray]

    def __post_init__(self):
        states = [self.module.split_x(x) for x in self.x]
        self.cell_soc = np.array([s["State of Charge / 1"] for s in states])
        self.cell_pol1 = np.array([s["Element-1 overpotential / V"] for s in states])
        self.cell_pol2 = np.array([s["Element-2 overpotential / V"] for s in states])
        self.cell_hyst = np.array([s["Hysteresis / 1"] for s in states])
        self.cell_voltage = np.array(self.y).flatten()

        variabels = [self.module.split_z(z) for z in self.z]
        self.cell_current = np.array([v["Cell current / A"] for v in variabels])

        self.module_current = self.u
        self.module_soc = (
            self.cell_soc.sum(axis=1).sum(axis=1)
            / self.module.n_series
            / self.module.n_parallel
        )
        self.module_voltage = np.array(
            [
                v["Positive terminal potential / V"]
                - v["Negative terminal potential / V"]
                for v in variabels
            ]
        )


def dcir(module: Module) -> SimulationResults:
    solver = DAESolver(module.ode, module.alg, module.obs, module.x0, module.z0)

    u_rest = np.zeros(30)
    u_pulse = np.ones(10)
    c_rate = [0.5, 1.0, 1.5, 2.0, 2.5]
    u_steps = [u_rest]
    for c in c_rate:
        u_steps.extend(
            [
                u_pulse * c * module.nominal_capacity,
                u_rest,
                -u_pulse * c * module.nominal_capacity,
                u_rest,
            ]
        )
    u = np.hstack(u_steps)
    t = np.arange(u.size)
    return SimulationResults(module, t, u, *solver.integrate(t, u))


def cycle(
    module: Module,
    t_rest: int = 600,
    c_charge: float = 1.0,
    c_discharge: float = 1.0,
    n_cycles: int = 2,
    dod: float = 0.8,
) -> SimulationResults:
    t_charge = int(module.nominal_capacity / c_charge * 3600 * dod)
    t_discharge = int(module.nominal_capacity / c_discharge * 3600 * dod)
    u_rest = np.zeros(t_rest)
    u_charge = np.ones(t_charge) * c_charge * module.nominal_capacity
    u_discharge = -np.ones(t_discharge) * c_discharge * module.nominal_capacity
    u_steps = [u_rest]
    for _ in range(n_cycles):
        u_steps.extend([u_charge, u_rest, u_discharge, u_rest])
    u = np.hstack(u_steps)
    t = np.arange(u.size)
    solver = DAESolver(module.ode, module.alg, module.obs, module.x0, module.z0)
    return SimulationResults(module, t, u, *solver.integrate(t, u))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    module = Module(
        n_parallel=5,
        n_series=3,
        cell_parameters={"Std SOR / %": 10, "Std SOH / %": 2},
        interpolants={"Tau1": "constant", "Tau2": "constant"},
    )
    sim = dcir(module)
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(15, 5))
    print(sim.module_soc.shape)
    print(sim.cell_soc.shape)
    axs[0, 0].plot(sim.t, sim.module_current)
    axs[0, 1].plot(sim.t, sim.module_voltage)
    axs[0, 2].plot(sim.t, sim.module_soc)

    axs[1, 0].plot(sim.t, sim.cell_current.reshape(sim.t.size, -1))
    axs[1, 1].plot(sim.t, sim.cell_voltage.reshape(sim.t.size, -1))
    axs[1, 2].plot(sim.t, sim.cell_soc.reshape(sim.t.size, -1))
    plt.show()
# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from enum import Enum
# from typing import Callable, Tuple

# import numpy as np

# from numtools import solve
# from utils import load_parameters, make_constants, make_luts, make_splines

# logger = logging.getLogger(__name__)


# class Interpolation(Enum):
#     LUT = "lut"
#     SPLINE = "spline"
#     CONSTANT = "constant"


# INTERPOLATION_DISPATCH = {
#     Interpolation.LUT: make_luts,
#     Interpolation.SPLINE: make_splines,
#     Interpolation.CONSTANT: make_constants,
# }


# @dataclass
# class SimulationResults:
#     time: np.ndarray
#     current: np.ndarray
#     voltage: np.ndarray
#     soc: np.ndarray


# class CellGrid:
#     def __init__(
#         self,
#         n_parallel: int,
#         n_series: int,
#         interpolation: Interpolation | dict[str, Interpolation] = Interpolation.LUT,
#         parameters: dict | None = None,
#     ):

#         self._n_parallel = n_parallel
#         self._n_series = n_series

#         if parameters is None:
#             parameters = {}
#         parameters.setdefault("Mean SOH / %", 100)
#         parameters.setdefault("Std SOH / %", 0)
#         parameters.setdefault("Mean SOR / %", 100)
#         parameters.setdefault("Std SOR / %", 0)
#         parameters.setdefault("Initial SOC / %", 50)
#         parameters.setdefault("Nominal capacity / Ah", 1)
#         parameters.setdefault("Hysteresis soc-constant / %", 10)  # SOC-based
#         self._parameters = parameters

#         self._SOH = np.random.lognormal(
#             mean=np.log(parameters["Mean SOH / %"] / 100),
#             sigma=parameters["Std SOH / %"] / 100,
#             size=(n_series, n_parallel),
#         )

#         self._SOR = np.random.lognormal(
#             mean=np.log(parameters["Mean SOR / %"] / 100),
#             sigma=parameters["Std SOR / %"] / 100,
#             size=(n_series, n_parallel),
#         )

#         _interp = {}
#         keys = [
#             "Open-circuit voltage [V]",
#             "R0 [Ohm]",
#             "R1 [Ohm]",
#             "R2 [Ohm]",
#             "Tau1 [s]",
#             "Tau2 [s]",
#         ]
#         for key in keys:
#             if isinstance(interpolation, Interpolation):
#                 _interp[key] = interpolation
#             elif isinstance(interpolation, dict) and (key not in interpolation):
#                 _interp[key] = Interpolation.LUT
#             elif isinstance(interpolation, dict) and (key in interpolation):
#                 _interp[key] = interpolation[key]
#         self._interpolation = _interp

#         param = load_parameters()
#         interpolators = {}
#         for col in keys:
#             interpolators.update(
#                 INTERPOLATION_DISPATCH[self._interpolation[col]](param, y_cols=col)
#             )
#         self._interpolators = interpolators

#         logger.debug(
#             f"Initialized CellGrid with n_parallel={n_parallel}, n_series={n_series}, interpolation={interpolation}, parameters={parameters}"
#         )
#         logger.debug(f"SOH: {self._SOH}")
#         logger.debug(f"SOR: {self._SOR}")
#         return

#     @property
#     def n_parallel(self) -> int:
#         return self._n_parallel

#     @property
#     def n_series(self) -> int:
#         return self._n_series

#     @property
#     def nominal_capacity(self) -> np.ndarray:
#         return self.parameters["Nominal capacity / Ah"]

#     @property
#     def interpolation(self) -> dict[str, Interpolation]:
#         return self._interpolation

#     @property
#     def interpolators(self) -> dict[str, dict[str, Callable]]:
#         return self._interpolators

#     @property
#     def parameters(self) -> dict:
#         return self._parameters

#     @property
#     def SOH(self) -> np.ndarray:
#         return self._SOH

#     @property
#     def SOR(self) -> np.ndarray:
#         return self._SOR

#     @property
#     def x0(self) -> np.ndarray:
#         """
#         Initial state vector for the cell grid.
#             SOC
#             POL-1
#             POL-2
#             HYST

#         """
#         x = np.zeros((4, self.n_series, self.n_parallel))
#         x[0] = self.parameters["Initial SOC / %"] / 100
#         x[1] = 0.0
#         x[2] = 0.0
#         x[3] = 0.0
#         return x

#     def split_x(self, x: np.ndarray) -> np.ndarray:
#         return dict(
#             zip(
#                 [
#                     "State of Charge / 1",
#                     "Element-1 overpotential / V",
#                     "Element-2 overpotential / V",
#                     "Hysteresis / 1",
#                 ],
#                 x,
#             )
#         )

#     def hmix(self, h: np.ndarray, chg: np.ndarray, dhg: np.ndarray) -> np.ndarray:

#         alpha = (1 - h) / 2
#         beta = (1 + h) / 2
#         return alpha * chg + beta * dhg

#     def ode(self, states: np.ndarray, current: np.ndarray) -> np.ndarray:

#         if current.shape == (1, 1):
#             current = current.flatten()
#             current = np.full((self.n_series, self.n_parallel), current)

#         if current.shape != (self.n_series, self.n_parallel):
#             raise ValueError(
#                 f"Current shape {current.shape} is not compatible with expected shape {(self.n_series, self.n_parallel)}"
#             )

#         x = self.split_x(states)
#         soc = x["State of Charge / 1"]
#         pol1 = x["Element-1 overpotential / V"]
#         pol2 = x["Element-2 overpotential / V"]
#         hyst = x["Hysteresis / 1"]

#         R1 = (
#             self.hmix(
#                 hyst,
#                 self.interpolators["R1 [Ohm]"]["Charge"](soc),
#                 self.interpolators["R1 [Ohm]"]["Discharge"](soc),
#             )
#             * self.SOR
#         )
#         R2 = (
#             self.hmix(
#                 hyst,
#                 self.interpolators["R2 [Ohm]"]["Charge"](soc),
#                 self.interpolators["R2 [Ohm]"]["Discharge"](soc),
#             )
#             * self.SOR
#         )
#         Tau1 = (
#             self.hmix(
#                 hyst,
#                 self.interpolators["Tau1 [s]"]["Charge"](soc),
#                 self.interpolators["Tau1 [s]"]["Discharge"](soc),
#             )
#             * self.SOR
#         )
#         Tau2 = (
#             self.hmix(
#                 hyst,
#                 self.interpolators["Tau2 [s]"]["Charge"](soc),
#                 self.interpolators["Tau2 [s]"]["Discharge"](soc),
#             )
#             * self.SOR
#         )

#         dSOC = current / self.parameters["Nominal capacity / Ah"] / self.SOH / 3600
#         dPOL1 = (current * R1 - pol1) / Tau1
#         dPOL2 = (current * R2 - pol2) / Tau2
#         dhyst = (
#             np.abs(np.sign(current))
#             * (np.sign(current) - hyst)
#             / (self.parameters["Hysteresis soc-constant / %"] / 100)
#         )

#         return np.stack([dSOC, dPOL1, dPOL2, dhyst * dSOC], axis=0)

#     def voltage(self, states: np.ndarray, current: np.ndarray) -> np.ndarray:
#         x = self.split_x(states)
#         soc = x["State of Charge / 1"]
#         pol1 = x["Element-1 overpotential / V"]
#         pol2 = x["Element-2 overpotential / V"]
#         hyst = x["Hysteresis / 1"]
#         ocv = self.hmix(
#             hyst,
#             self.interpolators["Open-circuit voltage [V]"]["Charge"](soc),
#             self.interpolators["Open-circuit voltage [V]"]["Discharge"](soc),
#         )
#         R0 = (
#             self.hmix(
#                 hyst,
#                 self.interpolators["R0 [Ohm]"]["Charge"](soc),
#                 self.interpolators["R0 [Ohm]"]["Discharge"](soc),
#             )
#             * self.SOR
#         )
#         return ocv + current * R0 + pol1 + pol2


# class Module:
#     def __init__(
#         self,
#         n_parallel: int,
#         n_series: int,
#         interpolation: Interpolation | dict[str, Interpolation] | None = None,
#         cell_parameters: dict | None = None,
#         bussbar_parameters: dict | None = None,
#     ):

#         self._cellgrid = CellGrid(
#             n_parallel=n_parallel,
#             n_series=n_series,
#             interpolation=interpolation or Interpolation.LUT,
#             parameters=cell_parameters,
#         )
#         if bussbar_parameters is None:
#             bussbar_parameters = {}
#         bussbar_parameters.setdefault("Positive terminal resistance / Ohm", 1e-3)
#         bussbar_parameters.setdefault("Negative terminal resistance / Ohm", 1e-3)
#         bussbar_parameters.setdefault("Positive terminal relative width / 1", 0)
#         bussbar_parameters.setdefault("Negative terminal relative width / 1", 0)
#         bussbar_parameters.setdefault("Mean series resistance / Ohm", 1e-2)
#         bussbar_parameters.setdefault("Std series resistance / Ohm", 0)
#         bussbar_parameters.setdefault("Mean parallel resistance / Ohm", 1e-2)
#         bussbar_parameters.setdefault("Std parallel resistance / Ohm", 0)

#         self._parameters = bussbar_parameters

#         w = np.linspace(-1, 1, n_parallel)
#         w_p = w * bussbar_parameters["Positive terminal relative width / 1"]
#         w_n = w * bussbar_parameters["Negative terminal relative width / 1"]
#         d_p = np.sqrt(1**2 + w_p**2)
#         d_n = np.sqrt(1**2 + w_n**2)

#         R_pos = bussbar_parameters["Positive terminal resistance / Ohm"] * d_p
#         R_neg = bussbar_parameters["Negative terminal resistance / Ohm"] * d_n

#         R_ser = np.random.lognormal(
#             mean=np.log(bussbar_parameters["Mean series resistance / Ohm"]),
#             sigma=bussbar_parameters["Std series resistance / Ohm"],
#             size=(n_series, n_parallel),
#         )

#         R_par = np.random.lognormal(
#             mean=np.log(bussbar_parameters["Mean parallel resistance / Ohm"]),
#             sigma=bussbar_parameters["Std parallel resistance / Ohm"],
#             size=(n_series + 1, n_parallel - 1),
#         )

#         self._series_resistance = np.vstack([R_neg[None, :], R_ser, R_pos[None, :]])
#         self._parallel_resistance = R_par

#         logger.debug(
#             f"Initialized Module with n_parallel={n_parallel}, n_series={n_series}, bussbar_parameters={bussbar_parameters}"
#         )

#         logger.debug(f"Series resistance: {self._series_resistance}")
#         logger.debug(f"Parallel resistance: {self._parallel_resistance}")
#         return

#     @classmethod
#     def from_cellgrid(
#         cls, cellgrid: CellGrid, bussbar_parameters: dict | None = None
#     ) -> Module:
#         return cls(
#             n_parallel=cellgrid.n_parallel,
#             n_series=cellgrid.n_series,
#             interpolation=cellgrid.interpolation,
#             cell_parameters=cellgrid.parameters,
#             bussbar_parameters=bussbar_parameters,
#         )

#     @property
#     def cellgrid(self) -> CellGrid:
#         return self._cellgrid

#     @property
#     def n_parallel(self) -> int:
#         return self.cellgrid.n_parallel

#     @property
#     def n_series(self) -> int:
#         return self.cellgrid.n_series

#     @property
#     def nominal_capacity(self) -> np.ndarray:
#         return self.cellgrid.nominal_capacity * self.n_parallel

#     @property
#     def cell_parameters(self) -> dict:
#         return self.cellgrid.parameters

#     @property
#     def bussbar_parameters(self) -> dict:
#         return self._parameters

#     @property
#     def series_resistance(self) -> np.ndarray:
#         return self._series_resistance

#     @property
#     def parallel_resistance(self) -> np.ndarray:
#         return self._parallel_resistance

#     @property
#     def x0(self) -> np.ndarray:
#         return self.cellgrid.x0

#     @property
#     def z0(self) -> np.ndarray:
#         """
#         Initial state vector for the bussbar.
#             J_series ...
#             J_parallel ...
#             U_electric ...

#         """
#         # Series current
#         z_s = np.zeros((self.n_series + 2) * self.n_parallel)

#         # Parallel current
#         z_p = np.zeros((self.n_series + 1) * (self.n_parallel - 1))

#         # Electric potential
#         z_e = np.zeros(2 + (self.n_series + 1) * self.n_parallel)
#         return np.hstack([z_s, z_p, z_e])

#     def split_x(self, x: np.ndarray) -> dict:
#         return self.cellgrid.split_x(x)

#     def split_z(self, z: np.ndarray) -> dict:
#         # Number of series currents
#         n_s = (self.n_series + 2) * self.n_parallel

#         # Number of parallel currents
#         n_p = (self.n_series + 1) * (self.n_parallel - 1)

#         # Number of node potentials
#         n_e = 2 + (self.n_series + 1) * self.n_parallel

#         # Series currents
#         z_s = z[:n_s]

#         # Parallel currents
#         z_p = z[n_s : n_s + n_p]

#         # Node potentials
#         z_e = z[n_s + n_p :]

#         return {
#             "Negative terminal potential / V": z_e[0],
#             "Positive terminal potential / V": z_e[-1],
#             "Inner node potential / V": z_e[1:-1].reshape(
#                 (self.n_series + 1, self.n_parallel)
#             ),
#             "Series current / A": z_s.reshape((self.n_series + 2, self.n_parallel)),
#             "Parallel current / A": z_p.reshape(
#                 (self.n_series + 1, self.n_parallel - 1)
#             ),
#             "Cell current / A": z_s.reshape((self.n_series + 2, self.n_parallel))[1:-1],
#         }

#     def ode(
#         self, states: np.ndarray, variables: np.ndarray, current: np.ndarray
#     ) -> np.ndarray:

#         z = self.split_z(variables)
#         dxdt = self.cellgrid.ode(states, z["Cell current / A"])
#         return dxdt

#     def alg(
#         self, states: np.ndarray, variables: np.ndarray, current: np.ndarray
#     ) -> np.ndarray:
#         """
#         Evaluates g in
#             dx/dt = f(x, z, u)
#             0 = g(x, z, u)
#         states are known as x, variables are known as z, and current is known as u.
#         """
#         z = self.split_z(variables)

#         J_s = z["Series current / A"]
#         J_p = z["Parallel current / A"]
#         J_c = z["Cell current / A"]
#         V_s = self.series_resistance * J_s
#         V_p = self.parallel_resistance * J_p
#         V_c = self.cellgrid.voltage(states, J_c)

#         U_i = z["Inner node potential / V"]
#         U_n = z["Negative terminal potential / V"]
#         U_p = z["Positive terminal potential / V"]

#         alg = [
#             # Negative terminal at zero volt
#             U_n,
#             # Potential difference equal voltage drop over bussbar
#             (U_i[0, :] - U_n) - V_s[0, :],
#             # Potential difference equal voltage drop over bussbar and cell current
#             (U_i[1::, :] - U_i[0:-1, :]) - (V_s[1:-1, :] + V_c),
#             # Potential difference equal voltage drop over bussbar
#             (U_p - U_i[-1, :]) - V_s[-1, :],
#             # Total current entering positive terminal equal total current
#             np.sum(J_s[-1, :]) - current,
#             # Parallel voltage drop. Positive direction left to right, top to bottom
#             (U_i[:, 1::] - U_i[:, 0:-1]) - V_p,
#         ]

#         # Sum of currents equals zero at inner nodes
#         alg_ = J_s[0:-1, :] - J_s[1::, :]
#         alg_[:, 0:-1] -= J_p
#         alg_[:, 1::] += J_p
#         alg.append(alg_)
#         residual = np.hstack([a.flatten() for a in alg])
#         return residual


# class Solver:
#     def __init__(
#         self,
#         module: Module,
#     ):
#         """
#         Initialize a DAE solver with given x0- and z0- initial conditions.
#         """
#         self._module = module
#         self._x0 = module.x0
#         self._z0 = module.z0
#         return

#     @property
#     def module(self) -> Module:
#         return self._module

#     @property
#     def f(self) -> Callable:
#         return self.module.ode

#     @property
#     def g(self) -> Callable:
#         return self.module.alg

#     @property
#     def x0(self) -> np.ndarray:
#         return self._x0

#     @property
#     def z0(self) -> np.ndarray:
#         return self._z0

#     @x0.setter
#     def x0(self, value: np.ndarray):
#         self._x0 = value
#         return

#     @z0.setter
#     def z0(self, value: np.ndarray):
#         self._z0 = value
#         return

#     def _solve_alg(
#         self, x: np.ndarray, z_guess: np.ndarray, u: np.ndarray
#     ) -> np.ndarray:
#         def objective(z):
#             return self.g(x, z, u)

#         result = solve(objective, z_guess)
#         return result

#     def calc_ic(self, x0: np.ndarray, z0: np.ndarray, u: np.ndarray) -> np.ndarray:
#         """
#         Just an exposed wrapper for the algebraic solver, to pre-calculate consistent initial conditions for z0.
#         """
#         return self._solve_alg(x0, z0, u)

#     def solve(
#         self,
#         dt: float,
#         x0: np.ndarray,
#         z0: np.ndarray,
#         u: np.ndarray,
#         ic_calc: bool = True,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Solve a DAE system on the form
#             dx/dt = f(x, z, u)
#             0 = g(x, z, u)
#         ODE is assued explicit
#         ic_calc True will pre-solve for z0 before stepping the ODE
#         """

#         if ic_calc:
#             logger.debug("Calculating consistent initial conditions for z0 ...")
#             z0 = self.calc_ic(x0, z0, u)

#         logger.debug("Updating cell states ...")
#         x = x0 + dt * self.f(x0, z0, u)

#         logger.debug("Solving algebraic equations ...")
#         z = self._solve_alg(x, z0, u)
#         return x, z

#     def integrate(
#         self,
#         t: np.ndarray,
#         u: np.ndarray,
#         x0: np.ndarray | None = None,
#         z0: np.ndarray | None = None,
#     ):
#         """
#         u should be constant over the interval
#         """
#         if x0 is None:
#             x0 = self.x0
#         if z0 is None:
#             z0 = self.z0
#         z0 = self.calc_ic(x0, z0, u[0])
#         x = [self.module.split_x(x0)]
#         z = [self.module.split_z(z0)]
#         logger.info("Starting integration ...")
#         for k, dt in enumerate(np.diff(t), start=1):
#             logger.debug(f"Step {k}/{len(t) - 1}, dt={dt} ...")
#             x0, z0 = self.solve(dt, x0, z0, u[k], ic_calc=u[k] != u[k - 1])
#             x.append(self.module.split_x(x0))
#             z.append(self.module.split_z(z0))

#         return x, z


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s %(levelname)s %(name)s: %(message)s",
#     )
#     module = Module(
#         n_parallel=5,
#         n_series=3,
#         interpolation=Interpolation.SPLINE,
#         cell_parameters={"Std SOH / %": 2, "Std SOR / %": 2},
#     )
#     solver = Solver(module)

#     k = 100
#     u = (
#         np.hstack(
#             [
#                 np.zeros(k, dtype=float),
#                 np.ones(k, dtype=float),
#                 np.zeros(k, dtype=float),
#             ]
#         )
#         * module.nominal_capacity
#     )
#     t = np.arange(u.size)
#     x, z = solver.integrate(t, u)

#     fig, axs = plt.subplots(1, 3, figsize=(20, 3), constrained_layout=True)
#     for t_, x_, z_ in zip(t, x, z):
#         soc = x_["State of Charge / 1"].flatten()
#         cur = z_["Cell current / A"].flatten()
#         vol = module.cellgrid.voltage(
#             np.stack(list(x_.values())), z_["Cell current / A"]
#         ).flatten()
#         for i, item in enumerate([soc, cur, vol]):
#             axs[i].plot(np.full(item.shape, t_), item, ls="none", marker=".", color="k")
#     plt.show()
