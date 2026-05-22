import logging
import pathlib
from typing import Callable

import numpy as np
import pandas as pd

from .interpolant import Interpolant, InterpolationKind

logger = logging.getLogger(__name__)


class CellGrid:
    def __init__(
        self,
        n_parallel: int,
        n_series: int,
        interpolants: dict[str, str | InterpolationKind] | None = None,
        interpolants_options: dict[str, dict] | None = None,
        parameters: dict | None = None,
    ):

        self._n_parallel = n_parallel
        self._n_series = n_series

        if parameters is None:
            parameters = {}
        parameters.setdefault("Mean SOH / %", 100)
        parameters.setdefault("Std SOH / %", 0)
        parameters.setdefault("Mean SOR / %", 100)
        parameters.setdefault("Std SOR / %", 0)
        parameters.setdefault("Initial SOC / %", 50)
        parameters.setdefault("Nominal capacity / Ah", 1)
        parameters.setdefault("Hysteresis soc-constant / %", 10)  # SOC-based
        self._parameters = parameters

        self._SOH = np.random.lognormal(
            mean=np.log(parameters["Mean SOH / %"] / 100),
            sigma=parameters["Std SOH / %"] / 100,
            size=(n_series, n_parallel),
        )

        self._SOR = np.random.lognormal(
            mean=np.log(parameters["Mean SOR / %"] / 100),
            sigma=parameters["Std SOR / %"] / 100,
            size=(n_series, n_parallel),
        )

        _interp = {}
        _opts = {}
        keys = [
            "Open-circuit voltage / V",
            "R0 / Ohm",
            "R1 / Ohm",
            "R2 / Ohm",
            "Tau1 / s",
            "Tau2 / s",
        ]

        if interpolants is None:
            interpolants = {}
        if interpolants_options is None:
            interpolants_options = {}
        for key in keys:
            interpolants.setdefault(key, InterpolationKind.LUT)
            if isinstance(interpolants[key], str):
                interpolants[key] = InterpolationKind(interpolants[key])
            interpolants_options.setdefault(key, {})
            if isinstance(interpolants[key], str):
                interpolants[key] = InterpolationKind(interpolants[key])

        path = pathlib.Path(__file__).parent.parent / "data/parameters.csv"
        param = pd.read_csv(path).drop_duplicates()
        interpolators = {}
        for key in keys:
            interpolators[key] = {}
            for mode in ["Charge", "Discharge"]:
                mask = param["Mode"] == mode
                x = param.loc[mask, "State of Charge / 1"].values
                y = param.loc[mask, key].values

                interpolators[key][mode] = Interpolant(
                    x, y, kind=interpolants[key], options=interpolants_options[key]
                )
        self._interpolators = interpolators
        return

    @property
    def n_parallel(self) -> int:
        return self._n_parallel

    @property
    def n_series(self) -> int:
        return self._n_series

    @property
    def nominal_capacity(self) -> np.ndarray:
        return self.parameters["Nominal capacity / Ah"]

    @property
    def interpolators(self) -> dict[str, dict[str, Callable]]:
        return self._interpolators

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def SOH(self) -> np.ndarray:
        return self._SOH

    @property
    def SOR(self) -> np.ndarray:
        return self._SOR

    @property
    def x0(self) -> np.ndarray:
        """
        Initial state vector for the cell grid.
            SOC
            POL-1
            POL-2
            HYST

        """
        x = np.zeros((4, self.n_series, self.n_parallel))
        x[0] = self.parameters["Initial SOC / %"] / 100
        x[1] = 0.0
        x[2] = 0.0
        x[3] = 0.0
        return x

    def split_x(self, x: np.ndarray) -> np.ndarray:
        return dict(
            zip(
                [
                    "State of Charge / 1",
                    "Element-1 overpotential / V",
                    "Element-2 overpotential / V",
                    "Hysteresis / 1",
                ],
                x,
            )
        )

    def hmix(self, h: np.ndarray, chg: np.ndarray, dhg: np.ndarray) -> np.ndarray:

        alpha = (1 - h) / 2
        beta = (1 + h) / 2
        return alpha * chg + beta * dhg

    def ode(self, states: np.ndarray, current: np.ndarray) -> np.ndarray:

        if current.shape == (1, 1):
            current = current.flatten()
            current = np.full((self.n_series, self.n_parallel), current)

        if current.shape != (self.n_series, self.n_parallel):
            raise ValueError(
                f"Current shape {current.shape} is not compatible with expected shape {(self.n_series, self.n_parallel)}"
            )

        x = self.split_x(states)
        soc = x["State of Charge / 1"]
        pol1 = x["Element-1 overpotential / V"]
        pol2 = x["Element-2 overpotential / V"]
        hyst = x["Hysteresis / 1"]

        R1 = (
            self.hmix(
                hyst,
                self.interpolators["R1 / Ohm"]["Charge"](soc),
                self.interpolators["R1 / Ohm"]["Discharge"](soc),
            )
            * self.SOR
        )
        R2 = (
            self.hmix(
                hyst,
                self.interpolators["R2 / Ohm"]["Charge"](soc),
                self.interpolators["R2 / Ohm"]["Discharge"](soc),
            )
            * self.SOR
        )
        Tau1 = (
            self.hmix(
                hyst,
                self.interpolators["Tau1 / s"]["Charge"](soc),
                self.interpolators["Tau1 / s"]["Discharge"](soc),
            )
            * self.SOR
        )
        Tau2 = (
            self.hmix(
                hyst,
                self.interpolators["Tau2 / s"]["Charge"](soc),
                self.interpolators["Tau2 / s"]["Discharge"](soc),
            )
            * self.SOR
        )

        dSOC = current / self.parameters["Nominal capacity / Ah"] / self.SOH / 3600
        dPOL1 = (current * R1 - pol1) / Tau1
        dPOL2 = (current * R2 - pol2) / Tau2
        tH = self.parameters["Hysteresis soc-constant / %"] / 100
        dhystdSOC = (
            np.abs(np.sign(current)) * np.sign(current) * (np.sign(current) - hyst) / tH
        )
        dhyst = dhystdSOC * dSOC

        return np.stack([dSOC, dPOL1, dPOL2, dhyst], axis=0)

    def voltage(self, states: np.ndarray, current: np.ndarray) -> np.ndarray:
        x = self.split_x(states)
        soc = x["State of Charge / 1"]
        pol1 = x["Element-1 overpotential / V"]
        pol2 = x["Element-2 overpotential / V"]
        hyst = x["Hysteresis / 1"]
        ocv = self.hmix(
            hyst,
            self.interpolators["Open-circuit voltage / V"]["Charge"](soc),
            self.interpolators["Open-circuit voltage / V"]["Discharge"](soc),
        )
        R0 = (
            self.hmix(
                hyst,
                self.interpolators["R0 / Ohm"]["Charge"](soc),
                self.interpolators["R0 / Ohm"]["Discharge"](soc),
            )
            * self.SOR
        )
        return ocv + current * R0 + pol1 + pol2
