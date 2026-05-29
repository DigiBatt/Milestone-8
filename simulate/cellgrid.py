import logging
import pathlib

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
        number_of_rc_elements: int = 2,
    ):

        self._n_parallel = n_parallel
        self._n_series = n_series
        self._number_of_rc_elements = number_of_rc_elements

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

        self._SOH = np.random.normal(
            loc=parameters["Mean SOH / %"] / 100,
            scale=parameters["Std SOH / %"] / 100,
            size=(n_series, n_parallel),
        )

        self._SOR = np.random.normal(
            loc=parameters["Mean SOR / %"] / 100,
            scale=parameters["Std SOR / %"] / 100,
            size=(n_series, n_parallel),
        )

        _interp = {}
        _opts = {}
        keys = [
            "Open-circuit voltage / V",
            "R0 / Ohm",
        ]
        for i in range(1, number_of_rc_elements + 1):
            keys.extend(
                [
                    f"R{i} / Ohm",
                    f"Tau{i} / s",
                ]
            )

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

        path = (
            pathlib.Path(__file__).parent.parent
            / f"data/parameters_{number_of_rc_elements}.csv"
        )
        # First two chunks are bad ....
        param = (
            pd.read_csv(path)
            .drop_duplicates()
            .sort_values(by=["Mode", "State of Charge / 1"], ascending=[False, True])
            .iloc[2::]
        )
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
    def interpolators(self) -> dict[str, dict[str, Interpolant]]:
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
    def number_of_rc_elements(self) -> int:
        return self._number_of_rc_elements

    @property
    def x0(self) -> np.ndarray:
        """
        Initial state vector for the cell grid.
            SOC
            POL-1 ... POL-n
            HYST

        """
        x = np.zeros((2 + self._number_of_rc_elements, self.n_series, self.n_parallel))
        x[0] = self.parameters["Initial SOC / %"] / 100
        x[1] = 0.0  # Hysteresis is initialized to zero
        for i in range(2, self._number_of_rc_elements + 2):
            x[i] = 0.0
        return x

    def split_x(self, x: np.ndarray) -> dict[str, np.ndarray]:
        keys = ["State of Charge / 1", "Hysteresis / 1"]
        for i in range(1, self._number_of_rc_elements + 1):
            keys.append(f"Element-{i} overpotential / V")
        return {key: x[i] for i, key in enumerate(keys)}

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
        hyst = x["Hysteresis / 1"]
        pol = [
            x[f"Element-{i} overpotential / V"]
            for i in range(1, self._number_of_rc_elements + 1)
        ]
        Ri = []
        Ti = []

        for i in range(1, self._number_of_rc_elements + 1):
            Ri.append(
                self.hmix(
                    hyst,
                    self.interpolators[f"R{i} / Ohm"]["Charge"](soc),
                    self.interpolators[f"R{i} / Ohm"]["Discharge"](soc),
                )
                * self.SOR
            )
            Ti.append(
                self.hmix(
                    hyst,
                    self.interpolators[f"Tau{i} / s"]["Charge"](soc),
                    self.interpolators[f"Tau{i} / s"]["Discharge"](soc),
                )
                * self.SOR
            )

        dSOC = current / self.parameters["Nominal capacity / Ah"] / self.SOH / 3600
        dPOL = [(current * R - U) / T for R, T, U in zip(Ri, Ti, pol)]
        tH = self.parameters["Hysteresis soc-constant / %"] / 100
        dhystdSOC = (
            np.abs(np.sign(current)) * np.sign(current) * (np.sign(current) - hyst) / tH
        )
        dhyst = dhystdSOC * dSOC

        return np.stack([dSOC, dhyst, *dPOL], axis=0)

    def voltage(self, states: np.ndarray, current: np.ndarray) -> np.ndarray:
        x = self.split_x(states)
        soc = x["State of Charge / 1"]
        hyst = x["Hysteresis / 1"]
        pol = [
            x[f"Element-{i} overpotential / V"]
            for i in range(1, self._number_of_rc_elements + 1)
        ]

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
        U = 0
        for Ui in pol:
            U += Ui
        return ocv + current * R0 + U
