import json
import pathlib

import bdat
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, make_lsq_spline


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.item()
        return super().default(o)


def load_gitt():
    """
    Reads the GITT data from the DigiBatt CG50 battery and returns a bdat CyclingData object.
    """
    path = (
        pathlib.Path(__file__).parent
        / "data"
        / "DigiBatt-BAK-5000-N21700CG-006-GITT-data.csv"
    )

    data = (
        pd.read_csv(path)
        .assign(Time_s=lambda x: x["Time_h"] * 3600)
        .rename(
            columns={
                "Time_s": "Test Time / s",
                "I_A": "Current / A",
                "U_V": "Voltage / V",
            }
        )[["Test Time / s", "Current / A", "Voltage / V"]]
    )

    batterytype = bdat.BatterySpecies(
        "BAK CG50", capacity=5.0, endOfChargeVoltage=4.2, endOfDischargeVoltage=2.5
    )
    battery = bdat.Battery("DigiBatt_CG50_1072", type=batterytype)
    cycling = bdat.Cycling("GITT", object=battery)

    dataspec = bdat.get_dataspec(cycling, data)
    return bdat.CyclingData(cycling, data, dataspec)


def load_ocv_and_tau():
    """
    Loads the ECM parameters
        - Open-circuit voltage [V]
        - Tau1 [s]
        - Tau2 [s]
    from the JSON files in the data directory and returns them as a pandas dataframe
    """
    data = []
    for file in pathlib.Path(__file__).parent.joinpath("data").glob("tau_ocv_*.json"):
        with open(file, "r") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


def load_resistances():
    """
    Loads the ECM parameters
        - R0 [Ohm]
        - R1 [Ohm]
        - R2 [Ohm]
    from the JSON files in the data directory and returns them as a pandas dataframe
    """
    data = []
    for file in (
        pathlib.Path(__file__).parent.joinpath("data").glob("resistances_*.json")
    ):
        with open(file, "r") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


def load_parameters():
    """
    Loads all parameters from the JSON files in the data directory and returns them as a pandas dataframe
    """
    ocv_tau = load_ocv_and_tau()
    res = load_resistances()
    return pd.concat([ocv_tau, res], ignore_index=True)


def make_luts(parameters: pd.DataFrame, y_cols: str | list[str] | None = None) -> dict:
    x_col = "State of Charge / 1"

    if y_cols is None:
        y_cols = [
            "Open-circuit voltage [V]",
            "R0 [Ohm]",
            "R1 [Ohm]",
            "R2 [Ohm]",
            "Tau1 [s]",
            "Tau2 [s]",
        ]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    modes = ["Charge", "Discharge"]
    luts = {}
    for y_col in y_cols:
        luts[y_col] = {}
        for mode in modes:
            df = parameters.loc[parameters["Mode"] == mode, [x_col, y_col]].dropna()
            x = df[x_col].values
            y = df[y_col].values
            luts[y_col][mode] = interp1d(
                x, y, bounds_error=False, fill_value="extrapolate"
            )
    return luts


def make_splines(
    parameters: pd.DataFrame,
    y_cols: str | list[str] | None = None,
    k: int = 2,
    n: int = 20,
) -> dict:
    x_col = "State of Charge / 1"
    if y_cols is None:
        y_cols = [
            "Open-circuit voltage [V]",
            "R0 [Ohm]",
            "R1 [Ohm]",
            "R2 [Ohm]",
            "Tau1 [s]",
            "Tau2 [s]",
        ]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    modes = ["Charge", "Discharge"]
    splines = {}
    for y_col in y_cols:
        splines[y_col] = {}
        for mode in modes:
            df = (
                parameters.loc[parameters["Mode"] == mode, [x_col, y_col]]
                .dropna()
                .sort_values(x_col)
            )
            x = df[x_col].values
            y = df[y_col].values

            t = np.r_[
                [x.min()] * (k + 1),
                np.linspace(x.min(), x.max(), n)[1:-1],
                [x.max()] * (k + 1),
            ]

            splines[y_col][mode] = make_lsq_spline(x, y, t, k=k)
    return splines


def make_constants(
    parameters: pd.DataFrame, y_cols: str | list[str] | None = None
) -> dict:
    if y_cols is None:
        y_cols = parameters.columns
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    constants = {}
    for col in y_cols:
        constants[col] = {}
        for mode in parameters["Mode"].unique():
            mask = parameters["Mode"] == mode
            df = parameters.loc[mask, col].dropna()
            constants[col][mode] = lambda soc, df=df: np.full_like(soc, df.mean())
    return constants
