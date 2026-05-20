import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_parameters, make_splines


def circuit():
    path = (
        pathlib.Path(__file__)
        .resolve()
        .parent.joinpath("data")
        .joinpath("parameters.csv")
    )
    param = pd.read_csv(path)
    x_col = "State of Charge / 1"
    y_cols = [
        "Open-circuit voltage [V]",
        "R0 [Ohm]",
        "R1 [Ohm]",
        "R2 [Ohm]",
        "Tau1 [s]",
        "Tau2 [s]",
    ]
    fig, ax = plt.subplots(1, 6, figsize=(25, 3), constrained_layout=True)
    for i, y_col in enumerate(y_cols):
        for mode in ["Charge", "Discharge"]:
            mask = param["Mode"] == mode
            df = param.loc[mask, [x_col, y_col]]
            x = df[x_col].values
            y = df[y_col].values
            (_line,) = ax[i].plot(
                x,
                y,
                ls="none",
                marker="o",
                mfc="w",
            )
    plt.show()
    return


def _circuit():
    param = load_parameters()
    mask = param["Step index / 1"] > 3
    param = param.loc[mask]
    # smoot = smooth_parameters(param)
    fig, ax = plt.subplots(1, 6, figsize=(25, 3), constrained_layout=True)
    x_col = "State of Charge / 1"
    for i, y_col in enumerate(
        [
            "Open-circuit voltage [V]",
            "R0 [Ohm]",
            "R1 [Ohm]",
            "R2 [Ohm]",
            "Tau1 [s]",
            "Tau2 [s]",
        ]
    ):
        splines = make_splines(param, y_col, n=10, k=2)
        for mode in ["Charge", "Discharge"]:
            mask = param["Mode"] == mode
            df = param.loc[mask, [x_col, y_col]].dropna()
            x = df[x_col].values
            y = df[y_col].values
            (_line,) = ax[i].plot(
                x,
                y,
                ls="none",
                marker="o",
                mfc="w",
            )
            x_spl = np.linspace(x.min(), x.max(), 100)
            y_spl = splines[y_col][mode](x_spl)
            ax[i].plot(
                x_spl,
                y_spl,
                ls="-",
                color=_line.get_color(),
            )
        ax[i].set_xlabel("State of Charge / 1")
        ax[i].set_ylabel(y_col)
        ax[i].legend()
    # plt.figure()
    # for mode in ["Charge", "Discharge"]:
    #     mask = param["Mode"] == mode
    #     plt.plot(
    #         param.loc[mask, "Step index / 1"],
    #         param.loc[mask, "State of Charge / 1"],
    #         ls="none",
    #         marker="o",
    #     )
    plt.show()


if __name__ == "__main__":
    circuit()
