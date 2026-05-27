import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_gitt


def data():
    cyc = load_gitt()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].plot(cyc.time / 3600, cyc.current, label="Current / A")
    axs[1].plot(cyc.time / 3600, cyc.voltage, label="Voltage / V")
    axs[0].set_xlabel("Time / h")
    axs[0].set_ylabel("Current / A")
    axs[1].set_xlabel("Time / h")
    axs[1].set_ylabel("Voltage / V")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    fig.savefig(
        pathlib.Path(__file__).parent / "figures" / "gitt_data.png",
        dpi=300,
        bbox_inches="tight",
    )
    return


def circuit():
    files = list(
        pathlib.Path(__file__).parent.joinpath("data").glob("parameters_*.csv")
    )
    fig, axs = plt.subplots(
        len(files),
        2 + 6 + 1,
        figsize=(3 * 10, 9),
        constrained_layout=True,
        squeeze=False,
    )
    x_col = "State of Charge / 1"
    y_cols = [
        "Open-circuit voltage / V",
        "R0 / Ohm",
        "R1 / Ohm",
        "R2 / Ohm",
        "R3 / Ohm",
        "Tau1 / s",
        "Tau2 / s",
        "Tau3 / s",
        "cost",
    ]
    for _, file in enumerate(files):
        i = int(file.stem.split("_")[-1]) - 1
        param = pd.read_csv(file)
        for y_col in y_cols:
            if y_col not in param.columns:
                continue
            j = y_cols.index(y_col)
            for mode in ["Charge", "Discharge"]:
                mask = param["Mode"] == mode
                df = param.loc[mask, [x_col, y_col]]
                x = df[x_col].values
                y = df[y_col].values
                (_line,) = axs[i, j].plot(
                    x,
                    y,
                    ls="none",
                    marker="o",
                    mfc="w",
                    label=mode,
                )
            axs[i, j].set_xlabel("State of Charge / 1")
            axs[i, j].set_ylabel(y_col)
            axs[i, j].legend(loc="upper right")
    for ax in axs[:, -1]:
        ax.set_ylabel("RMSE / V")
        ax.set_yscale("log")
    fig.savefig(
        pathlib.Path(__file__).parent / "figures" / "circuit_parameters_linear.png",
        dpi=300,
        bbox_inches="tight",
    )
    for ax in axs[:, 1::].ravel():
        ax.set_yscale("log")
    fig.savefig(
        pathlib.Path(__file__).parent / "figures" / "circuit_parameters_log.png",
        dpi=300,
        bbox_inches="tight",
    )
    return


if __name__ == "__main__":
    data()
    circuit()
