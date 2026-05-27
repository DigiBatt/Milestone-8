import pathlib

import bdat
import pandas as pd


def load_gitt() -> bdat.CyclingData:
    """
    Reads the GITT data from the DigiBatt CG50 battery and returns a bdat CyclingData object.
    """
    path = (
        pathlib.Path(__file__).parent
        / "data"
        / "DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet"
    )

    data = (
        pd.read_parquet(path)
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
    battery = bdat.Battery("DigiBatt-CG50-006", type=batterytype)
    cycling = bdat.Cycling("GITT", object=battery)

    dataspec = bdat.get_dataspec(cycling, data)
    return bdat.CyclingData(cycling, data, dataspec)


def load_parameters(n: int):
    path = pathlib.Path(__file__).parent / "data" / f"parameters_{n}.csv"
    return pd.read_csv(path).drop_duplicates()
