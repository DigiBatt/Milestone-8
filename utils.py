import pathlib

import bdat
import pandas as pd


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
