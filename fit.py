import json
import pathlib
import uuid

import bdat
import numpy as np
import pybamm
import pybop
import scipy.integrate as integrate
import scipy.optimize as optimize
import tqdm.auto as tqdm

from utils import NumpyEncoder, load_ocv_and_tau


def save_tau_ocv(results: dict):
    path = pathlib.Path(__file__).resolve().parent.joinpath("data")
    if not path.exists():
        path.mkdir()
    with open(path.joinpath(f"tau_ocv_{uuid.uuid4()}.json"), "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    return


def save_resistances(results: dict):
    path = pathlib.Path(__file__).resolve().parent.joinpath("data")
    if not path.exists():
        path.mkdir()
    with open(path.joinpath(f"resistances_{uuid.uuid4()}.json"), "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    return


def wipe_tau_ocv():
    path = pathlib.Path(__file__).resolve().parent.joinpath("data")
    if path.exists():
        files = path.glob("tau_ocv*.json")
        for file in files:
            file.unlink()
    return


def wipe_resistances():
    path = pathlib.Path(__file__).resolve().parent.joinpath("data")
    if path.exists():
        files = path.glob("resistances*.json")
        for file in files:
            file.unlink()
    return


def fit_chunk(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    soc: np.ndarray | None = None,
    ocv: float | pybamm.Interpolant | None = None,
    R0: float | pybamm.Interpolant | None = None,
    R1: float | pybamm.Interpolant | None = None,
    R2: float | pybamm.Interpolant | None = None,
    Tau1: float | pybamm.Interpolant | None = None,
    Tau2: float | pybamm.Interpolant | None = None,
):
    model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})
    param = model.default_parameter_values
    param_const = {
        "Entropic change [V/K]": 0,
        "Upper voltage cut-off [V]": 4.5,
        "Lower voltage cut-off [V]": 2.0,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "C1 [F]": pybamm.Parameter("Tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
        "C2 [F]": pybamm.Parameter("Tau2 [s]") / pybamm.Parameter("R2 [Ohm]"),
    }
    if soc is not None:
        param_const["SoC"] = pybamm.Interpolant(
            time,
            soc,
            [pybamm.t],
        )
        param_const["Initial SoC"] = soc[0]
    param_fit = {
        "Element-1 initial overpotential [V]": pybop.Parameter(
            initial_value=0, bounds=[-1, 1]
        ),
        "Element-2 initial overpotential [V]": pybop.Parameter(
            initial_value=0, bounds=[-1, 1]
        ),
    }

    if ocv is None:
        param_fit["Open-circuit voltage [V]"] = pybop.Parameter(
            initial_value=3.7, bounds=[2.5, 4.2]
        )
    else:
        param_const["Open-circuit voltage [V]"] = ocv

    if R0 is None:
        param_fit["R0 [Ohm]"] = pybop.Parameter(initial_value=1e-2, bounds=[1e-4, 1])
    else:
        param_const["R0 [Ohm]"] = R0

    if R1 is None:
        param_fit["R1 [Ohm]"] = pybop.Parameter(initial_value=1e-2, bounds=[1e-4, 1])
    else:
        param_const["R1 [Ohm]"] = R1

    if R2 is None:
        param_fit["R2 [Ohm]"] = pybop.Parameter(initial_value=1e-2, bounds=[1e-4, 1])
    else:
        param_const["R2 [Ohm]"] = R2

    if Tau1 is None:
        param_fit["Tau1 [s]"] = pybop.Parameter(
            initial_value=100,
            bounds=[2, 10**5],
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["Tau1 [s]"] = Tau1

    if Tau2 is None:
        param_fit["Tau2 [s]"] = pybop.Parameter(
            initial_value=1000,
            bounds=[2, 10**5],
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["Tau2 [s]"] = Tau2

    param.update(param_const)
    param.update(param_fit)
    dataset = pybop.Dataset(
        {"Time [s]": time, "Voltage [V]": voltage, "Current [A]": -current}
    )
    cost = pybop.RootMeanSquaredError(dataset)
    simulator = pybop.pybamm.Simulator(model, protocol=dataset, parameter_values=param)
    problem = pybop.Problem(simulator, cost)

    # Time constant constraints: Tau1 <= Tau2 <= ... <= TauN
    if ("Tau1 [s]" in param_fit) and ("Tau2 [s]" in param_fit):
        keys = list(problem.parameters.keys())
        nparams = len(keys)
        A = np.zeros((1, nparams))
        idx = keys.index("Tau1 [s]")
        jdx = keys.index("Tau2 [s]")
        A[0, jdx] = 1.0
        A[0, idx] = -1.0
        constraints = [optimize.LinearConstraint(A, lb=0, ub=np.inf)]
    else:
        constraints = None

    options = pybop.SciPyMinimizeOptions(
        method="trust-constr",
        multistart=10,
        # maxiter=10,
        constraints=constraints,
    )
    optimizer = pybop.SciPyMinimize(problem, options=options)

    tqdm.tqdm.write("Starting optimizer with parameters ...")
    for key, value in param.items():
        tqdm.tqdm.write(f"{key}: {value}")
    tqdm.tqdm.write("\n")
    results = optimizer.run()
    return {**results.best_inputs, "cost": results.best_cost}


def fit_timeconstants(data: bdat.CyclingData):
    """
    Fits the time constants of the GITT data using a simple exponential decay model.
    """

    # We know the data spans SOC=[0,1], so we cheat a little here
    q = integrate.cumulative_trapezoid(data.current, data.time, initial=0)
    soc = (q - q.min()) / (q.max() - q.min())
    steps = list(bdat.steps.find_steps(data))

    for i, step in enumerate(tqdm.tqdm(steps, desc="Fitting steps")):
        # Skip non-rest steps
        if step.charge != 0:
            continue
        start = step.rowStart
        end = step.rowEnd
        time = data.time[start:end]
        voltage = data.voltage[start:end]
        current = data.current[start:end]
        if steps[i - 1].charge > 0:
            mode = "Charge"
        elif steps[i - 1].charge < 0:
            mode = "Discharge"
        else:
            mode = "Unknown"

        try:
            results = fit_chunk(
                time - time.min(),
                current,
                voltage,
                R0=1,
                R1=1,
                R2=1,
            )
            results["Step index / 1"] = i + 1
            results["State of Charge / 1"] = soc[start:end].mean()
            results["Mode"] = mode
            save_tau_ocv(results)
        except Exception as e:
            tqdm.tqdm.write(f"Failed to fit step {i + 1} with error: {e}")
    return results


def fit_resistances(data: bdat.CyclingData):
    """
    Fits the resistances     of the GITT data using a simple exponential decay model.
    """

    ocv_tau = load_ocv_and_tau()

    # We know the data spans SOC=[0,1], so we cheat a little here
    q = integrate.cumulative_trapezoid(data.current, data.time, initial=0)
    soc = (q - q.min()) / (q.max() - q.min())
    steps = list(bdat.steps.find_steps(data))

    for i, step in enumerate(tqdm.tqdm(steps, desc="Fitting steps")):
        # Skip non-rest steps
        if step.charge == 0:
            continue
        if i == 0:
            continue

        # Go 10 seconds into the previous and next steps
        start = np.abs(data.time - (step.start - 10)).argmin()
        end = np.abs(data.time - (step.end + 10)).argmin()
        time = data.time[start:end]
        voltage = data.voltage[start:end]
        current = data.current[start:end]

        if step.charge > 0:
            mode = "Charge"
        elif step.charge < 0:
            mode = "Discharge"
        else:
            mode = "Unknown"

        mask = ocv_tau["Mode"] == mode
        ocv = np.interp(
            soc[start:end].mean(),
            ocv_tau.loc[mask, "State of Charge / 1"],
            ocv_tau.loc[mask, "Open-circuit voltage [V]"],
        )
        Tau1 = np.interp(
            soc[start:end].mean(),
            ocv_tau.loc[mask, "State of Charge / 1"],
            ocv_tau.loc[mask, "Tau1 [s]"],
        )
        Tau2 = np.interp(
            soc[start:end].mean(),
            ocv_tau.loc[mask, "State of Charge / 1"],
            ocv_tau.loc[mask, "Tau2 [s]"],
        )

        df = ocv_tau.loc[mask].sort_values("State of Charge / 1")
        try:
            results = fit_chunk(
                time,
                current,
                voltage,
                soc=soc[start:end],
                ocv=pybamm.Interpolant(
                    df["State of Charge / 1"].values,
                    df["Open-circuit voltage [V]"].values,
                    [pybamm.Variable("SoC")],
                ),
                Tau1=pybamm.Interpolant(
                    df["State of Charge / 1"].values,
                    df["Tau1 [s]"].values,
                    [pybamm.Variable("SoC")],
                ),
                Tau2=pybamm.Interpolant(
                    df["State of Charge / 1"].values,
                    df["Tau2 [s]"].values,
                    [pybamm.Variable("SoC")],
                ),
            )
            results["Step index / 1"] = i + 1
            results["State of Charge / 1"] = soc[start:end].mean()
            results["Mode"] = mode
            save_resistances(results)
        except Exception as e:
            tqdm.tqdm.write(f"Failed to fit step {i + 1} with error: {e}")
    return results


if __name__ == "__main__":
    from utils import load_gitt

    # wipe_tau_ocv()
    wipe_resistances()
    data = load_gitt()
    # fit_timeconstants(data)
    fit_resistances(data)
