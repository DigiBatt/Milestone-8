import concurrent.futures
import pathlib

import bdat
import numpy as np
import pandas as pd
import pybamm
import pybop
import scipy.integrate as integrate
import scipy.optimize as optimize
import tqdm.auto as tqdm


def fit_chunk(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
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
            distribution=pybop.Uniform(2.5, 4.2),
            initial_value=voltage.mean(),
        )
    else:
        param_const["Open-circuit voltage [V]"] = ocv

    if R0 is None:
        param_fit["R0 [Ohm]"] = pybop.Parameter(
            distribution=pybop.LogUniform(1e-4, 1),
            initial_value=1e-2,
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["R0 [Ohm]"] = R0

    if R1 is None:
        param_fit["R1 [Ohm]"] = pybop.Parameter(
            distribution=pybop.LogUniform(1e-4, 1),
            initial_value=1e-2,
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["R1 [Ohm]"] = R1

    if R2 is None:
        param_fit["R2 [Ohm]"] = pybop.Parameter(
            distribution=pybop.LogUniform(1e-4, 1),
            initial_value=1e-2,
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["R2 [Ohm]"] = R2

    if Tau1 is None:
        param_fit["Tau1 [s]"] = pybop.Parameter(
            distribution=pybop.LogUniform(2, int(24 * 3600)),
            initial_value=100,
            transformation=pybop.LogTransformation(),
        )
    else:
        param_const["Tau1 [s]"] = Tau1

    if Tau2 is None:
        param_fit["Tau2 [s]"] = pybop.Parameter(
            distribution=pybop.LogUniform(2, int(24 * 3600)),
            initial_value=1000,
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
        multistart=50,
        # maxiter=1000,
        constraints=constraints,
    )
    optimizer = pybop.SciPyMinimize(problem, options=options)

    results = optimizer.run()
    return {
        **{
            key.replace(" [", " / ").replace("]", ""): value
            for key, value in results.best_inputs.items()
        },
        "cost": results.best_cost,
    }


def fit_chunks(data: bdat.CyclingData):
    """
    Fits all
    """
    results = []

    q = integrate.cumulative_trapezoid(data.current, data.time, initial=0)
    soc_all = (q - q.min()) / (q.max() - q.min())
    steps = list(bdat.steps.find_steps(data))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for i, step in enumerate(tqdm.tqdm(steps, desc="Submitting jobs")):
            if step.charge != 0 or (i == 0):
                continue

            if steps[i - 1].charge == 0:
                continue

            if steps[i - 1].charge > 0:
                mode = "Charge"

            elif steps[i - 1].charge < 0:
                mode = "Discharge"
            else:
                mode = "Unknown"

            # 10 second back from pulse start
            start = np.abs(data.time - (steps[i - 1].start - 10)).argmin()

            # Last is last
            if i == len(steps) - 1:
                end = step.rowEnd
            # 10 seconds into next pulse if it's different sign
            elif np.sign(steps[i - 1].charge) != np.sign(steps[i + 1].charge):
                end = np.abs(data.time - (step.end + 10)).argmin()
            # All of next pulse of same sign
            elif np.sign(steps[i - 1].charge) == np.sign(steps[i + 1].charge):
                end = np.abs(data.time - (steps[i + 1].end)).argmin()

            else:
                raise ValueError(f"Unexpected step pattern at step {i}")

            time = data.time[start:end]
            voltage = data.voltage[start:end]
            current = data.current[start:end]
            futures[executor.submit(fit_chunk, time, current, voltage)] = (
                soc_all[start:end].mean(),
                mode,
            )
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Collecting results",
        ):
            try:
                soc, mode = futures[future]
                res = future.result()
                res["State of Charge / 1"] = soc
                res["Mode"] = mode
                results.append(res)
            except Exception as e:
                tqdm.tqdm.write(f"Failed to get future result with error: {e}")
                continue

    pd.DataFrame(results).to_csv(
        pathlib.Path(__file__).resolve().parent.joinpath("data/parameters.csv"),
        index=False,
    )


if __name__ == "__main__":
    from utils import load_gitt

    data = load_gitt()
    fit_chunks(data)
