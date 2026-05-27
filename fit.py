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
    number_of_rc_elements: int = 2,
    constants: dict | None = None,
):
    model = pybamm.equivalent_circuit.Thevenin(
        options={"number of rc elements": number_of_rc_elements}
    )
    if constants is None:
        constants = {}
    param = model.default_parameter_values
    param_const = {
        "Entropic change [V/K]": 0,
        "Upper voltage cut-off [V]": 4.5,
        "Lower voltage cut-off [V]": 2.0,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
    }
    param_fit = {}
    if "Open-circuit voltage / V" in constants:
        param_const["Open-circuit voltage [V]"] = constants["Open-circuit voltage / V"]
    else:
        param_fit["Open-circuit voltage [V]"] = pybop.Parameter(
            distribution=pybop.Uniform(2.5, 4.2),
            initial_value=voltage.mean(),
        )
    if "R0 / Ohm" in constants:
        param_const["R0 [Ohm]"] = constants["R0 / Ohm"]
    else:
        param_fit["R0 [Ohm]"] = pybop.Parameter(
            distribution=pybop.LogUniform(1e-3, 1e-1),
            initial_value=1e-2,
            transformation=pybop.LogTransformation(),
        )

    for i in range(1, number_of_rc_elements + 1):
        param_const[f"C{i} [F]"] = pybamm.Parameter(f"Tau{i} [s]") / pybamm.Parameter(
            f"R{i} [Ohm]"
        )  # ty:ignore[invalid-assignment]
        if f"R{i} / Ohm" in constants:
            param_const[f"R{i} [Ohm]"] = constants[f"R{i} / Ohm"]
        else:
            param_fit[f"R{i} [Ohm]"] = pybop.Parameter(
                distribution=pybop.LogUniform(1e-3, 1e-1),
                initial_value=1e-2,
                transformation=pybop.LogTransformation(),
            )
        if f"Tau{i} / s" in constants:
            param_const[f"Tau{i} [s]"] = constants[f"Tau{i} / s"]
        else:
            param_fit[f"Tau{i} [s]"] = pybop.Parameter(
                distribution=pybop.LogUniform(2, 10 ** (i + 2)),
                initial_value=(10 ** (i + 1)),
                transformation=pybop.LogTransformation(),
            )
        if f"Element-{i} initial overpotential [V]" in constants:
            param_const[f"Element-{i} initial overpotential [V]"] = constants[
                f"Element-{i} initial overpotential [V]"
            ]
        else:
            param_fit[f"Element-{i} initial overpotential [V]"] = pybop.Parameter(
                distribution=pybop.Uniform(-1, 1),
                initial_value=0,
            )

    param.update(param_const)
    param.update(param_fit)
    dataset = pybop.Dataset(
        {"Time [s]": time, "Voltage [V]": voltage, "Current [A]": -current}
    )
    cost = pybop.RootMeanSquaredError(dataset)
    simulator = pybop.pybamm.Simulator(model, protocol=dataset, parameter_values=param)
    problem = pybop.Problem(simulator, cost)

    if number_of_rc_elements >= 2:
        keys = list(problem.parameters.keys())
        nparams = len(keys)
        A = np.zeros((number_of_rc_elements - 1, nparams))
        for i in range(1, number_of_rc_elements):
            j = i + 1
            idx = keys.index(f"Tau{i} [s]")
            jdx = keys.index(f"Tau{j} [s]")
            A[i - 1, jdx] = 1.0
            A[i - 1, idx] = -1.0
        constraints = [optimize.LinearConstraint(A, lb=0, ub=np.inf)]
    else:
        constraints = None

    options = pybop.SciPyMinimizeOptions(
        method="trust-constr",
        multistart=30,
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


def fit_chunks(data: bdat.CyclingData, number_of_rc_elements: int = 2):
    """
    Fits all
    """
    results = []

    q = integrate.cumulative_trapezoid(data.current, data.time, initial=0)
    soc_all = (q - q.min()) / (q.max() - q.min())
    steps = list(bdat.steps.find_steps(data))

    constants = {
        # "Tau1 / s": 60,
    }
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
            futures[
                executor.submit(
                    fit_chunk,
                    time,
                    current,
                    voltage,
                    number_of_rc_elements=number_of_rc_elements,
                    constants=constants,
                )
            ] = (
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
                res.update(constants)
                results.append({**res, **constants})
            except Exception as e:
                tqdm.tqdm.write(f"Failed to get future result with error: {e}")
                continue

    pd.DataFrame(results).to_csv(
        pathlib.Path(__file__)
        .resolve()
        .parent.joinpath(f"data/parameters_{number_of_rc_elements}.csv"),
        index=False,
    )


if __name__ == "__main__":
    from utils import load_gitt

    data = load_gitt()
    for n in range(1, 4):
        fit_chunks(data, number_of_rc_elements=n)
