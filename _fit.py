import bdat
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize


def relaxation_model(t: np.ndarray, U_OC: float, U_pol: np.ndarray, tau: np.ndarray):
    """
    A simple exponential decay model for the voltage relaxation during a rest step.
    """
    pol = U_pol * np.exp(-t[:, None] / tau[None, :])
    return U_OC + pol.sum(axis=1)

    # U = np.zeros_like(t) + U_OC
    # U[0] += U_pol.sum()
    # for i, dt in enumerate(np.diff(t), start=1):
    #     U_pol = U_pol + dt * (-U_pol / tau)
    #     U[i] += U_pol.sum()
    # return U


def relaxation_objective(params, t, U):
    U_OC = params[0]
    U_pol = params[1::2]
    tau = 10 ** (params[2::2])
    U_pred = relaxation_model(t, U_OC, U_pol, tau)
    return np.sum((U - U_pred) ** 2)


def fit_relaxation(t: np.ndarray, U: np.ndarray, n: int = 2):

    x0 = np.zeros(1 + 2 * n)
    x0[0] = U[-1]
    x0[1::2] = U[0] - U[-1]
    x0[2::2] = np.arange(1, n + 1, n)

    bounds = [
        (2.5, 4.2),
    ]
    for i in range(n):
        bounds.append((-1, 1))
        bounds.append((1, n + 1))
    if n >= 2:
        constraints = [
            optimize.NonlinearConstraint(
                lambda x: x[2::2][1::] - x[2::2][:-1], lb=0, ub=np.inf
            )
        ]
    else:
        constraints = None
    result = optimize.differential_evolution(
        relaxation_objective,
        bounds=bounds,
        args=(t - t.min(), U),
        constraints=constraints,
        x0=x0,
    )
    return result.x


def fit_timeconstants(data: bdat.CyclingData, n: int = 4):
    """
    Fits the time constants of the GITT data using a simple exponential decay model.
    """

    # We know the data spans SOC=[0,1], so we cheat a little here
    q = integrate.cumulative_trapezoid(data.current, data.time, initial=0)
    soc = (q - q.min()) / (q.max() - q.min())
    steps = list(bdat.steps.find_steps(data))

    for i, step in enumerate(steps):
        # Skip non-rest steps
        if step.charge != 0:
            continue
        start = step.rowStart
        end = step.rowEnd
        time = data.time[start:end]
        voltage = data.voltage[start:end]

        popt = fit_relaxation(time, voltage, n=n)
        print(popt)
    return popt


if __name__ == "__main__":
    from utils import load_gitt

    data = load_gitt()
    popt = fit_timeconstants(data, n=4)
    print("Fitted parameters:", popt)
