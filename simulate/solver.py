import logging
from typing import Callable

import numpy as np
import scipy.sparse as sp
import tqdm.auto as tqdm
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)


def jacobian(
    fun: Callable,
    x: np.ndarray,
    eps: float | None = None,
    sparse: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray | csc_matrix:
    """
    Generic jacobian function using central difference finite differences.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)

    n = x.shape[0]
    m = fun(x, *args, **kwargs).shape[0]
    jac = np.empty((m, n))
    h = np.zeros(n)
    for i in range(n):
        h[i] = eps
        u = fun(x + h, *args, **kwargs)
        d = fun(x - h, *args, **kwargs)
        jac[:, i] = (u - d) / (2 * eps)
        h[i] = 0.0
    if sparse:
        jac = csc_matrix(jac)
    return jac


def newton(
    fun: Callable,
    x0: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 10,
    eps: float | None = None,
    sparse: bool = False,
    jac: np.ndarray | sp.csc_matrix | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray:
    """
    Generic newton's method for solving nonlinear equations.
    """
    logger.debug(
        f"Running newton's method with tol={tol}, maxiter={maxiter}, eps={eps}, sparse={sparse}"
    )
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sparse:
        linsolve = sp.linalg.spsolve
    else:
        linsolve = np.linalg.solve

    b = fun(x0, *args, **kwargs)
    itr = 0

    if jac is not None:
        if sparse and isinstance(jac, np.ndarray):
            jac = csc_matrix(jac)
        elif (not sparse) and isinstance(jac, sp.csc_matrix):
            jac = jac.toarray()

    while (np.linalg.norm(b) > tol) and (itr < maxiter):
        if jac is None:
            J = jacobian(fun, x0, eps=eps, sparse=sparse, args=args, kwargs=kwargs)
        else:
            J = jac
        logger.debug(f"Iteration {itr}: norm of residual = {np.linalg.norm(b)}")
        dx = linsolve(J, -b)  # ty:ignore[no-matching-overload]
        x0 = x0 + dx
        itr += 1
        b = fun(x0, *args, **kwargs)
        
    logger.debug(
        f"Finished newton's method after {itr} iterations with residual {np.linalg.norm(b)}"
    )
    return x0


class DAESolver:
    """
    Simple solver class for differential-algebraic equations (DAEs) of the form
        dx/dt = f(x, z, u)
            0 = g(x, z, u)

        where x is a numpy.ndarray of arbitrary shape,
        z is 1D ndarray of algebraic variables, shape (n_z,),
        u is 1D ndarray of inputs, shape (n_u,).

        dx/dt is solved for x using a simple forward Euler method, and
        g(x, z, u) is solved for z using a a simple newton's method

    """

    def __init__(
        self,
        f: Callable,
        g: Callable,
        h: Callable,
        x0: np.ndarray,
        z0: np.ndarray,
        newton_options: dict | None = None,
    ):

        self._f = f
        self._g = g
        self._h = h
        self._x0 = x0
        self._z0 = z0
        if newton_options is None:
            newton_options = {}
        newton_options.setdefault("tol", 1e-6)
        newton_options.setdefault("maxiter", 10)
        newton_options.setdefault("eps", None)
        newton_options.setdefault("sparse", False)
        self._newton_options = newton_options
        return

    @property
    def f(self):
        return self._f

    @property
    def g(self):
        return self._g

    @property
    def h(self):
        return self._h

    @property
    def x0(self):
        return self._x0

    @property
    def z0(self):
        return self._z0

    @property
    def newton_options(self):
        return self._newton_options

    def _solve_x(
        self, x0: np.ndarray, z: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        return x0 + dt * self.f(x0, z, u)

    def _solve_z(self, x: np.ndarray, z0: np.ndarray, u: np.ndarray) -> np.ndarray:
        def wrapper(z_):
            return self.g(x, z_, u)

        if self.newton_options.get("jac", None) is None:
            self._newton_options["jac"] = jacobian(
                wrapper,
                z0,
                eps=self.newton_options.get("eps", None),
                sparse=self.newton_options.get("sparse", False),
            )
        return newton(wrapper, z0, **self.newton_options)

    def solve(
        self,
        x0: np.ndarray,
        z0: np.ndarray,
        u: np.ndarray,
        dt: float,
        calc_ic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.debug(f"Solving DAE with calc_ic={calc_ic}")
        if calc_ic:
            z0 = self._solve_z(x0, z0, u)
        x = self._solve_x(x0, z0, u, dt)
        z = self._solve_z(x, z0, u)
        return x, z

    def integrate(
        self, t: np.ndarray, u: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:

        x0 = self.x0
        z0 = self._solve_z(x0, self.z0, u[0])

        x = [x0]
        z = [z0]
        y = [self.h(x0, z0, u[0])]
        for k, dt in enumerate(
            tqdm.tqdm(np.diff(t), desc="Integrating", unit="step"), start=1
        ):
            x0, z0 = self.solve(x0, z0, u[k], dt, calc_ic=(u[k] != u[k - 1]))
            x.append(x0)
            z.append(z0)
            y.append(self.h(x0, z0, u[k]))
        return x, z, y
