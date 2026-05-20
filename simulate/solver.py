from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def jacobian(
    fun: Callable,
    x: np.ndarray,
    eps: float | None = None,
    sparse: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray | csr_matrix:
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
        jac = csr_matrix(jac)
    return jac


def newton(
    fun: Callable,
    x0: np.ndarray,
    tol: float = 1e-8,
    maxiter: int = 100,
    eps: float | None = None,
    sparse: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray:
    """
    Generic newton's method for solving nonlinear equations.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sparse:
        linsolve = spsolve
    else:
        linsolve = np.linalg.solve

    b = fun(x0, *args, **kwargs)
    itr = 0
    while (np.linalg.norm(b) > tol) and (itr < maxiter):
        J = jacobian(fun, x0, eps=eps, sparse=sparse, args=args, kwargs=kwargs)
        dx = linsolve(J, -b)
        x0 = x0 + dx
        itr += 1
        b = fun(x0, *args, **kwargs)
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
        newton_options.setdefault("tol", 1e-8)
        newton_options.setdefault("maxiter", 100)
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

        return newton(wrapper, z0, **self.newton_options)

    def solve(
        self,
        x0: np.ndarray,
        z0: np.ndarray,
        u: np.ndarray,
        dt: float,
        calc_ic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        for k, dt in enumerate(np.diff(t), start=1):
            x0, z0 = self.solve(x0, z0, u[k], dt)
            x.append(x0)
            z.append(z0)
            y.append(self.h(x0, z0, u[k]))
        return x, z, y
