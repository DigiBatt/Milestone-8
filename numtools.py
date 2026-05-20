from typing import Callable

import numpy as np
import scipy.sparse


def jacobian(
    fun: Callable,
    x: np.ndarray,
    eps: float | None = None,
    sparse: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray | scipy.sparse.csr_matrix:
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    n = x.size
    m = fun(x, *args, **kwargs).size
    jac = np.empty((m, n), dtype=float)
    dx = np.zeros(n, dtype=float)
    for i in range(n):
        dx[i] = eps
        u = fun(x + dx, *args, **kwargs)
        d = fun(x - dx, *args, **kwargs)
        jac[:, i] = (u - d) / (2 * eps)
        dx[i] = 0.0

    if sparse:
        jac = scipy.sparse.csr_matrix(jac)
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

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sparse:
        linsolve = scipy.sparse.linalg.spsolve
    else:
        linsolve = np.linalg.solve

    for _ in range(maxiter):
        b = fun(x0, *args, **kwargs)
        if np.linalg.norm(b) < tol:
            break

        J = jacobian(fun, x0, eps=eps, args=args, kwargs=kwargs, sparse=sparse)
        dx = linsolve(J, -b)
        x0 = x0 + dx

    return x0


def solve(
    fun: Callable,
    x0: np.ndarray,
    tol: float = 1e-8,
    maxiter: int = 100,
    eps: float | None = None,
    sparse: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> np.ndarray:
    return newton(
        fun,
        x0,
        tol=tol,
        maxiter=maxiter,
        eps=eps,
        sparse=sparse,
        args=args,
        kwargs=kwargs,
    )
