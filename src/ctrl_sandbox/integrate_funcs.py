"""
**Module for integration functions**
"""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

import numpy.typing as npt
from typing import Callable


def step_euler_forward(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
) -> jnp.ndarray:
    """**Perform single step euler forward integration**"""
    return x_k + dyn_func(x_k, u_k) * h


def step_rk4(
    dyn_func_ad: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
) -> jnp.ndarray:
    """**Perform single step Runge Kutta 4th order integration**"""
    f1 = dyn_func_ad(x_k, u_k)
    f2 = dyn_func_ad(x_k + 0.5 * h * f1, u_k)
    f3 = dyn_func_ad(x_k + 0.5 * h * f2, u_k)
    f4 = dyn_func_ad(x_k + h * f3, u_k)
    return x_k + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


def step_solve_ivp(
    dyn_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray],
    h: float,
    x_k: npt.NDArray[np.float64],
    u_k: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """**Perform single step integration using scipy initial value problem**"""
    time_span = (0, h)
    dyn_func_zoh = lambda time_dyn, state: dyn_func(state, u_k.reshape(-1))
    y_0 = (x_k).reshape(-1)
    result_ivp = solve_ivp(dyn_func_zoh, time_span, y_0)
    x_kp1 = (np.array([result_ivp.y[:, -1]])).reshape(-1, 1)
    return x_kp1
