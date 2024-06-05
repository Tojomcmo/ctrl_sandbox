import jax.numpy as jnp
import scipy.optimize as spopt
from typing import Callable, Tuple

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl


class dirColConfig:
    def __init__(
        self,
        cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        time_step: float,
        x_len: int,
        u_len: int,
        len_seq: int,
    ) -> None:
        self.cost_func = cost_func
        self.cont_dyn_func = cont_dyn_func
        self.time_step = time_step
        self.len_seq = len_seq
        self.x_len = x_len
        self.u_len = u_len
        pass

    def create_dyn_constraint_func(
        self,
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        integrate_scheme,
    ) -> None:
        pass

    def create_dir_col_cost_func(
        self, cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]
    ) -> None:
        pass


def calculate_dir_col_cost(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ],
    len_seq: int,
    x_len: int,
    u_len: int,
    opt_vec: jnp.ndarray,
) -> jnp.ndarray:
    x_seq = jnp.reshape(opt_vec[: (len_seq - 1) * x_len], (len_seq, x_len))
    u_seq = jnp.reshape(opt_vec[(len_seq - 1) * x_len :], (len_seq - 1, u_len))
    cost_float, _, _, _ = gen_ctrl.calculate_total_cost(cost_func, x_seq, u_seq)
    return cost_float


def calculate_hermite_simpson_col_point(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    u_kp1: jnp.ndarray,
):
    """
    Calculate the collocation constraint using hermite-simpson integration
    x_kp1-x_k = (hk/6)(f(k)+4*f(k+1/2)+f(k+1))
    """
    x_mid = interp_state_traj_hermite_simpson(cont_dyn_func, h, x_k, u_k, x_kp1, u_kp1)
    u_mid = u_k + (h * 0.5) * (u_kp1 - u_k)
    return (h / 6) * (
        cont_dyn_func(x_k, u_k)
        + 4 * cont_dyn_func(x_mid, u_mid)
        + cont_dyn_func(x_kp1, u_kp1)
    )


def interp_state_traj_hermite_simpson(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    u_kp1: jnp.ndarray,
) -> jnp.ndarray:
    return (0.5) * (x_k + x_kp1) + (h / 8) * (
        cont_dyn_func(x_k, u_k) - cont_dyn_func(x_kp1, u_kp1)
    )
