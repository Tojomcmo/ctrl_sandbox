import jax.numpy as jnp
from jax import lax
from typing import Tuple, Callable

import gen_ctrl_funcs as gen_ctrl


def sim_dyn(
    discrete_dyn_func: Callable[[int, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    control_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    measure_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    disturb_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    noise_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    x_init: jnp.ndarray,
    seq_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    seq = jnp.arange(0, seq_len - 1, dtype=int)
    carry_init = (x_init, x_init)
    scan_func = curry_sim_dyn_scan_func(
        discrete_dyn_func, control_func, measure_func, disturb_func, noise_func
    )
    (_, _), (x_seq, x_est_seq, u_seq) = lax.scan(scan_func, carry_init, seq)
    x_seq = jnp.append(x_init.reshape(1, -1), x_seq, axis=0)
    return x_seq, x_est_seq, u_seq


def sim_dyn_scan_func(
    discrete_dyn_func: Callable[[int, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    control_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    measure_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    disturb_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    noise_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    carry: Tuple[jnp.ndarray, jnp.ndarray],
    seq: jnp.ndarray,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    # TODO fix typecheck error for k int as a ndarray of ints
    x_k, x_est_k = carry
    k = seq
    u_k = control_func(k, x_est_k)  # type:ignore
    x_kp1_pre_disturb = discrete_dyn_func(k, x_k, u_k)  # type:ignore
    x_kp1 = disturb_func(k, x_kp1_pre_disturb)  # type:ignore
    y_kp1 = noise_func(k, x_kp1)  # type:ignore
    x_est_kp1 = measure_func(k, y_kp1)  # type:ignore
    return (x_kp1, x_est_kp1), (x_k, x_est_k, u_k)  # type:ignore


def curry_sim_dyn_scan_func(
    discrete_dyn_func: Callable[[int, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    control_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    measure_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    disturb_func: Callable[[int, jnp.ndarray], jnp.ndarray],
    noise_func: Callable[[int, jnp.ndarray], jnp.ndarray],
):
    def sim_dyn_scan_func_curried(carry, seq):
        return sim_dyn_scan_func(
            discrete_dyn_func,
            control_func,
            measure_func,
            disturb_func,
            noise_func,
            carry,
            seq,
        )

    return sim_dyn_scan_func_curried


def direct_pass(int, vec: jnp.ndarray) -> jnp.ndarray:
    return vec


def prep_disturb_noise_WGN_for_sim(
    process_cov_mat: jnp.ndarray, noise_cov_mat: jnp.ndarray
) -> Tuple[
    Callable[[int, jnp.ndarray], jnp.ndarray], Callable[[int, jnp.ndarray], jnp.ndarray]
]:
    def disturb_func(k: int, x_k: jnp.ndarray) -> jnp.ndarray:
        return gen_ctrl.apply_cov_noise_to_vec(x_k, process_cov_mat)

    def noise_func(k: int, y_k: jnp.ndarray) -> jnp.ndarray:
        return gen_ctrl.apply_cov_noise_to_vec(y_k, noise_cov_mat)

    return disturb_func, noise_func


def prep_ff_fb_u_for_sim(
    u_ff_seq: jnp.ndarray, x_des_seq: jnp.ndarray, k_fb_seq: jnp.ndarray
) -> Callable[[int, jnp.ndarray], jnp.ndarray]:
    def control_func(k: int, x_k: jnp.ndarray) -> jnp.ndarray:
        return gen_ctrl.calculate_ff_fb_u(k_fb_seq[k], u_ff_seq[k], x_des_seq[k], x_k)

    return control_func
