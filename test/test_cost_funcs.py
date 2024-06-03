import pytest
from jax import numpy as jnp

import ctrl_sandbox.cost_functions as cost


def test_accepts_valid_system():
    Q = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
    R = jnp.array([[1, 0], [0, 10]], dtype=float)
    Qf = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0

    state = jnp.array([0.1, 1, 10], dtype=float)
    control = jnp.array([1, 0.1], dtype=float)
    state_des = jnp.ones((3, 3), dtype=float) * 2.0
    control_des = jnp.ones((3, 2), dtype=float) * 2.0
    k_step = 1

    cost_func_obj = cost.cost_quad_x_and_u(Q, R, Qf, state_des, control_des)
    j, j_state, j_control = (
        cost_func_obj.cost_func_quad_state_and_control_scan_compatible(
            state, control, k_step
        )
    )

    state_corr = (state - state_des[k_step]).reshape(-1, 1)
    control_corr = (control - control_des[k_step]).reshape(-1, 1)
    j_expect = jnp.array(
        0.5 * (state_corr.T @ Q @ state_corr + control_corr.T @ R @ control_corr)
    )

    assert j == j_expect


def test_accepts_valid_final_system():
    Q = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
    R = jnp.array([[1, 0], [0, 10]], dtype=float)
    Qf = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0

    state = jnp.array([0.1, 1, 10], dtype=float)
    control = jnp.array([1, 0.1], dtype=float)
    state_des = jnp.ones((3, 3), dtype=float) * 2.0
    control_des = jnp.ones((3, 2), dtype=float) * 2.0
    k_step = 2

    cost_func_obj = cost.cost_quad_x_and_u(Q, R, Qf, state_des, control_des)
    j, j_state, j_control = (
        cost_func_obj.cost_func_quad_state_and_control_scan_compatible(
            state, control, k_step
        )
    )

    state_corr = (state - state_des[k_step]).reshape(-1, 1)
    control_corr = (control - control_des[k_step]).reshape(-1, 1)
    j_expect = jnp.array(0.5 * (state_corr.T @ Qf @ state_corr))

    assert j == j_expect
