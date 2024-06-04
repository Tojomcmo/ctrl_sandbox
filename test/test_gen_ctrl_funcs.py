import pytest
from jax import numpy as jnp
from jax import lax
import numpy as np
import scipy
import numpy.typing as npt
import numpy.testing
from typing import Callable, Tuple

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.util_funcs as util


def test_simulate_forward_dynamics_seq_accepts_valid_input():
    def pend_unit_dyn_func(x_k_jax: jnp.ndarray, u_k_jax: jnp.ndarray) -> jnp.ndarray:
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array(
            [
                x_k_jax[1],
                -(b / l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g / l) + u_k_jax[0],
            ],
            dtype=float,
        )
        return state_dot

    len_seq = 4
    u_seq = jnp.ones((4, 1))
    x_init = jnp.array([1.0, 1.0])
    x_seq = gen_ctrl.simulate_forward_dynamics_seq(pend_unit_dyn_func, x_init, u_seq)
    x_seq_expected = np.zeros([5, 2])
    x_seq_expected[0] = x_init
    for idx in range(len_seq):
        x_delta = pend_unit_dyn_func(x_seq[idx], u_seq[idx])
        x_next = x_delta
        x_seq_expected[idx + 1] = x_next
    assert len(x_seq) == len_seq + 1
    assert x_seq[-1].tolist() == x_seq_expected[-1].tolist()


def test_linearize_dynamics_accepts_valid_case():
    def pend_unit_dyn_func(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array(
            [
                x_k_jax[1],
                -(b / l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g / l) + u_k_jax[0],
            ],
            dtype=float,
        )
        return state_dot

    def step_rk4(
        dyn_func_ad, h: float, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        f1 = dyn_func_ad(x_k, u_k)
        f2 = dyn_func_ad(x_k + 0.5 * h * f1, u_k)
        f3 = dyn_func_ad(x_k + 0.5 * h * f2, u_k)
        f4 = dyn_func_ad(x_k + h * f3, u_k)
        return x_k + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    h = 0.1
    x = jnp.array([1.0, 1.0])
    u = jnp.array([1.0])
    discrete_dyn_func = lambda x, u: step_rk4(pend_unit_dyn_func, h, x, u)
    xu = jnp.concatenate((x, u))
    a_lin, b_lin = gen_ctrl.linearize_dynamics(discrete_dyn_func, xu, len(x))
    g = 1.0
    l = 1.0
    b = 1.0
    state_pos = x[0]
    A_lin_expected = np.array([[0, 1], [-(g / l) * jnp.cos(state_pos), -(b / l)]])
    B_lin_expected = np.array([[0], [1]])
    A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
    B_lin_d_expected = B_lin_expected * h
    assert jnp.shape(a_lin) == (len(x), len(x))
    assert jnp.shape(b_lin) == (len(x), len(u))
    np.testing.assert_allclose(a_lin, A_lin_d_expected, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(b_lin, B_lin_d_expected, rtol=1e-1, atol=1e-1)


def test_calculate_linearized_state_space_seq_accepts_valid_system():
    def pend_unit_dyn_func(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array(
            [
                x_k_jax[1],
                -(b / l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g / l) + u_k_jax[0],
            ],
            dtype=float,
        )
        return state_dot

    def step_rk4(
        dyn_func_ad, h: float, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        f1 = dyn_func_ad(x_k, u_k)
        f2 = dyn_func_ad(x_k + 0.5 * h * f1, u_k)
        f3 = dyn_func_ad(x_k + 0.5 * h * f2, u_k)
        f4 = dyn_func_ad(x_k + h * f3, u_k)
        return x_k + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    len_seq = 4
    h = 0.005
    u_seq = jnp.ones((3, 1), dtype=float)
    x_seq = jnp.ones((4, 2), dtype=float)
    discrete_dyn_func = lambda x, u: step_rk4(pend_unit_dyn_func, h, x, u)
    A_lin_d_seq, B_lin_d_seq = gen_ctrl.calculate_linearized_state_space_seq(
        discrete_dyn_func, x_seq, u_seq
    )
    x_init_pos = x_seq[0]
    A_lin_expected = np.array(
        [[0, 1], [-(1.0 / 1.0) * np.cos(x_init_pos[0]), -(1.0 / 1.0)]]
    )
    B_lin_expected = np.array([[0], [1]])
    A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
    B_lin_d_expected = B_lin_expected * h
    assert len(A_lin_d_seq) == len_seq - 1
    assert len(B_lin_d_seq) == len_seq - 1
    np.testing.assert_allclose(A_lin_d_seq[0], A_lin_d_expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(B_lin_d_seq[0], B_lin_d_expected, rtol=1e-4, atol=1e-4)


def cost_func_quad_state_and_control_for_test_calc(
    x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Q = jnp.array([[1.0, 0], [0, 1.0]])
    R = jnp.array([[1.0]])
    x_des_seq = jnp.ones([3, 2, 1]) * 0.1
    u_des_seq = jnp.ones([2, 1, 1]) * 0.1
    x_k_corr = x_k.reshape(-1, 1) - (x_des_seq[k]).reshape(-1, 1)
    u_k_corr = u_k.reshape(-1, 1) - (u_des_seq[k]).reshape(-1, 1)
    x_cost = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
    u_cost = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
    total_cost = jnp.array(
        (0.5) * ((x_k_corr.T @ Q @ x_k_corr) + (u_k_corr.T @ R @ u_k_corr))
    )
    return total_cost.reshape(-1), x_cost.reshape(-1), u_cost.reshape(-1)


def cost_func_quad_state_and_control_for_test_diff(
    x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
) -> jnp.ndarray:
    Q = jnp.array([[1.0, 0], [0, 1.0]])
    R = jnp.array([[1.0]])
    x_des_seq = jnp.ones([3, 2, 1]) * 0.1
    u_des_seq = jnp.ones([2, 1, 1]) * 0.1
    x_k_corr = x_k.reshape(-1, 1) - (x_des_seq[k]).reshape(-1, 1)
    u_k_corr = u_k.reshape(-1, 1) - (u_des_seq[k]).reshape(-1, 1)
    total_cost = jnp.array(
        (0.5) * ((x_k_corr.T @ Q @ x_k_corr) + (u_k_corr.T @ R @ u_k_corr))
    )
    return total_cost.reshape(-1)


def test_calculate_total_cost_accepts_valid_inputs():
    cost_func_for_calc = cost_func_quad_state_and_control_for_test_calc
    x_seq = jnp.ones([3, 2])
    u_seq = jnp.ones([2, 1])
    total_cost, cost_seq, x_cost_seq, u_cost_seq = gen_ctrl.calculate_total_cost(
        cost_func_for_calc, x_seq, u_seq
    )
    total_cost_expect = (
        (0.5 * (2 * (0.9 * 0.9) + (0.9 * 0.9)))
        + (0.5 * (2 * (0.9 * 0.9) + (0.9 * 0.9)))
        + (0.5 * (2 * (0.9 * 0.9) + (0.1 * 0.1)))
    )

    cost_seq_expect = jnp.array(
        [
            (0.5 * (2 * (0.9 * 0.9) + (0.9 * 0.9))),
            (0.5 * (2 * (0.9 * 0.9) + (0.9 * 0.9))),
            (0.5 * (2 * (0.9 * 0.9) + (0.1 * 0.1))),
        ]
    )
    x_cost_seq_expect = jnp.array([(0.9 * 0.9), (0.9 * 0.9), (0.9 * 0.9)])
    u_cost_seq_expect = jnp.array([0.5 * (0.9 * 0.9), 0.5 * (0.9 * 0.9), 0.005])
    assert np.isclose(total_cost, total_cost_expect, atol=1e-6)
    np.testing.assert_allclose(cost_seq, cost_seq_expect, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x_cost_seq, x_cost_seq_expect, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(u_cost_seq, u_cost_seq_expect, rtol=1e-6, atol=1e-6)


def test_taylor_expand_cost_accepts_list_of_arrays():
    cost_func_for_diff = cost_func_quad_state_and_control_for_test_diff
    x_seq = jnp.ones([3, 2])
    u_seq = jnp.ones([2, 1])
    k = 1
    xu_seq = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
    l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost(
        cost_func_for_diff, xu_seq[k], len(x_seq[0]), k
    )  # type:ignore (Jax NDArray)
    cost_func_params = {
        "Q": np.array([[1.0, 0], [0, 1.0]]),
        "R": np.array([[1.0]]),
        "Qf": np.array([[1.0, 0], [0, 1.0]]),
    }
    x_des_seq = np.ones([3, 2]) * 0.1
    u_des_seq = np.ones([2, 1]) * 0.1
    l_x_expected = cost_func_params["Q"] @ (x_seq[k] - x_des_seq[k]).reshape(-1, 1)
    l_u_expected = cost_func_params["R"] @ (u_seq[k] - u_des_seq[k]).reshape(-1, 1)
    l_xx_expected = cost_func_params["Q"]
    l_uu_expected = cost_func_params["R"]
    l_ux_expected = np.zeros([len(u_seq[0]), len(x_seq[0])])

    np.testing.assert_allclose(l_x, l_x_expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(l_u, l_u_expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(l_xx, l_xx_expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(l_uu, l_uu_expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(l_ux, l_ux_expected, rtol=1e-7, atol=1e-7)


def test_prep_cost_func_for_diff_accepts_valid_cost_func_inputs():
    def cost_func(vec_1: jnp.ndarray, vec_2: jnp.ndarray, input_int: int):
        cost_val_1 = jnp.sum(vec_1)
        cost_val_2 = jnp.sum(vec_2)
        combined_cost = cost_val_1 + cost_val_2
        return combined_cost, cost_val_1, cost_val_2

    x_k = jnp.array([1, 2])
    u_k = jnp.array([3, 4])
    input_int = 1
    cost_func_for_diff = gen_ctrl.prep_cost_func_for_diff(cost_func)
    combined_cost = cost_func_for_diff(x_k, u_k, input_int)
    combined_cost_expected = jnp.array([10])
    assert combined_cost == combined_cost_expected
