import pytest
import jax.numpy as jnp
import jax
from typing import Tuple, Callable

import ctrl_sandbox.dir_col_funcs as dir_col


def test_calculate_dir_col_cost_returns_exp_values():

    assert False


def test_breakout_opt_vec_returns_exp_values():
    len_seq = 10
    x_len = 4
    u_len = 2
    opt_vec = jnp.linspace(0, len_seq * (x_len + u_len) - 1, len_seq * (x_len + u_len))
    x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    x_expected = jnp.reshape(
        jnp.linspace(0, (len_seq * x_len) - 1, len_seq * x_len), (len_seq, x_len)
    )
    u_expected = jnp.reshape(
        jnp.linspace((len_seq * x_len), (len_seq * u_len) - 1, len_seq * u_len),
        (len_seq, u_len),
    )
    assert x_seq.all() == x_expected.all()
    assert u_seq.all() == u_expected.all()


def test_breakout_opt_vec_is_diff_compatible():
    def calc_cost_seq(x_seq: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
        return (x_seq.reshape(-1, 1)).T @ x_seq.reshape(-1, 1) + (
            u_seq.reshape(-1, 1)
        ).T @ u_seq.reshape(-1, 1)

    def diff_func(len_seq, x_len, u_len, opt_vec):
        x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
        return calc_cost_seq(x_seq, u_seq)

    len_seq = 5
    x_len = 4
    u_len = 2
    diff_func_curried = lambda opt_vec: diff_func(len_seq, x_len, u_len, opt_vec)
    opt_vec = jnp.linspace(0, len_seq * (x_len + u_len) - 1, len_seq * (x_len + u_len))
    jac_vec = (jax.jacfwd(diff_func_curried)(opt_vec)).reshape(-1)
    jac_vec_expect = 2 * opt_vec
    assert jac_vec.all() == jac_vec_expect.all()


def test_calculate_init_final_constraint_returns_expected_values():
    x_o = jnp.array([0.1, 0.1, 0.1, 0.1])
    x_f = jnp.array([1.1, 1.1, 1.1, 1.1])
    x_seq = jnp.ones((4, 4), dtype=float)
    ceq_o, ceq_f = dir_col._calculate_init_final_constraint(x_seq, x_o, x_f)
    ceq_o_expect = x_o - x_seq[0]
    ceq_f_expect = x_f - x_seq[-1]
    assert ceq_o.all() == ceq_o_expect.all()
    assert ceq_f.all() == ceq_f_expect.all()


def test_calc_knot_point_dyn_scan_func_returns_expected_values():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    x_k = jnp.ones((4,), dtype=float)
    u_k = jnp.ones((2,), dtype=float)
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    carry, x_dot_k = dir_col.calc_knot_point_dyn_scan_func(
        dyn_func_curried, None, (x_k, u_k)
    )
    carry_expect = None
    x_dot_k_expect = (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1, 1)
    assert carry == None
    assert x_dot_k.all() == x_dot_k_expect.all()


def test_calc_knot_point_dyn_scan_func_is_scan_compatible():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # define dyn func and curry functions
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    calc_knot_point_scan = lambda carry, seq: dir_col.calc_knot_point_dyn_scan_func(
        dyn_func_curried, carry, seq
    )
    # create scan sequence
    len_seq = 4
    x_k = jnp.ones((len_seq, 4), dtype=float)
    u_k = jnp.ones((len_seq, 2), dtype=float)
    carry, x_dot_seq = jax.lax.scan(calc_knot_point_scan, None, (x_k, u_k))
    # create expected output
    x_dot_k_expect = (
        A @ (x_k[0]).reshape(-1, 1) + B @ (u_k[0]).reshape(-1, 1)
    ).reshape(-1, 1)
    x_dot_seq_expect = jnp.vstack([x_dot_k_expect] * len_seq)
    assert carry == None
    assert x_dot_seq.all() == x_dot_seq_expect.all()


def test_calc_knot_point_dyn_scan_func_is_diff_compatible():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    def diff_func(dyn_func, x_len, xu_k: jnp.ndarray) -> jnp.ndarray:
        x_k = xu_k[:x_len]
        u_k = xu_k[x_len:]
        _, x_dot = dir_col.calc_knot_point_dyn_scan_func(dyn_func, None, (x_k, u_k))
        return x_dot

    # define dyn func and curry functions
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    x_k = jnp.ones((4,), dtype=float)
    u_k = jnp.ones((2,), dtype=float)
    xu_k = jnp.append(x_k, u_k, axis=0)
    diff_func_curried = lambda xu_k: diff_func(dyn_func_curried, len(x_k), xu_k)
    jac_array = jax.jacfwd(diff_func_curried)(xu_k)
    jac_array_expect = jnp.ones((len(x_k), len(xu_k)), dtype=float)
    assert jac_array.all() == jac_array_expect.all()


def test_calc_knot_point_dyn_scan_func_is_diff_through_scan_compatible():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    def scan_seq_func(scan_func, len_seq, x_len, u_len, xu_seq: jnp.ndarray):
        x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, xu_seq)
        _, x_dot_seq = jax.lax.scan(scan_func, None, (x_seq, u_seq))
        return x_dot_seq

    # define dyn func and curry functions
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)

    len_seq = 3
    x_seq = jnp.ones((len_seq, 4), dtype=float)
    u_seq = jnp.ones((len_seq, 2), dtype=float)
    xu_seq = jnp.append(x_seq.reshape(-1), u_seq.reshape(-1), axis=0)
    scan_func_curried = lambda carry, seqs: dir_col.calc_knot_point_dyn_scan_func(
        dyn_func_curried, None, seqs
    )
    scan_seq_func_curried = lambda xu_seq: scan_seq_func(
        scan_func_curried, len_seq, x_seq.shape[1], u_seq.shape[1], xu_seq
    )
    jac_array = jax.jacfwd(scan_seq_func_curried)(xu_seq)
    jac_array_expect = jnp.ones((len_seq, x_seq.shape[1], len(xu_seq)), dtype=float)
    assert jac_array.shape == jac_array_expect.shape
    # TODO create assert statement for the expected tensor values


def test_interp_state_m_p_H_S_post_dyn_calc_returns_expected_values():
    h = 0.001
    x_k = jnp.ones((2,), dtype=float)
    x_kp1 = jnp.ones((2,), dtype=float) * 2
    dyn_k = jnp.ones((2,), dtype=float)
    dyn_kp1 = jnp.ones((2,), dtype=float) * 2
    x_m_p = dir_col.interp_state_m_p_H_S_post_dyn_calc(h, x_k, x_kp1, dyn_k, dyn_kp1)
    val_exp = (0.5) * (1 + 2) + (h / 8) * (1 - 2)
    x_m_p_expected = jnp.array([val_exp, val_exp])
    assert x_m_p.shape == x_k.shape
    assert x_m_p.all() == x_m_p_expected.all()


def test_create_mid_point_dyn_calc_seqs_returns_expected_values():

    assert False
