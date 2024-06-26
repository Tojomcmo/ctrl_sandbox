import pytest
import jax.numpy as jnp
import jax
from typing import Tuple, Callable
import numpy as np

import ctrl_sandbox.util_funcs as util
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.dir_col_funcs as dir_col


def test_calculate_dir_col_cost_returns_exp_values():
    def cost_func(
        x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x_k.reshape(-1, 1)
        u = u_k.reshape(-1, 1)
        x_cost = x.T @ x
        u_cost = u.T @ u
        total_cost = x_cost + u_cost
        return total_cost, x_cost, u_cost

    len_seq = 5
    x_len = 4
    u_len = 2
    x_seq = jnp.ones((len_seq, x_len), dtype=float)
    u_seq = jnp.ones((len_seq - 1, u_len), dtype=float)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    cost_for_calc_scan_func = gen_ctrl._curry_cost_for_calc_scan_func(cost_func, x_len)
    cost_float = dir_col.calc_dir_col_cost_pre_decorated(
        cost_for_calc_scan_func, len_seq, x_len, u_len, jnp.array(opt_vec)
    )
    cost_float_expect = jnp.array([4.0 * 5 + 2.0 * 4], dtype=float)
    np.testing.assert_array_equal(cost_float, cost_float_expect)


def test_prep_dir_col_cost_curried_for_calc_output_func_returns_expected_values():
    def cost_func(
        x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x_k.reshape(-1, 1)
        u = u_k.reshape(-1, 1)
        x_cost = x.T @ x
        u_cost = u.T @ u
        total_cost = x_cost + u_cost
        return total_cost, x_cost, u_cost

    len_seq = 5
    x_len = 4
    u_len = 2

    cost_for_calc_scan_func = gen_ctrl._curry_cost_for_calc_scan_func(cost_func, x_len)
    dir_col_cost_func_curried = lambda opt_vec: dir_col.calc_dir_col_cost_pre_decorated(
        cost_for_calc_scan_func, len_seq, x_len, u_len, jnp.array(opt_vec)
    )
    dir_col_cost_calc = dir_col.prep_dir_col_cost_curried_for_calc(
        dir_col_cost_func_curried
    )
    x_seq = np.ones((len_seq, x_len), dtype=float)
    u_seq = np.ones((len_seq - 1, u_len), dtype=float)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    cost_float = dir_col_cost_calc(opt_vec)
    cost_float_expect = 4.0 * 5 + 2.0 * 4
    assert type(cost_float) == type(cost_float_expect)
    assert cost_float == cost_float_expect


def test_prep_dir_col_cost_curried_for_diff_output_func_returns_expected_values():
    def cost_func(
        x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x_k.reshape(-1, 1)
        u = u_k.reshape(-1, 1)
        x_cost = x.T @ x
        u_cost = u.T @ u
        total_cost = x_cost + u_cost
        return total_cost, x_cost, u_cost

    len_seq = 5
    x_len = 4
    u_len = 2
    opt_vec_len = len_seq * x_len + (len_seq - 1) * u_len
    cost_for_calc_scan_func = gen_ctrl._curry_cost_for_calc_scan_func(cost_func, x_len)
    dir_col_cost_func_curried = lambda opt_vec: dir_col.calc_dir_col_cost_pre_decorated(
        cost_for_calc_scan_func, len_seq, x_len, u_len, jnp.array(opt_vec)
    )
    dir_col_cost_diff = dir_col.prep_dir_col_cost_curried_for_diff(
        dir_col_cost_func_curried
    )
    x_seq = np.ones((len_seq, x_len), dtype=float)
    u_seq = np.ones((len_seq - 1, u_len), dtype=float)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    jac_vec = dir_col_cost_diff(opt_vec)
    cost_float_expect = np.ones((opt_vec_len,), dtype=float) * 2
    np.testing.assert_array_equal(jac_vec, cost_float_expect)


def test_create_opt_vec_from_x_u_seqs_returns_expected_values():
    len_seq = 10
    x_len = 4
    u_len = 2
    opt_vec_len = len_seq * x_len + (len_seq - 1) * u_len
    x_seq = jnp.reshape(
        jnp.linspace(0, (len_seq * x_len) - 1, len_seq * x_len), (len_seq, x_len)
    )
    u_seq = jnp.reshape(
        jnp.linspace(
            (len_seq * x_len),
            opt_vec_len - 1,
            (len_seq - 1) * u_len,
        ),
        (len_seq - 1, u_len),
    )
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    opt_vec_expect = np.linspace(0, opt_vec_len - 1, opt_vec_len)
    np.testing.assert_array_almost_equal(opt_vec, opt_vec_expect, decimal=5)
    assert type(opt_vec) == type(opt_vec_expect)


def test_breakout_opt_vec_returns_exp_values():
    len_seq = 10
    x_len = 4
    u_len = 2
    opt_vec_len = len_seq * x_len + (len_seq - 1) * u_len
    opt_vec = jnp.linspace(0, opt_vec_len - 1, opt_vec_len)
    x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    x_expected = jnp.reshape(
        jnp.linspace(0, (len_seq * x_len) - 1, len_seq * x_len), (len_seq, x_len)
    )
    opt_vec_len_expect = len_seq * x_len + (len_seq - 1) * u_len
    u_expected = jnp.reshape(
        jnp.linspace(
            (len_seq * x_len),
            opt_vec_len_expect - 1,
            (len_seq - 1) * u_len,
        ),
        (len_seq - 1, u_len),
    )
    np.testing.assert_array_equal(x_seq, x_expected)
    np.testing.assert_array_almost_equal(u_seq, u_expected, decimal=5)


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
    opt_vec_len = len_seq * x_len + (len_seq - 1) * u_len
    opt_vec = jnp.linspace(
        0,
        opt_vec_len - 1,
        opt_vec_len,
    )
    jac_vec = (jax.jacfwd(diff_func_curried)(opt_vec)).reshape(-1)
    jac_vec_expect = 2 * opt_vec
    assert jac_vec.all() == jac_vec_expect.all()


def test_calc_dyn_colloc_ceqs_full_returns_expected_values():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # curry information
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    h = 0.1
    # curry input functions
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    lin_interp_mid_point_func = util.create_lin_interp_seq_mid_point_func()
    knot_point_dyn_scan_func = (
        lambda carry, seqs: dir_col.calc_knot_point_dyn_scan_func(
            dyn_func_curried, carry, seqs
        )
    )
    mid_point_dyn_scan_func = lambda carry, seqs: dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, carry, seqs
    )
    colloc_calc_scan_func = lambda carry, seqs: dir_col.calc_colloc_H_S_scan_func(
        h, carry, seqs
    )
    # create arguments
    len_seq = 5
    x_len = 4
    u_len = 2
    x_seq = jnp.ones((len_seq, x_len), dtype=float)
    u_seq = jnp.ones((len_seq - 1, u_len), dtype=float)
    x_o = jnp.zeros((x_len,), dtype=float)
    x_f = jnp.ones((x_len,), dtype=float) * 2
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    ceq_seq = dir_col.calc_dyn_colloc_ceqs_full(
        lin_interp_mid_point_func,
        knot_point_dyn_scan_func,
        mid_point_dyn_scan_func,
        colloc_calc_scan_func,
        len_seq,
        x_len,
        u_len,
        x_o,
        x_f,
        jnp.array(opt_vec),
    )
    assert ceq_seq.shape == ((len_seq + 1) * (x_len),)
    # TODO add assert statment for the seq values


def test_calc_dyn_colloc_ceqs_full_is_diff_compatible():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # curry information
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    h = 0.1
    # curry input functions
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    lin_interp_mid_point_func = util.create_lin_interp_seq_mid_point_func()
    knot_point_dyn_scan_func = (
        lambda carry, seqs: dir_col.calc_knot_point_dyn_scan_func(
            dyn_func_curried, carry, seqs
        )
    )
    mid_point_dyn_scan_func = lambda carry, seqs: dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, carry, seqs
    )
    colloc_calc_scan_func = lambda carry, seqs: dir_col.calc_colloc_H_S_scan_func(
        h, carry, seqs
    )
    # create arguments
    len_seq = 5
    x_len = 4
    u_len = 2
    x_seq = jnp.ones((len_seq, x_len), dtype=float)
    u_seq = jnp.ones((len_seq - 1, u_len), dtype=float)
    x_o = jnp.zeros((x_len,), dtype=float)
    x_f = jnp.ones((x_len,), dtype=float) * 2
    # curry func
    calculate_nonlcon = lambda opt_vec: dir_col.calc_dyn_colloc_ceqs_full(
        lin_interp_mid_point_func,
        knot_point_dyn_scan_func,
        mid_point_dyn_scan_func,
        colloc_calc_scan_func,
        len_seq,
        x_len,
        u_len,
        x_o,
        x_f,
        jnp.array(opt_vec),
    )
    # formulate primal point
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    # create test condition
    jac_array = jax.jacfwd(calculate_nonlcon)(jnp.array(opt_vec))

    assert jac_array.shape == ((len_seq + 1) * x_len, opt_vec.shape[0])
    # TODO add assert statment for the array values


def test_prep_dyn_colloc_curried_for_calc_output_returns_expected_values():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # curry information
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    h = 0.1
    # curry input functions
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    lin_interp_mid_point_func = util.create_lin_interp_seq_mid_point_func()
    knot_point_dyn_scan_func = (
        lambda carry, seqs: dir_col.calc_knot_point_dyn_scan_func(
            dyn_func_curried, carry, seqs
        )
    )
    mid_point_dyn_scan_func = lambda carry, seqs: dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, carry, seqs
    )
    colloc_calc_scan_func = lambda carry, seqs: dir_col.calc_colloc_H_S_scan_func(
        h, carry, seqs
    )
    # create arguments
    len_seq = 5
    x_len = 4
    u_len = 2
    x_o = jnp.zeros((x_len,), dtype=float)
    x_f = jnp.ones((x_len,), dtype=float) * 2
    # curry func
    calculate_nonlcon = lambda opt_vec: dir_col.calc_dyn_colloc_ceqs_full(
        lin_interp_mid_point_func,
        knot_point_dyn_scan_func,
        mid_point_dyn_scan_func,
        colloc_calc_scan_func,
        len_seq,
        x_len,
        u_len,
        x_o,
        x_f,
        jnp.array(opt_vec),
    )
    dyn_colloc_calc = dir_col.prep_dyn_colloc_curried_for_calc(calculate_nonlcon)
    x_seq = jnp.ones((len_seq, x_len), dtype=float)
    u_seq = jnp.ones((len_seq - 1, u_len), dtype=float)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    ceq_seq = dyn_colloc_calc(opt_vec)
    assert ceq_seq.shape == ((len_seq + 1) * x_len,)
    # TODO add an assert for the constraint equation values


def test_prep_dyn_colloc_curried_for_diff_output_returns_expected_values():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # curry information
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    h = 0.1
    # curry input functions
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    lin_interp_mid_point_func = util.create_lin_interp_seq_mid_point_func()
    knot_point_dyn_scan_func = (
        lambda carry, seqs: dir_col.calc_knot_point_dyn_scan_func(
            dyn_func_curried, carry, seqs
        )
    )
    mid_point_dyn_scan_func = lambda carry, seqs: dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, carry, seqs
    )
    colloc_calc_scan_func = lambda carry, seqs: dir_col.calc_colloc_H_S_scan_func(
        h, carry, seqs
    )
    # create arguments
    len_seq = 5
    x_len = 4
    u_len = 2
    x_o = jnp.zeros((x_len,), dtype=float)
    x_f = jnp.ones((x_len,), dtype=float) * 2
    # curry func
    calculate_nonlcon = lambda opt_vec: dir_col.calc_dyn_colloc_ceqs_full(
        lin_interp_mid_point_func,
        knot_point_dyn_scan_func,
        mid_point_dyn_scan_func,
        colloc_calc_scan_func,
        len_seq,
        x_len,
        u_len,
        x_o,
        x_f,
        jnp.array(opt_vec),
    )
    dyn_colloc_calc = dir_col.prep_dyn_colloc_curried_for_diff(calculate_nonlcon)
    x_seq = jnp.ones((len_seq, x_len), dtype=float)
    u_seq = jnp.ones((len_seq - 1, u_len), dtype=float)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    jac_array = dyn_colloc_calc(opt_vec)
    assert jac_array.shape == ((len_seq + 1) * x_len, len(opt_vec))
    # TODO add an assert for the constraint equation values


def test_calculate_init_final_constraint_returns_expected_values():
    x_o = jnp.array([0.1, 0.1, 0.1, 0.1])
    x_f = jnp.array([1.1, 1.1, 1.1, 1.1])
    x_seq = jnp.ones((4, 4), dtype=float)
    ceq_o, ceq_f = dir_col.calculate_init_final_constraint(x_seq, x_o, x_f)
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
        u_seq = jnp.append(
            u_seq, (jnp.zeros((u_len,), dtype=float)).reshape(1, -1), axis=0
        )
        _, x_dot_seq = jax.lax.scan(scan_func, None, (x_seq, u_seq))
        return x_dot_seq

    # define dyn func and curry functions
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)

    len_seq = 3
    x_seq = jnp.ones((len_seq, 4), dtype=float)
    u_seq = jnp.ones((len_seq - 1, 2), dtype=float)
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


def test_calc_mid_point_dyn_scan_func_returns_expected_values():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # define dynamics
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    h = 0.1
    # create sequence vector instances
    x_k = jnp.ones((4,), dtype=float)
    x_kp1 = jnp.ones((4,), dtype=float) * 2
    dyn_k = jnp.ones((4,), dtype=float)
    dyn_kp1 = jnp.ones((4,), dtype=float) * 2
    u_m_p = jnp.ones((2,), dtype=float)
    seqs = (x_k, x_kp1, dyn_k, dyn_kp1, u_m_p)
    # calculate func
    carry, dyn_m_p = dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, None, seqs
    )
    # calculate expected output
    x_m_p = dir_col.interp_state_m_p_H_S_post_dyn_calc(h, x_k, x_kp1, dyn_k, dyn_kp1)
    dyn_m_p_expect = dyn_func_curried(x_m_p, u_m_p)
    assert carry == None
    np.testing.assert_array_equal(dyn_m_p, dyn_m_p_expect)


def test_calc_mid_point_dyn_scan_func_is_scan_compatible():
    def dyn_func(
        A: jnp.ndarray, B: jnp.ndarray, x_k: jnp.ndarray, u_k: jnp.ndarray
    ) -> jnp.ndarray:
        return (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)).reshape(-1)

    # define dynamics
    A = jnp.ones((4, 4), dtype=float)
    B = jnp.ones((4, 2), dtype=float)
    dyn_func_curried = lambda x, u: dyn_func(A, B, x, u)
    h = 0.1
    scan_func_curried = lambda carry, seqs: dir_col.calc_mid_point_dyn_scan_func(
        dyn_func_curried, h, carry, seqs
    )
    # create sequence vector instances
    len_seq = 4
    x_k_seq = jnp.ones((len_seq, 4), dtype=float)
    x_kp1_seq = jnp.ones((len_seq, 4), dtype=float) * 2
    dyn_k_seq = jnp.ones((len_seq, 4), dtype=float)
    dyn_kp1_seq = jnp.ones((len_seq, 4), dtype=float) * 2
    u_m_p_seq = jnp.ones((len_seq, 2), dtype=float)
    seqs = (x_k_seq, x_kp1_seq, dyn_k_seq, dyn_kp1_seq, u_m_p_seq)
    # calculate func
    carry, dyn_m_p_seq = jax.lax.scan(scan_func_curried, None, seqs)
    # calculate expected output
    x_m_p = dir_col.interp_state_m_p_H_S_post_dyn_calc(
        h, x_k_seq[0], x_kp1_seq[0], dyn_k_seq[0], dyn_kp1_seq[0]
    )
    dyn_m_p_expect = dyn_func_curried(x_m_p, u_m_p_seq[0])
    dyn_m_p_seq_expect = jnp.vstack([dyn_m_p_expect] * len_seq)
    assert carry == None
    np.testing.assert_array_equal(dyn_m_p_seq, dyn_m_p_seq_expect)


def test_calc_mid_point_dyn_scan_func_is_diff_compatible():

    assert False


def test_calc_mid_point_dyn_scan_func_is_diff_through_scan_compatible():

    assert False


def test_calc_colloc_H_S_scan_func_returns_expected_values():
    h = 0.1
    carry = None
    x_k = jnp.ones((4,), dtype=float)
    x_kp1 = jnp.ones((4,), dtype=float) * 2
    dyn_k = jnp.ones((4,), dtype=float)
    dyn_m_p = jnp.ones((4,), dtype=float) * 2
    dyn_kp1 = jnp.ones((4,), dtype=float) * 3
    seqs = (x_k, x_kp1, dyn_k, dyn_m_p, dyn_kp1)
    carry, ceq_m_p = dir_col.calc_colloc_H_S_scan_func(h, carry, seqs)
    state_integrate_expect = dir_col.calc_H_S_integ_post_dyn_calc(
        h, dyn_k, dyn_m_p, dyn_kp1
    )
    ceq_m_p_expect = state_integrate_expect - (x_kp1 - x_k)
    np.testing.assert_array_equal(ceq_m_p, ceq_m_p_expect)


def test_calc_colloc_H_S_scan_func_is_scan_compatible():

    assert False


def test_calc_colloc_H_S_scan_func_is_diff_compatible():

    assert False


def test_calc_colloc_H_S_scan_func_is_diff_through_scan_compatible():

    assert False


def test_create_mid_point_dyn_calc_seqs_returns_expected_values():

    assert False


def test_create_colloc_calc_seqs_returns_expected_values():

    assert False


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


def test_calc_H_S_integ_post_dyn_calc_returns_expected_values():
    h = 0.001
    dyn_k = jnp.ones((2,), dtype=float)
    dyn_m_p = jnp.ones((2,), dtype=float) * 2
    dyn_kp1 = jnp.ones((2,), dtype=float) * 3
    dyn_m_p = dir_col.calc_H_S_integ_post_dyn_calc(h, dyn_k, dyn_m_p, dyn_kp1)
    val_exp = (h / 6) * (1 + 4 * 2 + 3)
    dyn_m_p_expect = jnp.array([val_exp, val_exp])
    assert dyn_m_p.shape == dyn_k.shape
    assert dyn_m_p.all() == dyn_m_p_expect.all()
