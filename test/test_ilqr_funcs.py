import pytest
import numpy as np
from typing import Callable, Tuple
from jax import numpy as jnp

import ctrl_sandbox.ilqr_funcs as ilqr
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.integrate_funcs as integrate

# for each input
#   test one known case
#   if discrete change of behavior (abs value add, check pos and neg)
#   check edge cases (NaN, overflow, string, divide by zero)


class shared_unit_test_data_and_funcs:
    def __init__(self) -> None:
        self.cost_func_quad_unit_params = cost.costFuncQuadStateAndControlParams(
            Q=jnp.array([[1.0, 0], [0, 1.0]]),
            R=jnp.array([[1.0]]),
            Qf=jnp.array([[1.0, 0], [0, 1.0]]),
            x_des_seq=jnp.zeros((5, 2)),
            u_des_seq=jnp.zeros((4, 1)),
        )

        self.pend_dyn_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=1.0, b=1.0, l=1.0)
        self.cost_func_obj = cost.cost_quad_x_and_u(
            self.cost_func_quad_unit_params.Q,
            self.cost_func_quad_unit_params.R,
            self.cost_func_quad_unit_params.Qf,
            self.cost_func_quad_unit_params.x_des_seq,
            self.cost_func_quad_unit_params.u_des_seq,
        )
        self.ilqr_config = ilqr.ilqrConfigStruct(
            x_len=2, u_len=1, len_seq=5, time_step=0.1
        )
        self.ilqr_config.config_cost_func(
            self.cost_func_obj.cost_func_quad_state_and_control_scan_compatible
        )
        self.ilqr_config.config_for_dyn_func(
            self.pend_dyn_obj.cont_dyn_func, integrate.step_rk4
        )
        self.ilqr_config.create_curried_funcs()

        self.seed_x_vec = jnp.ones([self.ilqr_config.x_len])
        self.seed_u_seq = jnp.ones(
            [self.ilqr_config.len_seq - 1, self.ilqr_config.u_len]
        )

        self.x_seq_example = jnp.ones([self.ilqr_config.len_seq, 2])
        self.x_seq_example = self.x_seq_example.at[:, 1].set(2.0)
        self.u_seq_example = jnp.ones([self.ilqr_config.len_seq - 1, 1])

        self.K_seq_example = jnp.ones(
            [
                self.ilqr_config.len_seq - 1,
                self.ilqr_config.u_len,
                self.ilqr_config.x_len,
            ]
        )
        self.d_seq_example = jnp.ones(
            [self.ilqr_config.len_seq - 1, self.ilqr_config.u_len, 1]
        )
        self.d_v_seq_example = jnp.ones([self.ilqr_config.len_seq, 1, 2])
        self.cost_float_example = 10.0
        self.ilqr_ctrl_state = ilqr.ilqrControllerState(
            self.ilqr_config, self.seed_x_vec, self.seed_u_seq
        )


def test_accepts_valid_linear_dyn_inputs():
    test_obj = shared_unit_test_data_and_funcs()
    ro_reg = 0.0
    k_seq, d_seq, d_v_seq, ro_reg = ilqr.calculate_backwards_pass(
        test_obj.ilqr_config,
        test_obj.x_seq_example,
        test_obj.u_seq_example,
        ro_reg,
        ro_reg_increase_bool=False,
    )
    assert len(k_seq) == test_obj.ilqr_config.len_seq - 1
    assert len(d_seq) == test_obj.ilqr_config.len_seq - 1
    assert len(d_v_seq) == test_obj.ilqr_config.len_seq - 1
    assert jnp.shape(k_seq[0]) == (
        test_obj.ilqr_config.u_len,
        test_obj.ilqr_config.x_len,
    )
    assert jnp.shape(d_seq[0]) == (test_obj.ilqr_config.u_len, 1)
    assert jnp.shape(d_v_seq[0]) == (2,)


def test_calculate_forwards_pass_accepts_valid_linear_dyn_inputs():
    test_obj = shared_unit_test_data_and_funcs()
    ilqr_config = test_obj.ilqr_config
    ilqr_ctrl_state = test_obj.ilqr_ctrl_state
    x_seq_prev = test_obj.x_seq_example
    u_seq_prev = test_obj.u_seq_example
    K_seq = test_obj.K_seq_example
    (
        x_seq_new,
        u_seq_new,
        cost_float_new,
        line_search_factor,
        ro_reg_change_bool,
    ) = ilqr.calculate_fwd_pass(
        ilqr_config,
        test_obj.x_seq_example,
        test_obj.u_seq_example,
        test_obj.K_seq_example,
        test_obj.d_seq_example,
        test_obj.d_v_seq_example,
        test_obj.cost_float_example,
    )
    assert True


def test_simulate_forward_pass_accepts_valid_system():
    def example_cost_func(x_k: jnp.ndarray, u_k: jnp.ndarray, k: int) -> jnp.ndarray:
        x = x_k.reshape(-1, 1)
        u = u_k.reshape(-1, 1)
        cost_float = x.T @ x + u.T @ u
        return cost_float

    def example_dyn_func(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        A = jnp.eye(len(x))
        B = jnp.ones((len(x), len(u)))
        x_new = A @ x.reshape(-1, 1) + B @ u.reshape(-1, 1)
        return x_new.reshape(-1)

    x_seq = jnp.ones((4, 2))
    u_seq = jnp.ones([3, 2])
    K_seq = jnp.ones([3, 2, 2])
    d_seq = jnp.ones([3, 2, 1])
    lsf = 0.5
    carry_in = (0, jnp.zeros((1), dtype=float), x_seq, lsf)
    seqs_in = (x_seq, u_seq, K_seq, d_seq)
    fp_scan_func_curried = lambda carry_in, seqs_in: ilqr.forward_pass_scan_func(
        example_dyn_func, example_cost_func, carry_in, seqs_in
    )
    x_seq_new, u_seq_new, cost_float = ilqr.simulate_forward_pass(
        fp_scan_func_curried, example_cost_func, x_seq, u_seq, K_seq, d_seq, lsf
    )
    assert True


def test_forward_pass_scan_func_accepts_valid_system():
    def example_cost_func(x_k: jnp.ndarray, u_k: jnp.ndarray, k: int) -> jnp.ndarray:
        x = x_k.reshape(-1, 1)
        u = u_k.reshape(-1, 1)
        cost_float = x.T @ x + u.T @ u
        return cost_float

    def example_dyn_func(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        A = jnp.eye(len(x))
        B = jnp.ones((len(x), len(u)))
        x_new = A @ x.reshape(-1, 1) + B @ u.reshape(-1, 1)
        return x_new.reshape(-1)

    x_k = jnp.array([1.0, 1.0])
    u_k = jnp.array([1.0, 1.0])
    K_k = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    d_k = jnp.array([[1.0], [1.0]])
    k = 0
    lsf = 0.5
    carry_in = (0, jnp.zeros(1, dtype=float), x_k, lsf)
    seqs_in = (x_k, u_k, K_k, d_k)
    carry_out, seqs_out = ilqr.forward_pass_scan_func(
        example_dyn_func, example_cost_func, carry_in, seqs_in
    )
    int_expected = 1
    u_k_new_expected = (
        u_k.reshape(-1, 1) + K_k @ (x_k.reshape(-1, 1) - x_k.reshape(-1, 1)) + lsf * d_k
    ).reshape(-1)
    cost_float_expected = example_cost_func(x_k, u_k_new_expected, k)
    x_kp1_new_expected = example_dyn_func(x_k, u_k_new_expected)
    assert carry_out[0] == int_expected
    assert carry_out[1] == cost_float_expected
    assert jnp.all(carry_out[2] == seqs_out[0])
    assert jnp.all(seqs_out[0] == x_kp1_new_expected)
    assert jnp.all(seqs_out[1] == u_k_new_expected)


def test__initialize_forwards_pass_accepts_valid_inputs():
    shared_data_funcs = shared_unit_test_data_and_funcs()
    seed_state_vec = shared_data_funcs.seed_x_vec
    u_len = shared_data_funcs.ilqr_config.u_len
    x_len = shared_data_funcs.ilqr_config.x_len
    len_seq = shared_data_funcs.ilqr_config.len_seq
    (
        state_seq_updated,
        control_seq_updated,
        cost_float_updated,
        in_bounds_bool,
    ) = ilqr._initialize_forwards_pass(x_len, u_len, seed_state_vec, len_seq)
    assert len(state_seq_updated) == len_seq
    assert len(control_seq_updated) == len_seq - 1
    assert cost_float_updated == 0
    assert not in_bounds_bool


def test_calculate_backstep_ctg_approx_accepts_valid_inputs():
    data_and_funcs = shared_unit_test_data_and_funcs()
    x_len = data_and_funcs.ilqr_config.x_len
    u_len = data_and_funcs.ilqr_config.u_len
    q_x = jnp.ones([x_len, 1])
    q_u = jnp.ones([u_len, 1])
    q_xx = jnp.ones([x_len, x_len])
    q_uu = jnp.ones([u_len, u_len])
    q_ux = jnp.ones([u_len, x_len])
    K_k = jnp.ones([u_len, x_len])
    d_k = jnp.ones([u_len, 1])
    P_k, p_k, Del_V_vec_k = ilqr.calculate_backstep_ctg_approx(
        q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k
    )
    P_k_expect = jnp.ones([x_len, x_len]) * 4.0
    p_k_expect = jnp.ones([x_len, 1]) * 4.0
    Del_V_vec_k_expect = jnp.array([1.0, 0.5])
    assert jnp.shape(P_k) == (x_len, x_len)
    assert jnp.shape(p_k) == (x_len, 1)
    assert jnp.shape(Del_V_vec_k) == (2,)
    assert P_k.tolist() == P_k_expect.tolist()
    assert p_k.tolist() == p_k_expect.tolist()
    assert Del_V_vec_k.tolist() == Del_V_vec_k_expect.tolist()


def test_calculate_optimal_gains_accepts_valid_inputs():
    q_uu = jnp.array([[1.0, 1.0], [0, 1.0]])
    q_ux = jnp.array([[1.0, 1.0], [0, 1.0]])
    q_u = jnp.array([[1.0], [1.0]])
    K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
    K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
    d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
    K_k_expected = -(jnp.array([[1.0, -1.0], [0, 1.0]])) @ (
        jnp.array([[1.0, 1.0], [0, 1.0]])
    )
    d_k_expected = -(jnp.array([[1.0, -1.0], [0, 1.0]])) @ (jnp.array([[1.0], [1.0]]))
    assert K_k.tolist() == K_k_expected.tolist()
    assert d_k.tolist() == d_k_expected.tolist()


def test_calculate_optimal_gains_accepts_valid_inputs_from_shared():
    q_uu = jnp.array([[1.0, 1.0], [0, 1.0]])
    q_ux = jnp.array([[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]])
    q_u = jnp.array([[1], [2]])
    K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
    K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
    d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
    assert K_k.tolist() == K_k_expected.tolist()
    assert d_k.tolist() == d_k_expected.tolist()


def test_calculate_u_k_new_accepts_valid_inputs():
    u_nom_k = jnp.array([1.0, 2.0])
    x_nom_k = jnp.array([1.0, 1.0])
    x_new_k = x_nom_k + 1.0
    K_k = jnp.ones([2, 2])
    d_k = jnp.ones([2, 1]) * 2.0
    line_search_factor = 1
    u_k_updated_new = ilqr.calculate_u_k_new(
        u_nom_k, x_nom_k, x_new_k, K_k, d_k, line_search_factor
    )
    u_k_expected = np.array(
        [
            1.0 + (1.0 * (2.0 - 1.0) + 1.0 * (2.0 - 1.0)) + 1.0 * 2.0,
            2.0 + (1.0 * (2.0 - 1.0) + 1.0 * (2.0 - 1.0)) + 1.0 * 2.0,
        ]
    )
    assert u_k_updated_new.tolist() == u_k_expected.tolist()


def test_calculate_u_k_new_accepts_valid_inputs_with_shared_data():
    data_and_funcs = shared_unit_test_data_and_funcs()
    k_step = 1
    u_nom_k = data_and_funcs.u_seq_example[k_step]
    x_nom_k = data_and_funcs.x_seq_example[k_step]
    x_new_k = x_nom_k + 0.1
    K_k = data_and_funcs.K_seq_example[k_step]
    d_k = data_and_funcs.d_seq_example[k_step]
    line_search_factor = 0.5
    u_k_updated_new = ilqr.calculate_u_k_new(
        u_nom_k, x_nom_k, x_new_k, K_k, d_k, line_search_factor
    )
    u_k_expected = (
        u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
    ).reshape(-1)
    assert u_k_updated_new.tolist() == u_k_expected.tolist()


def test_calculate_cost_decrease_ratio_accepts_valid_inputs():
    prev_cost_float = 100.0
    new_cost_float = 70.0
    Del_V_vec_seq = jnp.ones([3, 2])
    Del_V_vec_seq = Del_V_vec_seq.at[:, 1].set(2.0)
    line_search_factor = 2.0
    cost_decrease_ratio = ilqr.calculate_cost_decrease_ratio(
        prev_cost_float, new_cost_float, Del_V_vec_seq, line_search_factor
    )
    del_V_sum_exp = ((1 + 1 + 1) * line_search_factor) + (
        (2 + 2 + 2) * line_search_factor**2
    )
    expected_output = np.array(
        (1 / -del_V_sum_exp) * (prev_cost_float - new_cost_float)
    )
    assert cost_decrease_ratio == expected_output


def test_calculate_expected_cost_decrease_accepts_valid_inputs():
    Del_V_vec_seq = jnp.ones([3, 2])
    Del_V_vec_seq = Del_V_vec_seq.at[:, 1].set(2.0)
    line_search_factor = 2.0
    Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
    expected_output = ((1 + 1 + 1) * line_search_factor) + (
        (2 + 2 + 2) * line_search_factor**2
    )
    assert Del_V_sum == expected_output


def test_analyze_cost_decrease_returns_true_for_in_bound_inputs():
    data_and_funcs = shared_unit_test_data_and_funcs()
    bounds = data_and_funcs.ilqr_config.cost_ratio_bounds
    cost_decrease_ratio = bounds[0] + (bounds[0] + bounds[1]) * 0.5
    in_bounds_bool = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
    assert in_bounds_bool is True


def test_analyze_cost_decrease_returns_false_for_input_too_high():
    data_and_funcs = shared_unit_test_data_and_funcs()
    bounds = data_and_funcs.ilqr_config.cost_ratio_bounds
    cost_decrease_ratio = bounds[1] + 1
    in_bounds_bool = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
    assert in_bounds_bool is False


def test_analyze_cost_decrease_returns_false_for_input_too_low():
    data_and_funcs = shared_unit_test_data_and_funcs()
    bounds = data_and_funcs.ilqr_config.cost_ratio_bounds
    cost_decrease_ratio = bounds[0] - 1.0
    in_bounds_bool = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
    assert in_bounds_bool is False


def test_calculate_avg_ff_gains_accepts_valid_inputs():
    d_seq = jnp.ones((4, 2, 1))
    u_seq = jnp.ones((4, 2, 1))
    ff_avg_gains = ilqr.calculate_avg_ff_gains(d_seq, u_seq)
    ff_avg_gains_expect = 1 / (4 - 1) * (1 / (jnp.sqrt(2) + 1)) * 4
    assert ff_avg_gains == ff_avg_gains_expect
