# https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
# https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax.numpy as jnp
from jax import lax
import copy
import numpy as np
from numpy import typing as npt
from typing import Sequence, Callable, Tuple, Union
import mujoco as mujoco
import time

import ilqr_utils as util
import gen_ctrl_funcs as gen_ctrl
import mujoco_funcs as mj_funcs
import cost_functions as cost
import dyn_functions as dyn


class ilqrConfigStruct:
    def __init__(self, x_len: int, u_len: int, len_seq: int, time_step: float) -> None:
        self.x_len: int = x_len
        self.u_len: int = u_len
        self.len_seq: int = len_seq
        self.time_step: float = time_step
        self.max_iter: int = 20
        self.converge_crit: float = 1e-6
        self.ff_gain_tol: float = 1e-4
        self.cost_ratio_bounds: Tuple[float, float] = (1e-8, 10)
        self.ro_reg_start: float = 0.0
        self.ro_reg_change: float = 2.0
        self.fp_max_iter: int = 5
        self.ls_scale_alpha_param: float = 0.5
        self.log_ctrl_history: bool = False
        self.is_dyn_configured: bool = False
        self.is_cost_configured: bool = False
        self.is_curried: bool = False

    def config_cost_func(
        self,
        cost_func: Callable[
            [jnp.ndarray, jnp.ndarray, int],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
    ) -> None:
        """
        **Configure the cost function** \n
        - prep the cost function for cost calculation and differentiation \n
        Generate he respective sequence functions for:
        - calculating taylor expansions
        - calculating total cost and cost sequences
        """
        # store different forms of cost function requred
        self.cost_func_verbose = cost_func
        self.cost_func_float = gen_ctrl.prep_cost_func_for_diff(cost_func)
        # create taylor expansion sequence function
        cost_exp_scan_fun = gen_ctrl._curry_cost_for_exp_scan_func(
            self.cost_func_float, self.x_len
        )
        self.quad_exp_cost_seq = (
            lambda x_seq, u_seq: gen_ctrl.taylor_expand_cost_seq_pre_decorated(
                cost_exp_scan_fun, x_seq, u_seq
            )
        )
        # create cost calculation sequence function
        cost_calc_scan_func = gen_ctrl._curry_cost_for_calc_scan_func(
            self.cost_func_verbose, self.x_len
        )
        self.calculate_total_cost = (
            lambda x_seq, u_seq: gen_ctrl.calculate_total_cost_pre_decorated(
                cost_calc_scan_func, x_seq, u_seq
            )
        )
        # announce completion of cost configuration
        self.is_cost_configured = True

    def config_for_dyn_func(
        self,
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        integrate_func: Callable[
            [
                Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                float,
                jnp.ndarray,
                jnp.ndarray,
            ],
            jnp.ndarray,
        ],
    ) -> None:
        """
        This function receives python functions for dynamics and prepares the curried functions for the controller to use.\n
        dyn_func[in] - callable that returns the continuous dynamics (state_dot) given current state, control, and a set of paramters\n
        dyn_func_params[in] - struct of dynamics function parameters to specify dynamic function values (mass, damping, springrate etc.)\n
        integrate_func[in] - callable that recieves the continuous dynamic function and returns the integrated value at t + timestep.
        Can be viewed as a discretizer (rk4, euler etc.)
        """
        # check whether struct already configured for dynamics
        if self.is_dyn_configured is True:
            ValueError("struct already configured for dynamics")
        # Populate discrete dynamics function
        self.cont_dyn_func = cont_dyn_func
        self.integrate_func = integrate_func
        self.is_cost_configured = True
        self.mj_ctrl: bool = False
        # TODO create update-timestep script to update currying if timestep is changed after func config

    def config_for_mujoco(self, mjcf_model) -> None:
        # check whether struct already configured for dynamics
        if self.is_dyn_configured is True:
            ValueError("struct already configured for dynamics")
        self.mj_model, _, self.mj_data = mj_funcs.create_mujoco_model(
            mjcf_model, self.time_step
        )
        self.mjcf_model = mjcf_model
        mujoco.mj_forward(self.mj_model, self.mj_data)  # type: ignore
        self.mj_ctrl: bool = True

    def create_curried_funcs(self):
        if self.is_cost_configured is False or self.is_dyn_configured is False:
            ValueError("Controller is not configured yet")
        self._curry_analyze_cost_dec_func()
        if self.mj_ctrl is True:
            self._curry_funcs_for_mujoco()
        else:
            self._curry_funcs_for_dyn_funcs()
        self.is_curried = True

    def _curry_funcs_for_mujoco(self) -> None:
        self.discrete_dyn_mj_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
            lambda x, u: mj_funcs.fwd_sim_mj_w_ctrl_step(
                self.mj_model, self.mj_data, x, u
            )
        )

        self.simulate_fwd_dyn_seq: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
            lambda x_init, u_seq: mj_funcs.fwd_sim_mj_w_ctrl(
                self.mj_model, self.mj_data, x_init, u_seq
            )
        )

        self.linearize_dyn_seq: Callable[
            [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
        ] = lambda x_seq, u_seq: mj_funcs.linearize_mj_seq(
            self.mj_model, self.mj_data, x_seq, u_seq
        )

    def _curry_funcs_for_dyn_funcs(self) -> None:
        self.discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
            lambda x, u: self.integrate_func(self.cont_dyn_func, self.time_step, x, u)
        )

        sim_dyn_scan_func = gen_ctrl._curry_dyn_for_sim_scan_func(
            self.discrete_dyn_func
        )
        self.simulate_fwd_dyn_seq: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
            lambda x_init, u_seq: gen_ctrl.simulate_forward_dynamics_seq_pre_decorated(
                sim_dyn_scan_func, x_init, u_seq
            )
        )

        lin_dyn_scan_func = gen_ctrl._curry_dyn_lin_scan_func(
            self.discrete_dyn_func, self.x_len
        )
        self.linearize_dyn_seq: Callable[
            [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
        ] = lambda x_seq, u_seq: gen_ctrl.calculate_linearized_state_space_seq_pre_decorated(
            lin_dyn_scan_func, x_seq, u_seq
        )

        fp_scan_func = _curry_forward_pass_scan_func(
            self.discrete_dyn_func, self.cost_func_float
        )
        self.simulate_forward_pass = (
            lambda x_seq, u_seq, K_seq, d_seq, lsf: simulate_forward_pass(
                fp_scan_func, self.cost_func_float, x_seq, u_seq, K_seq, d_seq, lsf
            )
        )

    def _curry_analyze_cost_dec_func(self) -> None:
        self.analyze_cost_dec: Callable[[float], bool] = (
            lambda cost_decrease_ratio: analyze_cost_decrease(
                cost_decrease_ratio, self.cost_ratio_bounds
            )
        )


class ilqrControllerState:
    def __init__(
        self,
        ilqr_config: ilqrConfigStruct,
        init_x_vec: Union[jnp.ndarray, npt.NDArray[np.float64]],
        init_u_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    ):
        # direct parameter population
        self.init_x_vec = jnp.array(init_x_vec)
        self.init_u_seq = jnp.array(init_u_seq)
        self.validate_input_data(ilqr_config)
        self.cost_float = 0.0
        self.iter_int = 0
        self.ro_reg_change_bool = False

        # dependent parameter population
        self.u_seq = self.init_u_seq
        self.len_seq = self.get_len_seq()
        self.x_len = self.get_num_states()
        self.u_len = self.get_num_controls()
        self.time_seq = jnp.arange(self.len_seq) * ilqr_config.time_step
        self.init_x_seq = self.get_init_x_seq(ilqr_config.simulate_fwd_dyn_seq)
        self.u_seq = self.init_u_seq
        self.x_seq = self.init_x_seq
        self.initialize_ctrl_state_zeros()

        if ilqr_config.log_ctrl_history is True:
            self.u_seq_history: list = []
            self.x_seq_history: list = []
            self.K_seq_history: list = []
            self.d_seq_history: list = []
            self.d_v_seq_history: list = []
            self.cost_seq_history: list = []
            self.x_cost_seq_history: list = []
            self.u_cost_seq_history: list = []
            self.cost_float_history: list = []

    def validate_input_data(self, ilqr_config: ilqrConfigStruct):
        if self.get_len_seq() != ilqr_config.len_seq:
            ValueError("sequences are the wrong length")
        if self.get_num_states() != ilqr_config.x_len:
            ValueError("incorrect state vector length")
        if self.get_num_controls() != ilqr_config.u_len:
            ValueError("incorrect control vector length")
        # TODO create validation criteria

    def get_len_seq(self):
        return len(self.init_u_seq) + 1

    def get_num_states(self):
        return len(self.init_x_vec)

    def get_num_controls(self):
        return len(self.init_u_seq[0])

    def get_init_x_seq(
        self, dyn_seq_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        return dyn_seq_func(self.init_x_vec, self.init_u_seq)

    def initialize_ctrl_state_zeros(self):
        self.K_seq = jnp.zeros([self.len_seq - 1, self.u_len, self.x_len])
        self.d_seq = jnp.zeros([self.len_seq - 1, self.u_len, 1])
        self.d_v_seq = jnp.zeros([self.len_seq - 1, 2])
        self.cost_seq = jnp.zeros([self.len_seq])
        self.x_cost_seq = jnp.zeros([self.len_seq])
        self.u_cost_seq = jnp.zeros([self.len_seq])

    def set_ctrl_state(
        self,
        iter_int: int,
        x_seq: jnp.ndarray,
        u_seq: jnp.ndarray,
        K_seq: jnp.ndarray,
        d_seq: jnp.ndarray,
        d_v_seq: jnp.ndarray,
        cost_seq: jnp.ndarray,
        x_cost_seq: jnp.ndarray,
        u_cost_seq: jnp.ndarray,
        cost_float: float,
    ):
        self.iter_int = iter_int
        self.x_seq = x_seq
        self.u_seq = u_seq
        self.K_seq = K_seq
        self.d_seq = d_seq
        self.d_v_seq = d_v_seq
        self.cost_seq = cost_seq
        self.x_cost_seq = x_cost_seq
        self.u_cost_seq = u_cost_seq
        self.cost_float = cost_float

    def log_ctrl_state_to_history(self):
        self.u_seq_history.append(self.u_seq)
        self.x_seq_history.append(self.x_seq)
        self.cost_seq_history.append(self.cost_seq)
        self.x_cost_seq_history.append(self.x_cost_seq)
        self.u_cost_seq_history.append(self.u_cost_seq)
        self.cost_float_history.append(self.cost_float)
        self.K_seq_history.append(self.K_seq)
        self.d_seq_history.append(self.d_seq)
        self.d_v_seq_history.append(self.d_v_seq)


def run_ilqr_controller(
    ilqr_config: ilqrConfigStruct, init_ctrl_state: ilqrControllerState
) -> ilqrControllerState:
    ctrl_state = copy.deepcopy(init_ctrl_state)
    u_seq = init_ctrl_state.init_u_seq
    x_seq = init_ctrl_state.init_x_seq
    cost_float, _, _, _ = ilqr_config.calculate_total_cost(x_seq, u_seq)
    converge_reached = False
    ro_reg = ilqr_config.ro_reg_start
    reg_inc_bool = False
    iter_int = 1
    while (iter_int < ilqr_config.max_iter) and (converge_reached is False):
        prev_cost_float = cost_float
        K_seq, d_seq, d_v_seq, ro_reg = calculate_backwards_pass(
            ilqr_config, x_seq, u_seq, ro_reg, reg_inc_bool
        )
        x_seq, u_seq, cost_float, lsf_float, reg_inc_bool = calculate_forwards_pass(
            ilqr_config, x_seq, u_seq, K_seq, d_seq, d_v_seq, cost_float
        )
        converge_measure, converge_reached, avg_ff_gain = (
            analyze_ilqr_pass_for_convergence(
                ilqr_config,
                iter_int,
                reg_inc_bool,
                d_seq,
                u_seq,
                cost_float,
                prev_cost_float,
                lsf_float,
            )
        )
        print_ctrl_progress(
            u_seq,
            ro_reg,
            reg_inc_bool,
            iter_int,
            lsf_float,
            converge_measure,
            avg_ff_gain,
        )
        if converge_reached is True:
            _, cost_seq, x_cost_seq, u_cost_seq = ilqr_config.calculate_total_cost(
                x_seq, u_seq
            )
            ctrl_state.set_ctrl_state(
                iter_int,
                x_seq,
                u_seq,
                K_seq,
                d_seq,
                d_v_seq,
                cost_seq,
                x_cost_seq,
                u_cost_seq,
                cost_float,
            )
        elif ilqr_config.log_ctrl_history is True and reg_inc_bool is False:
            _, cost_seq, x_cost_seq, u_cost_seq = ilqr_config.calculate_total_cost(
                x_seq, u_seq
            )
            ctrl_state.set_ctrl_state(
                iter_int,
                x_seq,
                u_seq,
                K_seq,
                d_seq,
                d_v_seq,
                cost_seq,
                x_cost_seq,
                u_cost_seq,
                cost_float,
            )
            ctrl_state.log_ctrl_state_to_history()
        iter_int += 1
    return ctrl_state


def print_ctrl_progress(
    u_seq, ro_reg, reg_inc_bool, iter_int, lsf_float, converge_measure, avg_ff_gain
):
    print("------------------")
    print("iteration number    : ", iter_int)
    print(f"converge_measure    : {converge_measure:.6f}")
    print(f"max_control_effort  : {max(u_seq).item():.3f}")
    print(f"ff_gain_avg_measure : {avg_ff_gain:.5f}")
    print("interation lsf      : ", lsf_float)
    print("ro_reg_value        : ", ro_reg)
    print("need to increase ro?: ", reg_inc_bool)


def analyze_ilqr_pass_for_convergence(
    ilqr_config,
    iter_int,
    reg_inc_bool,
    d_seq,
    u_seq,
    cost_float,
    prev_cost_float,
    lsf_float,
):
    avg_ff_gain = calculate_avg_ff_gains(d_seq, u_seq)
    converge_measure = (
        jnp.abs((prev_cost_float - cost_float) / cost_float).item()
    ) / lsf_float
    # determine convergence criteria
    converge_reached = determine_convergence(
        ilqr_config, iter_int, reg_inc_bool, converge_measure, avg_ff_gain
    )
    return converge_measure, converge_reached, avg_ff_gain


def calculate_backwards_pass(
    ilqr_config: ilqrConfigStruct,
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
    ro_reg: float,
    ro_reg_increase_bool: bool,
):
    dyn_lin_approx_seq = ilqr_config.linearize_dyn_seq(x_seq, u_seq)
    cost_quad_approx_seq = ilqr_config.quad_exp_cost_seq(x_seq, u_seq)
    if ro_reg_increase_bool is True:
        ro_reg += ilqr_config.ro_reg_change
    else:
        ro_reg = ilqr_config.ro_reg_start
    K_seq, d_seq, Del_V_vec_seq, ro_reg = sweep_back_pass(
        dyn_lin_approx_seq, cost_quad_approx_seq, ro_reg, ilqr_config.ro_reg_change
    )
    return K_seq, d_seq, Del_V_vec_seq, ro_reg


def sweep_back_pass(
    dyn_lin_approx_seqs: Tuple[jnp.ndarray, jnp.ndarray],
    cost_quad_approx_seqs: Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    ro_reg: float,
    ro_reg_inc_val_float: float,
):
    # Grab final cost-to-go approximation
    p_N, _, P_N, _, _ = tuple(seq[-1] for seq in cost_quad_approx_seqs)
    # Flip dynamics linearizations for backwards pass
    A_lin_rev, B_lin_rev = tuple(jnp.flip(seq, axis=0) for seq in dyn_lin_approx_seqs)
    # remove final step and flip cost approximations for backwards pass
    l_x_rev, l_u_rev, l_xx_rev, l_uu_rev, l_xu_rev = tuple(
        jnp.flip(seq[:-1], axis=0) for seq in cost_quad_approx_seqs
    )
    # create tuples for lax scan function
    scan_seqs = (
        (A_lin_rev, B_lin_rev),
        (l_x_rev, l_u_rev, l_xx_rev, l_uu_rev, l_xu_rev),
    )
    is_pos_def_bool = False
    while is_pos_def_bool == False:
        carry_init = (ro_reg, P_N, p_N, True)
        (_, _, _, is_pos_def_bool), (K_seq, d_seq, Del_V_vec_seq) = lax.scan(
            sweep_back_pass_scan_func, carry_init, scan_seqs
        )
        if is_pos_def_bool is False:
            ro_reg += ro_reg_inc_val_float
            # unflip the vectors to match forward time
    return (
        jnp.flip(K_seq, axis=0),
        jnp.flip(d_seq, axis=0),
        jnp.flip(Del_V_vec_seq, axis=0),
        ro_reg,
    )


def sweep_back_pass_scan_func(
    carry: Tuple[float, jnp.ndarray, jnp.ndarray, bool],
    seqs: Tuple[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ],
) -> Tuple[
    Tuple[float, jnp.ndarray, jnp.ndarray, bool],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    ro_reg, P_kp1, p_kp1, is_pos_def_bool = carry
    dyn_lin_k, cost_quad_approx_k = seqs
    if is_pos_def_bool is False:
        return (
            ro_reg,
            P_kp1,
            p_kp1,
            is_pos_def_bool,
        ), set_back_pass_scan_output_to_null_arrays(dyn_lin_k)
    q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = taylor_expand_pseudo_hamiltonian(
        dyn_lin_k, cost_quad_approx_k, P_kp1, p_kp1, ro_reg
    )
    if util.is_pos_def(q_uu_reg) is False:
        is_pos_def_bool = False
        return (
            ro_reg,
            P_kp1,
            p_kp1,
            is_pos_def_bool,
        ), set_back_pass_scan_output_to_null_arrays(dyn_lin_k)
    K_k, d_k = calculate_optimal_gains(q_uu_reg, q_ux_reg, q_u)
    P_kp1, p_kp1, D_V_k = calculate_backstep_ctg_approx(
        q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k
    )
    return (ro_reg, P_kp1, p_kp1, is_pos_def_bool), (K_k, d_k, D_V_k)


def set_back_pass_scan_output_to_null_arrays(
    dyn_lin_k: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    K_null = jnp.zeros((len(dyn_lin_k[1][1]), len(dyn_lin_k[0][1])))
    d_null = jnp.zeros((len(dyn_lin_k[1][1]), 1))
    D_V_null = jnp.zeros((2,))
    return K_null, d_null, D_V_null


def calculate_forwards_pass(
    ilqr_config: ilqrConfigStruct,
    x_seq_prev: jnp.ndarray,
    u_seq_prev: jnp.ndarray,
    K_seq: jnp.ndarray,
    d_seq: jnp.ndarray,
    d_v_seq: jnp.ndarray,
    cost_float_prev: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, bool]:
    lsf = 1.0
    iter_count = 0
    reg_inc_bool = False
    in_bounds_bool = False
    while not in_bounds_bool and (iter_count < ilqr_config.fp_max_iter):
        x_seq_fp, u_seq_fp, cost_float_fp = ilqr_config.simulate_forward_pass(
            x_seq_prev, u_seq_prev, K_seq, d_seq, lsf
        )
        cost_decrease_ratio = calculate_cost_decrease_ratio(
            cost_float_prev, cost_float_fp, d_v_seq, lsf
        )
        in_bounds_bool = ilqr_config.analyze_cost_dec(cost_decrease_ratio)
        if in_bounds_bool:
            break
        elif iter_count == ilqr_config.fp_max_iter - 1:
            x_seq_fp = x_seq_prev
            u_seq_fp = u_seq_prev
            cost_float_fp = cost_float_prev
            reg_inc_bool = True
        else:  # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
            lsf *= ilqr_config.ls_scale_alpha_param
        iter_count += 1
    return x_seq_fp, u_seq_fp, cost_float_fp, lsf, reg_inc_bool


def calculate_forwards_pass_mj(
    ilqr_config: ilqrConfigStruct,
    x_seq_prev: jnp.ndarray,
    u_seq_prev: jnp.ndarray,
    K_seq: jnp.ndarray,
    d_seq: jnp.ndarray,
    d_v_seq: jnp.ndarray,
    cost_float_prev: float,
):
    """lsf is the line search factor, scaling the size of the newton step"""
    # initialize forward pass state vec, control_vec, cost and cost seqs, line search factor, armijo parameter bool
    x_seq_fp, u_seq_fp, cost_float_fp, in_bounds_bool = initialize_forwards_pass(
        x_seq_prev.shape[1], u_seq_prev.shape[1], x_seq_prev[0], x_seq_prev.shape[0]
    )
    lsf = 1.0
    iter_count = 0
    reg_inc_bool = False
    while not in_bounds_bool and (iter_count < ilqr_config.fp_max_iter):
        for k in range(x_seq_prev.shape[0] - 1):
            u_seq_fp = u_seq_fp.at[k].set(
                calculate_u_k_new(
                    u_seq_prev[k], x_seq_prev[k], x_seq_fp[k], K_seq[k], d_seq[k], lsf
                )
            )
            x_seq_fp = x_seq_fp.at[k + 1].set(
                ilqr_config.discrete_dyn_mj_func(x_seq_fp[k], u_seq_fp[k])
            )
            cost_float_fp, _, _, _ = gen_ctrl.calculate_total_cost(
                ilqr_config.cost_func_verbose, jnp.array(x_seq_fp), jnp.array(u_seq_fp)
            )
        cost_decrease_ratio = calculate_cost_decrease_ratio(
            cost_float_prev, cost_float_fp, d_v_seq, lsf
        )
        in_bounds_bool = ilqr_config.analyze_cost_dec(cost_decrease_ratio)
        if in_bounds_bool:
            break
        elif iter_count == ilqr_config.fp_max_iter - 1:
            iter_count += 1
            x_seq_fp = x_seq_prev
            u_seq_fp = u_seq_prev
            cost_float_fp = cost_float_prev
            reg_inc_bool = True
        else:  # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
            x_seq_fp, u_seq_fp, cost_float_fp, in_bounds_bool = (
                initialize_forwards_pass(
                    x_seq_prev.shape[1],
                    u_seq_prev.shape[1],
                    x_seq_prev[0],
                    x_seq_prev.shape[0],
                )
            )
            lsf *= ilqr_config.ls_scale_alpha_param
            iter_count += 1
    return x_seq_fp, u_seq_fp, cost_float_fp, lsf, reg_inc_bool


def simulate_forward_pass(
    fp_scan_func: Callable[
        [
            Tuple[int, jnp.ndarray, jnp.ndarray, float],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
        Tuple[
            Tuple[int, jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]
        ],
    ],
    cost_func_for_calc: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
    K_seq: jnp.ndarray,
    d_seq: jnp.ndarray,
    lsf: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    seqs = (x_seq[:-1], u_seq, K_seq, d_seq)
    carry_init = (int(0), jnp.zeros((1), dtype=float), x_seq[0], lsf)
    (_, cost_float, _, _), (x_seq_new, u_seq_new) = lax.scan(
        fp_scan_func, carry_init, seqs
    )
    x_seq_new = jnp.append((x_seq[0]).reshape(1, -1), x_seq_new, axis=0)
    cost_out = cost_float + (
        cost_func_for_calc(
            x_seq_new[-1], jnp.zeros((len(u_seq[1]))), len(x_seq_new) - 1
        )
    )
    return x_seq_new, u_seq_new, cost_out.item()


def forward_pass_scan_func(
    discrete_dyn_func,
    cost_func_float,
    carry: Tuple[int, jnp.ndarray, jnp.ndarray, float],
    seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[
    Tuple[int, jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]
]:
    x_k, u_k, K_k, d_k = seqs
    k, cost_float, x_k_new, lsf = carry
    u_k_new = calculate_u_k_new(u_k, x_k, x_k_new, K_k, d_k, lsf)
    cost_float += (cost_func_float(x_k_new, u_k_new, k)).reshape(-1)
    x_kp1_new = discrete_dyn_func(x_k_new, u_k_new)
    k += 1
    return (k, cost_float, x_kp1_new, lsf), (x_kp1_new, u_k_new)


def _curry_forward_pass_scan_func(
    discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    cost_func_float: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
) -> Callable[
    [
        Tuple[int, jnp.ndarray, jnp.ndarray, float],
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ],
    Tuple[Tuple[int, jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
]:
    def fp_scan_func_curried(
        carry: Tuple[int, jnp.ndarray, jnp.ndarray, float],
        seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[
        Tuple[int, jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        carry_out, seq_out = forward_pass_scan_func(
            discrete_dyn_func, cost_func_float, carry, seqs
        )
        return carry_out, seq_out

    return fp_scan_func_curried


def initialize_forwards_pass(
    x_len: int, u_len: int, seed_state_vec: jnp.ndarray, len_seq: int
):
    control_seq_fp = jnp.zeros([(len_seq - 1), u_len])
    x_seq_fp = jnp.zeros([(len_seq), x_len])
    x_seq_fp.at[0].set(seed_state_vec)
    cost_float_fp = 0.0
    in_bounds_bool = False
    return x_seq_fp, control_seq_fp, cost_float_fp, in_bounds_bool


def taylor_expand_pseudo_hamiltonian(
    dyn_lin_approx_k: Tuple[jnp.ndarray, jnp.ndarray],
    cost_quad_approx_k: Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    P_kp1: jnp.ndarray,
    p_kp1: jnp.ndarray,
    ro_reg: float,
):
    #   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
    #   where V' is value function at the next time step
    #  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
    # q_xx = l_xx + A^T P' A
    # q_uu = l_uu + B^T P' B
    # q_ux = l_ux + B^T P' A
    # q_x  = l_x  + A^T p'
    # q_u  = l_u  + B^T p'
    A_lin_k, B_lin_k = dyn_lin_approx_k
    l_x, l_u, l_xx, l_uu, l_ux = cost_quad_approx_k
    P_kp1_reg = util.reqularize_mat(P_kp1, ro_reg)
    q_x = l_x + A_lin_k.T @ p_kp1
    q_u = l_u + B_lin_k.T @ p_kp1
    q_xx = l_xx + A_lin_k.T @ P_kp1 @ A_lin_k
    q_uu = l_uu + B_lin_k.T @ P_kp1 @ B_lin_k
    q_ux = l_ux + B_lin_k.T @ P_kp1 @ A_lin_k
    q_uu_reg = l_uu + B_lin_k.T @ P_kp1_reg @ B_lin_k
    q_ux_reg = l_ux + B_lin_k.T @ P_kp1_reg @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg


def calculate_backstep_ctg_approx(
    q_x: jnp.ndarray,
    q_u: jnp.ndarray,
    q_xx: jnp.ndarray,
    q_uu: jnp.ndarray,
    q_ux: jnp.ndarray,
    K_k: jnp.ndarray,
    d_k: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # P_k is the kth hessian approximation of the cost-to-go
    # p_k is the kth jacobian approximation of the cost-to-go
    # Del_V_k is the kth expected change in the value function
    assert q_x.shape == (len(q_x), 1), "q_x must be column vector (n,1)"
    assert q_u.shape == (len(q_u), 1), "q_u must be column vector (n,1)"
    assert K_k.shape == (
        len(q_u),
        len(q_x),
    ), "K_k incorrect shape, must be (len(q_u), len(q_x))"
    assert d_k.shape == (len(q_u), 1), "d_k incorrect shape, must be (len(q_u), 1)"
    assert q_xx.shape == (
        len(q_x),
        len(q_x),
    ), "q_xx incorrect shape, must be (len(q_x), len(q_x))"
    assert q_uu.shape == (
        len(q_u),
        len(q_u),
    ), "q_uu incorrect shape, must be (len(q_u), len(q_u))"
    assert q_ux.shape == (
        len(q_u),
        len(q_x),
    ), "q_ux incorrect shape, must be (len(q_u), len(q_x))"
    q_xu = q_ux.T
    P_k = (q_xx) + (K_k.T @ q_uu @ K_k) + (K_k.T @ q_ux) + (q_xu @ K_k)
    p_k = (q_x) + (K_k.T @ q_uu @ d_k) + (K_k.T @ q_u) + (q_xu @ d_k)
    # TODO verify Del_V_vec_k calculation signs, absolute vs actual, actual gives different signs...
    Del_V_vec_k = jnp.array([(d_k.T @ q_u), ((0.5) * (d_k.T @ q_uu @ d_k))]).reshape(-1)
    return P_k, p_k, Del_V_vec_k


def calculate_optimal_gains(
    q_uu_reg: jnp.ndarray, q_ux: jnp.ndarray, q_u: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # K_k is a linear feedback term related to del_x from nominal trajectory
    # d_k is a feedforward term related to trajectory correction
    inverted_q_uu = jnp.linalg.inv(q_uu_reg)
    K_k = -inverted_q_uu @ q_ux
    d_k = -inverted_q_uu @ q_u
    return K_k, d_k


def calculate_u_k_new(
    u_current_k: jnp.ndarray,
    x_current_k: jnp.ndarray,
    x_new_k: jnp.ndarray,
    K_fb_k: jnp.ndarray,
    d_ff_k: jnp.ndarray,
    line_search_factor: float,
) -> jnp.ndarray:
    # check valid dimensions
    assert K_fb_k.shape == (
        len(u_current_k),
        len(x_current_k),
    ), "K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))"
    assert d_ff_k.shape == (
        len(u_current_k),
        1,
    ), "d_k incorrect shape, must be (len(u_nom_k), )"
    assert len(x_new_k) == len(
        x_current_k
    ), "x_nom_k and x_new_k must be the same length"
    assert (x_current_k.shape) == (
        len(x_current_k),
    ), "x_nom_k must be column vector (n,)"
    assert (x_new_k.shape) == (len(x_new_k),), "x_new_k must be column vector (n,)"
    assert (u_current_k.shape) == (
        len(u_current_k),
    ), "u_nom_k must be column vector (m,)"
    # return (u_nom_k.reshape(-1,1) + K_k @ (x_new_k.reshape(-1,1) - x_nom_k.reshape(-1,1)) + line_search_factor * d_k).reshape(-1)
    return gen_ctrl.calculate_ff_fb_u(
        K_fb_k, u_current_k, x_current_k, x_new_k
    ) + line_search_factor * d_ff_k.reshape(-1)


def calculate_cost_decrease_ratio(
    prev_cost_float: float,
    new_cost_float: float,
    Del_V_vec_seq: jnp.ndarray,
    lsf: float,
) -> float:
    """
    Calculates the ratio between the expected cost decrease from approximated dynamics and cost functions, and the calculated
    cost from the forward pass shooting method. The cost decrease is expected to be positive, as the Del_V_sum should be a negative
    expected value, and the new cost should be smaller than the previous cost.
    """
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, lsf)
    cost_decrease_ratio = (1 / Del_V_sum) * (new_cost_float - prev_cost_float)
    return cost_decrease_ratio


def calculate_expected_cost_decrease(Del_V_vec_seq: jnp.ndarray, lsf: float) -> float:
    Del_V_sum = jnp.sum((lsf * Del_V_vec_seq[:, 0]) + (lsf**2 * Del_V_vec_seq[:, 1]))
    return Del_V_sum.item()


def calculate_expected_cost_decrease_step(Del_V_vec: jnp.ndarray, lsf: float) -> float:
    Del_V_sum = (lsf * Del_V_vec[0]) + (lsf**2 * Del_V_vec[1])
    return Del_V_sum.item()


def analyze_cost_decrease(
    cost_decrease_ratio: float, cost_ratio_bounds: Tuple[float, float]
) -> bool:
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (
        cost_decrease_ratio < cost_ratio_bounds[1]
    ):
        return True
    else:
        return False


def calculate_avg_ff_gains(d_seq: jnp.ndarray, u_seq: jnp.ndarray) -> float:
    norm_factor: float = 1 / (len(u_seq) - 1)
    norm_gain_sum: float = jnp.sum(
        jnp.abs(d_seq.reshape(-1).max()) / (jnp.linalg.norm(u_seq, axis=1) + 1)
    ).item()
    return norm_factor * norm_gain_sum


def calculate_u_star(
    k_fb_k: jnp.ndarray, u_des_k: jnp.ndarray, x_des_k: jnp.ndarray, x_k: jnp.ndarray
) -> jnp.ndarray:
    return (
        u_des_k.reshape(-1, 1) + k_fb_k @ (x_k.reshape(-1, 1) - x_des_k.reshape(-1, 1))
    ).reshape(-1)


def simulate_ilqr_output(
    sim_dyn_func_step: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ctrl_obj: ilqrControllerState,
    x_sim_init: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_sim_seq = jnp.zeros([ctrl_obj.len_seq, ctrl_obj.x_len])
    u_sim_seq = jnp.zeros([ctrl_obj.len_seq - 1, ctrl_obj.u_len])
    x_sim_seq = x_sim_seq.at[0].set(x_sim_init)
    for k in range(ctrl_obj.len_seq - 1):
        u_sim_seq = u_sim_seq.at[k].set(
            gen_ctrl.calculate_ff_fb_u(
                ctrl_obj.K_seq[k], ctrl_obj.u_seq[k], ctrl_obj.x_seq[k], x_sim_seq[k]
            )
        )
        x_sim_seq = x_sim_seq.at[k + 1].set(
            sim_dyn_func_step(x_sim_seq[k], u_sim_seq[k])
        )
    return x_sim_seq, u_sim_seq


def determine_convergence(
    ilqr_config: ilqrConfigStruct,
    iter_int: int,
    reg_inc_bool: bool,
    converge_measure: float,
    avg_ff_gain: float,
) -> bool:
    converge_reached = False
    if (converge_measure < ilqr_config.converge_crit) and (reg_inc_bool is False):
        converge_reached = True
        print("controller converged")
    elif avg_ff_gain < ilqr_config.ff_gain_tol:
        converge_reached = True
        print("controller ff gains below convergence tolerance")
    elif iter_int == ilqr_config.max_iter:
        print("controller reached iteration limit")
    return converge_reached
