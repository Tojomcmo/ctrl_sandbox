import numpy as np
from numpy import typing as npt

import gen_ctrl_funcs as gen_ctrl
import ilqr_utils as util


class ttlqrControllerConfig:
    def __init__(self, config_dict, x_des_seq, u_des_seq):
        # verify input data
        self.verify_config_dict(config_dict)
        # direct parameter fill
        self.time_step = config_dict["time_step"]
        self.x_des_seq = x_des_seq
        self.u_des_seq = u_des_seq
        self.Q = config_dict["Q"]
        self.R = config_dict["R"]
        self.Qf = config_dict["Qf"]
        # dependent parameter fill
        self.len_seq = len(x_des_seq)
        self.x_len = len(x_des_seq[0])
        self.u_len = len(u_des_seq[0])
        self.time_seq = np.arange(self.len_seq) * self.time_step
        # create curried functions
        self.c2d_func = self.curry_c2d_func(config_dict["c2d_method"])
        self.ss_discrete, self.lin_dyn_sys = self.curry_lin_dyn_func(
            config_dict["lti_ss_params"], config_dict["lti_ss"]
        )
        self.ctg_params_func = self.curry_ctg_params_func()
        # time saving values
        self.RinvBT = np.linalg.inv(self.R) @ (self.ss_discrete.b).T
        self.BRinvBT = self.ss_discrete.b @ self.RinvBT

    def verify_config_dict(self, config_dict):
        expected_keys = [
            "Q",
            "R",
            "Qf",
            "lti_ss",
            "lti_ss_params",
            "time_step",
            "c2d_method",
        ]
        # if not (config_dict['Q']).any() or not (config_dict['R']).any() or not (config_dict['Qf']).any():
        #     raise ValueError("config_dict requires keys 'Q', 'R', and 'Qf' with np.arrays of dimension (n,n), (m,m), and (n,n), respectively")
        if not config_dict["lti_ss"] or not config_dict["lti_ss_params"]:
            raise ValueError(
                "config_dict requires keys 'lti_ss' and 'lti_ss_params' with dynamics func and paired parameter set"
            )
        if not config_dict["time_step"]:
            raise ValueError(
                "config_dict requires key 'time_step' with system descritization timestep float"
            )
        if not config_dict["c2d_method"]:
            raise ValueError(
                "config_dict requires key 'c2d_method' of val 'euler', 'zoh', or 'zohCombineed' for descritization method"
            )

    def verify_input_des_seqs(self, u_des_seq, x_des_seq):
        return

    def curry_c2d_func(self, c2d_method: str):
        time_step = self.time_step
        c2d_func = lambda cont_state_space: gen_ctrl.discretize_continuous_state_space(
            cont_state_space, time_step, c2d_method
        )
        return c2d_func

    def curry_lin_dyn_func(self, lin_dyn_sys_params: dict, gen_lin_dyn_sys):
        # create statespace from provided general statespace
        ss_cont = gen_lin_dyn_sys(lin_dyn_sys_params)
        ss_discrete = self.c2d_func(ss_cont)
        curried_lin_dyn_func = lambda t, x, u: (
            ss_discrete.a @ (x.reshape(-1)) + ss_discrete.b @ (u.reshape(-1))
        )
        return ss_discrete, curried_lin_dyn_func

    def curry_ctg_params_func(self):
        ctg_params_func = lambda xdt, udt, sxxt, sxt, s0t: calculate_ctg_params_dot(
            self.Q,
            self.R,
            xdt,
            udt,
            self.BRinvBT,
            sxxt,
            sxt,
            s0t,
            self.ss_discrete.a,
            self.ss_discrete.b,
        )
        return ctg_params_func


class ttlqrControllerState:
    def __init__(self, ctrl_config: ttlqrControllerConfig):
        s_xx_seq = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, ctrl_config.x_len])
        s_x_seq = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, 1])
        s_0_seq = np.zeros([ctrl_config.len_seq, 1])
        g_ff_seq = np.zeros(
            [ctrl_config.len_seq - 1, ctrl_config.u_len, ctrl_config.x_len]
        )
        g_fb_seq = np.zeros(
            [ctrl_config.len_seq - 1, ctrl_config.u_len, ctrl_config.x_len]
        )


def calculate_ttlqr_seq(ctrl_config: ttlqrControllerConfig):
    """
    [IN]  - ctrl_config - ttlqr controller configuration class
    [OUT] - s_xx_seq    - np array (len_seq, x_len, x_len) of quadratic state weights for cost-to-go steps
    [OUT] - s_x_seq     - np array (len_seq, x_len, 1) of linear state weights cost-to-go steps
    [OUT] - s_0_seq     - np_array (len_seq, 1) of cost dc offset steps
    [OUT] - g_ff_seq    - np_array (len_seq-1, u_len, u_len) of feedforward gains at each step
    [OUT] - g_fb_seq    - np_array (len_seq-1, u_len, x_len) of feedback gains at each step
    """
    s_xx_seq, s_x_seq, s_0_seq, g_fb_seq, g_ff_seq = initialize_ttlqr_seqs(
        ctrl_config.len_seq, ctrl_config.x_len, ctrl_config.u_len
    )
    s_xx_seq[-1], s_x_seq[-1], s_0_seq[-1] = calculate_final_ctg_params(
        ctrl_config.Qf, ctrl_config.x_des_seq[-1]
    )
    for idx in range(ctrl_config.len_seq - 1):
        k = -(idx + 2)
        s_xx_dot, s_x_dot, s_0_dot = ctrl_config.ctg_params_func(
            ctrl_config.x_des_seq[k],
            ctrl_config.u_des_seq[k + 1],
            s_xx_seq[k + 1],
            s_x_seq[k + 1],
            s_0_seq[k + 1],
        )
        s_xx_seq[k], s_x_seq[k], s_0_seq[k] = back_integrate_euler_ctg_params(
            s_xx_dot,
            s_x_dot,
            s_0_dot,
            s_xx_seq[k + 1],
            s_x_seq[k + 1],
            s_0_seq[k + 1],
            ctrl_config.time_step,
        )
        g_ff_seq[k + 1], g_fb_seq[k + 1] = calculate_u_ff_fb_gains_at_k(
            s_xx_seq[k], s_x_seq[k], ctrl_config.RinvBT
        )

    # if not verify_backstep_params(s_xx_seq, s_x_seq, s_0_seq):
    #     raise ValueError('invalid sequence of backstepped parameters, J(x,t)')
    return s_xx_seq, s_x_seq, s_0_seq, g_ff_seq, g_fb_seq


def calculate_final_ctg_params(Qf: npt.ArrayLike, x_des_N: npt.ArrayLike):
    # Calculate final cost to go
    s_xx_N = Qf
    s_x_N = -2 * (Qf @ x_des_N)  # type: ignore
    s_0_N = x_des_N.T @ Qf @ x_des_N  # type: ignore
    return s_xx_N, s_x_N, s_0_N


def calculate_ctg_params_disc(
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    x_des_k: npt.ArrayLike,
    s_xx_kp1: npt.ArrayLike,
    s_x_kp1: npt.ArrayLike,
    s_0_kp1: npt.ArrayLike,
    A: npt.ArrayLike,
    B: npt.ArrayLike,
):

    # discrete version of backstepping calculation - (CMU-24-677 Module 2-2 notes slide 43)
    U_kp1 = np.linalg.inv(R + B.T @ s_xx_kp1 @ B)  # type: ignore
    s_xx_k = Q + (A.T @ s_xx_kp1 @ A) - (A.T @ s_xx_kp1 @ B @ U_kp1 @ B.T @ s_xx_kp1 @ A)  # type: ignore
    s_x_k = (A.T @ s_x_kp1) - (A.T @ s_xx_kp1 @ B @ U_kp1 @ B.T @ s_x_kp1) - (Q @ x_des_k)  # type: ignore
    s_0_k = s_0_kp1 - ((0.5) * s_x_kp1.T @ B @ U_kp1 @ B.T @ s_x_kp1) + ((0.5) * x_des_k.T @ Q @ x_des_k)  # type: ignore
    g_fb_k = -U_kp1 @ B.T  # type: ignore
    g_ff_k = U_kp1 @ B.T @ s_xx_kp1 @ A  # type: ignore
    return s_xx_k, s_x_k, s_0_k, g_ff_k, g_fb_k


def calculate_ctg_params_dot(
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    x_des_t: npt.ArrayLike,
    u_des_t: npt.ArrayLike,
    BRinvBT: npt.ArrayLike,
    s_xx_t: npt.ArrayLike,
    s_x_t: npt.ArrayLike,
    s_0_t: npt.ArrayLike,
    A: npt.ArrayLike,
    B: npt.ArrayLike,
):

    # continuous version optimal tracking (Tedrake Underact chapter 8.2.4)
    s_xx_dot = Q - (s_xx_t @ BRinvBT @ s_xx_t) + (s_xx_t @ A) + (A.T @ s_xx_t)  # type: ignore
    s_x_dot = -(2 * Q @ x_des_t) + ((A.T - s_xx_t @ BRinvBT) @ s_x_t) + (2 * s_xx_t @ B @ u_des_t)  # type: ignore
    s_0_dot = -((x_des_t.T @ Q @ x_des_t) - (s_x_t.T @ BRinvBT @ s_x_t) + 2 * (s_x_t.T @ B @ u_des_t))  # type: ignore
    return s_xx_dot, s_x_dot, s_0_dot


def back_integrate_euler_ctg_params(
    s_xx_dot, s_x_dot, s_0_dot, s_xx_t, s_x_t, s_0_t, time_step
):
    s_xx_tm1 = s_xx_t + s_xx_dot * time_step
    s_x_tm1 = s_x_t + s_x_dot * time_step
    s_0_tm1 = s_0_t + s_0_dot * time_step
    return s_xx_tm1, s_x_tm1, s_0_tm1


def calculate_u_ff_fb_gains_at_k(s_xx_k, s_x_k, RinvBT):
    g_ff_k = RinvBT @ ((0.5) * s_x_k)
    g_fb_k = RinvBT @ s_xx_k
    return g_ff_k, g_fb_k


def verify_backstep_params(s_xx_seq, s_x_seq, s_0_seq):
    is_valid_seqs_bool = True
    for idx in range(len(s_xx_seq)):
        if not util.is_pos_def(s_xx_seq[idx]) or (
            s_0_seq[idx]
            < (s_x_seq[idx]).T @ np.linalg.inv(s_xx_seq[idx]) @ s_x_seq[idx]
        ):
            is_valid_seqs_bool = False
            break
    return is_valid_seqs_bool


def initialize_ttlqr_seqs(len_seq, x_len, u_len):
    s_xx_seq = np.zeros([len_seq, x_len, x_len])
    s_x_seq = np.zeros([len_seq, x_len, 1])
    s_0_seq = np.zeros([len_seq, 1])
    g_fb_seq = np.zeros([len_seq - 1, u_len, x_len])
    g_ff_seq = np.zeros([len_seq - 1, u_len, u_len])
    return s_xx_seq, s_x_seq, s_0_seq, g_fb_seq, g_ff_seq


def calculate_u_opt_k(g_ff_k, g_fb_k, x_k, u_des_k, x_des_k):
    """
    g_ff_k - param in - np array(ulen, ulen)
    """
    ff_corr = g_ff_k * 0
    fb_corr = g_fb_k @ (x_k - x_des_k)
    u_opt_k = u_des_k - (fb_corr + ff_corr)
    return u_opt_k


def simulate_ttlqr_controller(
    sim_dyn_func, ctrl_state: ttlqrControllerState, ctrl_config: ttlqrControllerConfig
):
    x_output_seq = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, 1])
    u_output_seq = np.zeros([ctrl_config.len_seq - 1, ctrl_config.u_len, 1])
    x_output_seq[0] = ctrl_config.x_des_seq[0]
    for k_step in range(ctrl_config.len_seq - 1):
        u_opt_k = calculate_u_opt_k(ctrl_state.g_ff_seq[k_step], ctrl_state.g_ff_seq[k_step], ctrl_state.s_x_seq[k_step], x_output_seq[k_step])  # type: ignore
        x_kp1 = sim_dyn_func(x_output_seq[k_step], u_opt_k)
        x_output_seq[k_step + 1] = x_kp1
        u_output_seq[k_step] = u_opt_k
    return x_output_seq, u_output_seq


# Tedrake notes calculate u ---
# def calculate_u(RinvBT, u_des_k, S_xx_k, s_x_k, x_k):
#     u_k = u_des_k - RinvBT @ (S_xx_k @ x_k + s_x_k)
#     return u_k
