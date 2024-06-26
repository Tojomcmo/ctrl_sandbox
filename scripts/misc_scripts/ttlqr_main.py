import numpy as np
from matplotlib import pyplot as plt

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.ttlqr_funcs as ttlqr
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.statespace_funcs as ss_funcs


if __name__ == "__main__":
    # Define system paramaters

    gen_lin_dyn_sys = dyn.single_pm_pend_dyn(g=10, b=0.7, l=1)
    lti_ss_params = {"g": 10.0, "b": 0.7, "l": 1.0}
    time_step = 0.01

    config_dict = {
        "lti_ss": gen_lin_dyn_sys,
        "lti_ss_params": lti_ss_params,
        "time_step": time_step,
        "Q": np.array([[10.0, 0.0], [0.0, 10.0]]) * 1,
        "R": np.array([[1.0]]),
        "Qf": np.array([[1.0, 0.0], [0.0, 1.0]]) * 1,
        "c2d_method": "zohCombined",
    }

    len_seq = 300
    u_traj_gen_seq = np.ones([len_seq - 1, 1, 1]) * 20.0
    x_init_vec = np.array([[0.1], [0.1]])
    # Define trajectory

    traj_gen_ss_cont = gen_lin_dyn_sys.cont_lti_pend_ss_down()
    traj_gen_ss_discrete = ss_funcs.discretize_continuous_state_space(
        traj_gen_ss_cont, time_step, c2d_method="zohCombined"
    )
    x_des_seq = np.zeros([len_seq, len(x_init_vec), 1])
    x_des_seq[0] = x_init_vec
    for k in range(len_seq - 1):
        x_des_seq[k + 1] = (
            traj_gen_ss_discrete.a @ x_des_seq[k]
            + traj_gen_ss_discrete.b @ u_traj_gen_seq[0]
        )

    # lin_dyn_func = gen_ctrl.ss_2_dyn_func(dyn.lti_pend_ss_cont(config_dict['lti_ss_params']))
    # x_des_seq = gen_ctrl.simulate_forward_dynamics_seq(lin_dyn_func, x_init_vec, u_traj_gen_seq, config_dict['time_step'], sim_method = 'solve_ivp_zoh')

    print(x_des_seq)
    # Initialize controller with system and parameters
    ctrl_config = ttlqr.ttlqrControllerConfig(config_dict, x_des_seq, u_traj_gen_seq)
    ctrl_state = ttlqr.ttlqrControllerState(ctrl_config)

    # Run controller
    ctrl_state.s_xx_seq, ctrl_state.s_x_seq, ctrl_state.s_0_seq, ctrl_state.g_ff_seq, ctrl_state.g_fb_seq = ttlqr.calculate_ttlqr_seq(ctrl_config)  # type: ignore

    # initialize simulation parameters

    lti_ss_sim_params = {"g": 10.0, "b": 0.5, "l": 1.0}
    sim_ss_cont = gen_lin_dyn_sys.cont_lti_pend_ss_down()
    sim_ss_discrete = ss_funcs.discretize_continuous_state_space(
        sim_ss_cont, time_step, c2d_method="zohCombined"
    )
    x_sim_seq = np.zeros([len_seq, len(x_init_vec), 1])
    u_sim_seq = np.zeros([len_seq - 1, 1, 1])
    x_sim_seq[0] = x_init_vec
    x_sim_seq[0] = np.array([[-0.25], [-0.25]])
    disturbance = np.array([[0], [0.01]])
    for k in range(len_seq - 1):
        u_sim_seq[k] = ttlqr.calculate_u_opt_k(ctrl_state.g_ff_seq[k], ctrl_state.g_fb_seq[k], x_sim_seq[k], u_traj_gen_seq[k], x_des_seq[k])  # type: ignore
        x_sim_seq[k + 1] = (
            sim_ss_discrete.a @ x_sim_seq[k] + sim_ss_discrete.b @ u_sim_seq[k]
        )

    # sim_config_dict = config_dict
    # sim_config_dict['lti_ss_params'] = {'g':1.0, 'b':1.0, 'l':1.0}
    # sim_config_dict['c2d_method']    = 'zoh'
    # sim_config = ttlqr.ttlqrControllerConfig(sim_config_dict, x_des_seq, u_traj_gen_seq)
    # sim_dyn_func = lambda x_k, u_k: gen_ctrl.simulate_forward_dynamics_step(sim_config.lin_dyn_sys, x_k, u_k, sim_config_dict['time_step'],
    #                                                                         sim_method = 'solve_ivp_zoh')

    # simulate system response
    # x_output_seq, u_output_seq = ttlqr.simulate_ttlqr_controller(sim_dyn_func, ctrl_state, ctrl_config)

    # simulate system response with disturbance

    # plot / analyze output
    plt.figure()
    plt.plot(x_des_seq[:, 0], x_des_seq[:, 1], color="blue", label="desired_trajectory")
    plt.plot(
        x_sim_seq[:, 0],
        x_sim_seq[:, 1],
        color="red",
        label="output_trajectory",
        linestyle="--",
    )
    plt.legend()
    plt.show()
