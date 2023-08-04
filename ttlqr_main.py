import numpy as np

import src.gen_ctrl_funcs as gen_ctrl
import src.ttlqr_funcs as ttlqr
import src.dyn_functions as dyn


if __name__ == '__main__':
    # Define system paramaters
    lti_ss_params = {'g':1.0, 'b':1.0, 'l':1.0}
    config_dict = {'lti_ss'        : dyn.lti_pend_ss_cont,
                    'lti_ss_params' : lti_ss_params,
                    'time_step'     : 0.1,
                    'Q'             : np.array([[1.,0.],[0.,1.]]),
                    'R'             : np.array([[1.]]),
                    'Qf'            : np.array([[2.,0.],[0.,2.]]),
                    'c2d_method'    : 'euler'}
    len_seq        = 20
    u_traj_gen_seq = np.ones([len_seq-1, 1, 1])
    x_init_vec     = np.array([[1.],[1.]])
    # Define trajectory
    lin_dyn_func = gen_ctrl.ss_2_dyn_func(dyn.lti_pend_ss_cont(config_dict['lti_ss_params']))
    x_des_seq = gen_ctrl.simulate_forward_dynamics(lin_dyn_func, x_init_vec, u_traj_gen_seq, config_dict['time_step'])

    print(x_des_seq)
    # Initialize controller with system and parameters
    ctrl_config = ttlqr.ttlqrControllerConfig(config_dict, x_des_seq, u_traj_gen_seq)
    ctrl_state  = ttlqr.ttlqrControllerState(ctrl_config)

    # Run controller
    ctrl_state.s_xx_seq, ctrl_state.s_x_seq, ctrl_state.s_0_seq, ctrl_state.g_ff_seq, ctrl_state.g_fb_seq = ttlqr.calculate_ttlqr_seq(ctrl_config) # type: ignore

    # simulate system response

    # simulate system response with disturbance

    # plot / analyze output