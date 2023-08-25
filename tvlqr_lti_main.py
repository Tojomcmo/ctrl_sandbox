import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import src.gen_ctrl_funcs as gen_ctrl
import src.tvlqr_funcs as tvlqr
import src.dyn_functions as dyn
import src.analyze_ilqr_output_funcs as analyze


if __name__ == '__main__':

    """
    time-varying trajectory tracking LQR

    Accepts a time-varying linear system (can be a linearized set of locally valid
    linear systems along a viable trajectory through a non-linear system)
    ctrl_output = gen_tvlqr_controller(ctrl_config, A_seq, B_seq, x_nom, u_nom)    
    u_k         = run_tvlqr_controller(ctrl_output, x_k, t_k)

    """ 

    # define general system paramters
    time_step = 0.01
    len_seq   = 200

    x_init    = np.array([[1.],[1.]])
    u_nom_seq = np.ones([len_seq-1, 1, 1]) * 5

    Q  = np.array([[1.,0.],[0.,1.]]) * 10
    R  = np.array([[0.1]])
    Qf = np.array([[1.,0.],[0.,1.]]) * 10

    ctrl_config = tvlqr.ctrlConfig(time_step, len_seq, len(x_init), len(u_nom_seq[0]), Q, R, Qf)

    # create dynamic system for control planning
    lti_ss_params = {'g':10.0, 'b':1.0, 'l':1.0}
    ss_cont       = dyn.lti_pend_ss_cont(lti_ss_params)
    A_mat_cont    = ss_cont.a
    B_mat_cont    = ss_cont.b

    # create x_nom_seq and u_nom_seq with dynamic system
    x_nom_seq    = np.zeros([len_seq, ctrl_config.x_len, 1])
    x_nom_seq[0] = x_init
    for k in range(len_seq-1):
        x_nom_dt_k     = A_mat_cont @ x_nom_seq[k] + B_mat_cont @ u_nom_seq[k]
        x_nom_seq[k+1] = x_nom_seq[k] + x_nom_dt_k * ctrl_config.time_step

    # create A and B sequences from lti dynamic system
    A_cont_seq    = np.zeros([len_seq-1, A_mat_cont.shape[0], A_mat_cont.shape[1]])
    B_cont_seq    = np.zeros([len_seq-1, B_mat_cont.shape[0], B_mat_cont.shape[1]])
    A_cont_seq[:] = A_mat_cont
    B_cont_seq[:] = B_mat_cont

    # calculate optimal TV controller gains
    ctrl_out = tvlqr.calc_tvlqr_controller(ctrl_config, A_cont_seq, B_cont_seq, x_nom_seq, u_nom_seq)

    # run system forward
    lti_ss_sim_params    = {'g':10.0, 'b':1.0, 'l':1.0}
    x_sim_init           = np.array([[1.1],[1.1]])
    ss_sim_cont          = dyn.lti_pend_ss_cont(lti_ss_sim_params)
    x_sim_seq, u_sim_seq = tvlqr.simulate_lti_tvlqr(ctrl_config, ctrl_out, x_sim_init, ss_sim_cont)

    # plot outputs 
    time_seq = np.arange(ctrl_config.len_seq-1) * ctrl_config.time_step
    gs  = gridspec.GridSpec(2, 1)
    fig = plt.figure(figsize=(10,12))
    ax1 = fig.add_subplot(gs[0,0]) # row 0, col 0
    ax2 = fig.add_subplot(gs[1,0]) # row 0, col 0    
    tvlqr.plot_2d_state_and_control(x_nom_seq, ctrl_out, x_sim_seq, u_sim_seq, time_seq, ax1, ax2)
    plt.show()