
import numpy as np
from . import gen_ctrl_funcs as gen_ctrl

"""
Time-varying LQR functions

TVLQR is an optimization algorithm that accepts a time-varying linear dynamic system time-dependent linear quadratic controller

"""

class ctrlConfig():
    def __init__(self, time_step, len_seq, x_len, u_len, Q, R, Qf) -> None:
        self.time_step = time_step
        self.len_seq   = len_seq
        self.x_len     = x_len
        self.u_len     = u_len
        self.Q         = Q
        self.R         = R
        self.Qf        = Qf
        

class ctrlOutput():
    def __init__(self, ctrl_config:ctrlConfig):
        self.x_nom_seq = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, 1])
        self.u_nom_seq = np.zeros([ctrl_config.len_seq-1, ctrl_config.u_len, 1])
        self.k_fb_seq   = np.zeros([ctrl_config.len_seq-1, ctrl_config.u_len, ctrl_config.x_len])
        self.s_xx_seq  = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, ctrl_config.x_len])


def calc_tvlqr_controller(ctrl_config:ctrlConfig, A_seq, B_seq, x_nom_seq, u_nom_seq)-> ctrlOutput:
    ctrl_out              = ctrlOutput(ctrl_config)
    ctrl_out.x_nom_seq    = x_nom_seq
    ctrl_out.u_nom_seq    = u_nom_seq
    ctrl_out.s_xx_seq[-1] = ctrl_config.Qf
    N = ctrl_config.len_seq
    for idx in range(N-1):
        k = N -(idx+2)
        BRinvBT                = (B_seq[k]) @ np.linalg.inv(ctrl_config.R) @ (B_seq[k]).T
        s_xx_dt                = gen_ctrl.calculate_s_xx_dot(ctrl_config.Q, A_seq[k], BRinvBT, ctrl_out.s_xx_seq[k+1])
        s_xx_km1               = ctrl_out.s_xx_seq[k+1] - s_xx_dt * ctrl_config.time_step
        ctrl_out.s_xx_seq[k]   = s_xx_km1
        ctrl_out.k_fb_seq[k]   = calculate_k_fb(ctrl_config.R, B_seq[k], ctrl_out.s_xx_seq[k])
    return ctrl_out    

def calculate_k_fb(R_k, B_k, s_xx_k):
    k_fb_k = np.linalg.inv(R_k) @ B_k.T @ s_xx_k
    return k_fb_k

def calculate_u_star(k_fb_k, u_des_k, x_des_k, x_k):
    u_star = u_des_k - k_fb_k @ (x_k - x_des_k)
    return u_star

def simulate_lti_tvlqr(ctrl_config:ctrlConfig, ctrl_out, x_sim_init, ss_sim_cont, sim_method = 'solve_ivp_zoh'):
    A_sim_cont = ss_sim_cont.a
    B_sim_cont = ss_sim_cont.b
    A_sim_cont_seq = np.zeros([ctrl_config.len_seq-1, A_sim_cont.shape[0], A_sim_cont.shape[1]])
    B_sim_cont_seq = np.zeros([ctrl_config.len_seq-1, B_sim_cont.shape[0], B_sim_cont.shape[1]])
    A_sim_cont_seq[:] = A_sim_cont
    B_sim_cont_seq[:] = B_sim_cont
    x_sim_seq  = np.zeros([ctrl_config.len_seq, ctrl_config.x_len, 1])
    u_sim_seq  = np.zeros([ctrl_config.len_seq-1, ctrl_config.u_len, 1])
    x_sim_seq[0] = x_sim_init    
    for k in range(ctrl_config.len_seq-1):
        u_sim_seq[k]     = calculate_u_star(ctrl_out.k_fb_seq[k], ctrl_out.u_nom_seq[k], ctrl_out.x_nom_seq[k], x_sim_seq[k])
        x_dyn_func_k  = lambda time, x_nom, u_nom:A_sim_cont @ x_nom + B_sim_cont @ u_nom
        x_sim_seq[k+1]   = gen_ctrl.simulate_forward_dynamics_step(x_dyn_func_k, x_sim_seq[k], u_sim_seq[k],ctrl_config.time_step, sim_method=sim_method)

    return x_sim_seq,u_sim_seq

def plot_2d_state_and_control(x_nom_seq, ctrl_out, x_sim_seq, u_sim_seq, time_seq, ax1, ax2):
    ax1.plot(x_nom_seq[0,0], x_nom_seq[0,1], color = 'blue', marker = 'o', label = "nominal start")
    ax1.plot(x_sim_seq[0,0], x_sim_seq[0,1], color = 'red', marker = 'o', label = "sim start")
    ax1.plot(x_nom_seq[:,0], x_nom_seq[:,1], color = 'blue', label = 'desired trajectory')
    ax1.plot(x_sim_seq[:,0], x_sim_seq[:,1], color = 'red' , label = 'output trajectory', linestyle = '--')
    ax1.plot(x_nom_seq[0,0], x_nom_seq[0,1], )
    ax1.legend()
    ax2.plot(time_seq, ctrl_out.u_nom_seq[:,0],color = 'blue', label = "nominal control input" )    
    ax2.plot(time_seq, u_sim_seq[:,0],color = 'red', label = "optimal control input", linestyle = '--')
    ax2.legend()