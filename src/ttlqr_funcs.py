import numpy as np
from numpy import typing as npt

import src.gen_ctrl_funcs as gen_ctrl
import src.ilqr_utils as util

class ttlqrControllerConfig():
    def __init__(self, config_dict, u_des_seq, x_des_seq):
        # verify input data

        # direct parameter fill
        self.time_step   = config_dict['time_step']
        self.x_des_seq   = x_des_seq
        self.u_des_seq   = u_des_seq
        self.Q           = config_dict['Q' ]
        self.R           = config_dict['R' ]
        self.Qf          = config_dict['Qf']
        self.c2d_func    = self.curry_c2d_func(config_dict)        
        self.ss_discrete, self.lin_dyn_sys = self.curry_lin_dyn_func(config_dict['lin_dyn_sys_params'], config_dict['lin_dyn_sys'])        
        # dependent parameter fill
        self.len_seq     = len(x_des_seq)
        self.x_len       = len(x_des_seq[0])
        self.u_len       = len(u_des_seq[0])
        self.time_seq    = np.arange(self.len_seq) * self.time_step
        # time saving values
        self.RinvBT     = np.linalg.inv(self.R) @ (self.ss_discrete.b).T
        self.BRinvBT    = self.ss_discrete.b    @  self.RinvBT


    def verify_config_dict(self, config_dict):
        expected_keys = ['Q','R','Qf',
                         'lin_dyn_sys',
                         'lin_dyn_sys_params',
                         'time_step',
                         'c2d_method']
        if not config_dict['Q'] or not config_dict['R'] or not config_dict['R']:
            raise ValueError("config_dict requires keys 'Q', 'R', and 'Qf' with np.arrays of dimension (n,n), (m,m), and (n,n), respectively")
        if not config_dict['lin_dyn_sys'] or not config_dict['lin_dyn_sys_params']:
            raise ValueError("config_dict requires keys 'lin_dyn_sys' and 'lin_dyn_sys_params' with dynamics func and paired parameter set")
        if not config_dict['time_step']:
            raise ValueError("config_dict requires key 'time_step' with system descritization timestep float")
        if not config_dict['c2d_method']:
            raise ValueError("config_dict requires key 'c2d_method' of val 'euler', 'zoh', or 'zohCombineed' for descritization method")

    def verify_input_des_seqs(self, u_des_seq, x_des_seq):    
        return
    
    def curry_c2d_func(self, config_dict):
        time_step  = self.time_step
        c2d_method = config_dict['c2d_method']
        c2d_func   = lambda cont_state_space: gen_ctrl.discretize_state_space(cont_state_space, time_step, c2d_method)
        return c2d_func

    def curry_lin_dyn_func(self, lin_dyn_sys_params:dict, gen_lin_dyn_sys):
        # create statespace from provided general statespace
        ss_cont               = gen_lin_dyn_sys(lin_dyn_sys_params)
        ss_discrete           = self.c2d_func(ss_cont)
        curried_lin_dyn_func  = lambda t, x, u: (ss_discrete.a @ x + ss_discrete @ u)
        return ss_discrete, curried_lin_dyn_func


class ttlqrControllerState:
    def __init__(self, u_len, x_len, len_seq):
        s_xx_seq    = np.zeros([len_seq, x_len, x_len])
        s_x_seq     = np.zeros([len_seq, x_len, 1])
        s_0_seq     = np.zeros([len_seq, 1])        
        K_gain_seq  = np.zeros([len_seq, u_len, 1])
        cost_seq    = np.zeros([len_seq, 1])
        cost_float  = 0.0
        ctrl_output = np.zeros([len_seq])

def calculate_backstep_params_seq(ctrl_config:ttlqrControllerConfig):
    if not ctrl_config.lin_dyn_sys.type == 'discrete': 
        raise TypeError('lin_dyn_sys is not discrete. state space must be discrete for function to be valid')
    len_seq = ctrl_config.len_seq
    x_len   = ctrl_config.x_len
    s_xx_seq = np.zeros([len_seq, x_len, x_len]) # type: ignore
    s_x_seq  = np.zeros([len_seq, x_len, 1    ]) # type: ignore
    s_0_seq  = np.zeros([len_seq, 1])            # type: ignore
    s_xx_seq[-1], s_x_seq[-1], s_x_seq[-1] = calculate_final_ctg_params(x_des_seq[-1], Qf) # type: ignore
    for idx in range(len_seq-1):
        k_step = - (idx + 2)
        s_xx_seq[k_step], s_x_seq[k_step], s_x_seq[k_step] = calculate_ctg_params(ctrl_config.Q, ctrl_config.R, ctrl_config.BRinvBT,
                                                                                  ctrl_config.x_des_seq[k_step],
                                                                                  ctrl_config.u_des_seq[k_step],
                                                                                  s_xx_seq[k_step+1],s_x_seq[k_step+1],s_0_seq[k_step+1],
                                                                                  ctrl_config.ss_discrete)
    if not verify_backstep_params(s_xx_seq, s_x_seq, s_0_seq):
        raise ValueError('invalid sequence of backstepped parameters, J(x,t)')
    return s_xx_seq, s_x_seq, s_0_seq

def calculate_final_ctg_params(Qf:npt.ArrayLike, x_des_N:npt.ArrayLike):
    # Calculate final cost to go
    s_xx_N = Qf
    s_x_N  = -(Qf @ x_des_N) # type: ignore
    s_0_N  = x_des_N.T @ Qf @ x_des_N  # type: ignore
    return s_xx_N, s_x_N, s_0_N

def calculate_ctg_params(Q:npt.ArrayLike, R:npt.ArrayLike, BRinvBT:npt.ArrayLike,
                         x_des_k:npt.ArrayLike, u_des_k:npt.ArrayLike, 
                         s_xx_kp1:npt.ArrayLike, s_x_kp1:npt.ArrayLike, s_0_kp1:npt.ArrayLike, 
                         lin_dyn_sys:gen_ctrl.stateSpace):
    A = lin_dyn_sys.a
    B = lin_dyn_sys.b
    s_xx_k = Q - (s_xx_kp1 @ BRinvBT @ s_xx_kp1) + (s_xx_kp1 @ A) + (A.T @ s_xx_kp1) # type: ignore
    s_x_k  = -(Q @ x_des_k) + ((A.T - s_xx_kp1 @ BRinvBT) @ s_x_kp1) + (s_xx_kp1 @ B @ u_des_k) # type: ignore
    s_0_k  = (x_des_k.T @ Q @ x_des_k) - (s_x_kp1.T @ BRinvBT @ s_x_kp1) + 2 * (s_x_kp1.T @ B @ u_des_k) # type: ignore
    return s_xx_k, s_x_k, s_0_k


def verify_backstep_params(s_xx_seq, s_x_seq, s_0_seq):
    is_valid_seqs_bool = True
    for idx in range(len(s_xx_seq)):
        if  not util.is_pos_def(s_xx_seq[idx]) or (s_0_seq[idx] < (s_x_seq[idx]).T @ np.linalg.inv(s_xx_seq[idx]) @ s_x_seq[idx]):
            is_valid_seqs_bool = False
            break
    return is_valid_seqs_bool

def calculate_u(RinvBT, u_des_k, S_xx_k, s_x_k, x_k):
    u_k = u_des_k - RinvBT @ (S_xx_k @ x_k + s_x_k)
    return u_k