import numpy as np

import src.gen_ctrl_funcs as gen_ctrl
import src.ttlqr_funcs as ttlqr



# Define system paramaters
lin_dyn_sys_generic = 
lin_dyn_sys_params = 
quad_cost_func_params = {Q,R,Qf}
time_step = 
len_seq = 

# Define trajectory
x_init_vec = 
u_traj_gen_seq = 
x_des_seq = gen_ctrl.simulate_forward_dynamics(lin_dyn_sys, x_init_vec, u_traj_gen_seq, time_step)

# Initialize controller with system and parameters
class ttlqrControllerConfig
    cost_func_w_params
    lin_dyn_sys_w_params
    u_des_seq = u_traj_gen_seq
    len_seq
    time_step
    x_init_vec

class ttlqrConfigFuncs
    cost_func_w_params
    lin_dyn_sys_w_params

class ttlqrControllerState
    P_k_seq
    p_k_seq
    K_gain_seq
    cost_seq

# Run controller
    S_xx_N, s_x_N = ttlqr.calculate_final_ctg_params
    S_xx_seq, s_x_seq, s_0_seq =  def calculate_ctg_params_seq
    is_valid_calc_bool = verify_ctg_parameters  # J must be uniformly positive
    u_output_params_seq = def calculate_u_params(R, B, S_xx_k, s_x_k)
# simulate system response

# simulate system response with disturbance

# plot / analyze output