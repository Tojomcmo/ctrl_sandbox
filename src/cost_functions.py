from jax import numpy as jnp
from . import ilqr_utils as util

# library of cost functions and associated functions for manipulating cost functions
# Cost functions may return multiple values:
#  - first value MUST be the float value of calculated query cost

def prep_cost_func_for_diff(cost_func):
    # Function for reducing generic cost function output to single queried value
    # useful for preparing functions for jax differentiation
    def cost_func_for_diff(vec):
        cost_out = cost_func(vec)
        return cost_out[0]
    return cost_func_for_diff

def cost_func_quad_state_and_control(cost_func_params:dict, state_vec, control_vec, k_step, state_des_seq, control_des_seq, is_final_bool=False):
# This function calculates a quadratic cost wrt state and control
# Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
# R[in]     - Control cost matrix, square, dim(control, control) Positive definite
# Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
# cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
    Q  = cost_func_params['Q']
    R  = cost_func_params['R']
    Qf = cost_func_params['Qf']
    # check that dimensions match [TODO]
    if is_final_bool:
        x_k_corr     = util.calc_and_shape_array_diff(state_vec  , state_des_seq[k_step], shape='col')
        state_cost   = (0.5) * (jnp.transpose(x_k_corr)  @ Qf @ x_k_corr)
        control_cost = 0.0
    elif k_step == 0:
        u_k_corr     = util.calc_and_shape_array_diff(control_vec, control_des_seq[k_step], shape='col')
        state_cost   = 0.0        
        control_cost = (0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr)
    else:
        x_k_corr     = util.calc_and_shape_array_diff(state_vec  , state_des_seq[k_step], shape='col')   
        u_k_corr     = util.calc_and_shape_array_diff(control_vec, control_des_seq[k_step], shape='col')
        state_cost   = (0.5) * (jnp.transpose(x_k_corr) @ Q  @ x_k_corr)
        control_cost = (0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr)
    cost_float = state_cost + control_cost    
    return cost_float#, state_cost, control_cost                   