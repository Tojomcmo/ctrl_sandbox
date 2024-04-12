from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
from . import ilqr_utils as util

# library of cost functions and associated functions for manipulating cost functions
# Cost functions may return multiple values:
#  - first value MUST be the float value of calculated query cost


def cost_func_quad_state_and_control(cost_func_params:dict, x_vec:npt.NDArray, u_vec:npt.NDArray, k_step:int, x_des_seq:npt.NDArray, u_des_seq:npt.NDArray, is_final_bool:bool):
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
        x_k_corr = util.calc_and_shape_array_diff(x_vec  , x_des_seq[k_step], shape='col')
        x_cost     = ((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr)).item()
        u_cost     = 0.0
        total_cost = x_cost
    elif k_step == 0:
        u_k_corr   = util.calc_and_shape_array_diff(u_vec, u_des_seq[k_step], shape='col')
        x_cost     = 0.0    
        u_cost     = (0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr).item()
        total_cost = u_cost
    else:
        x_k_corr = util.calc_and_shape_array_diff(x_vec, x_des_seq[k_step], shape='col')   
        u_k_corr = util.calc_and_shape_array_diff(u_vec, u_des_seq[k_step], shape='col')
        x_cost     = ((0.5) * (jnp.transpose(x_k_corr) @ Q @ x_k_corr)).item()
        u_cost     = ((0.5) * (jnp.transpose(u_k_corr) @ R @ u_k_corr)).item()
        total_cost = ((0.5) * ((jnp.transpose(x_k_corr) @ Q @ x_k_corr)+(jnp.transpose(u_k_corr) @ R @ u_k_corr))).item() 
    return total_cost, x_cost, u_cost                   
