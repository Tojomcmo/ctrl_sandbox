from jax import numpy as jnp

def cost_func_quad_state_and_control(cost_func_params:dict, state_vec, control_vec, is_final_bool=False):
# This function calculates a quadratic cost wrt state and control
# Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
# R[in]     - Control cost matrix, square, dim(control, control) Positive definite
# S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
# cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
    Q = cost_func_params['Q']
    R = cost_func_params['R']
    Qf = cost_func_params['Qf']
    # check that dimensions match [TODO]
    if is_final_bool:
        cost = (0.5) * (  state_vec   @ Qf @ jnp.transpose(state_vec  ))
    else:
        cost = (0.5) * ( (state_vec   @ Q  @ jnp.transpose(state_vec  ))
                        +(control_vec @ R  @ jnp.transpose(control_vec)))
    return cost                   