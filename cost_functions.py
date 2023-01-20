from jax import numpy as jnp

def cost_func_quad_state_and_control(Q, R, state_vec, control_vec):
# This function calculates a quadratic cost wrt state and control
# Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
# R[in]     - Control cost matrix, square, dim(control, control) Positive definite
# S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
# cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   

    cost = (0.5) * ((state_vec   @ Q @ jnp.transpose(state_vec  )) 
                   +(control_vec @ R @ jnp.transpose(control_vec)))    
    return cost                   