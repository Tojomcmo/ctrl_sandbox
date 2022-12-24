from jax import numpy as jnp

def cost_func_quad_state_and_control(Q, R, S, state_vec, control_vec = None):
    if control_vec == None:
        cost = (0.5)*state_vec @ S @ jnp.transpose(state_vec)
    else:
        cost = (0.5) * (jnp.transpose(state_vec) @ Q @ state_vec + jnp.transpose(control_vec) @ R @ control_vec)    