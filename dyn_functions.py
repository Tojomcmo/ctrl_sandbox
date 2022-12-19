from jax import numpy as jnp

def pend_dyn_nl(state, control, g = 1.0, l = 1.0, b = 1.0 ):
# continuous time dynamic equation for simple pendulum 
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot

    state_dot = jnp.array([
                state[1],
                -(b/l) * state[1] - (jnp.sin(state[0]) * g/l) + control[0]
                 ])
    return state_dot
