from jax import numpy as jnp
import ilqr_utils as util

def pend_dyn_nl(params, time, state, control):
# continuous time dynamic equation for simple pendulum 
# time[in]       - time component, necessary prarmeter for ode integration
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
    g = params['g']
    b = params['b']
    l = params['l']
    state_dot = jnp.array([
                state[1],
                -(b/l) * state[1] - (jnp.sin(state[0]) * g/l) + control[0]
                ])
    return state_dot

def pend_dyn_lin(params, time, state, control):
# continuous time dynamic equation for simple pendulum 
# time[in]       - time component, necessary prarmeter for ode integration
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
    g = params['g']
    l = params['l']
    b = params['b']
    vel = state[1]
    pos = state[0]
    tq  = control[0]
    state_dot = jnp.array([
                vel,
                -((b/l) * vel) - ((g/l) * pos) + tq
                ])
    return state_dot