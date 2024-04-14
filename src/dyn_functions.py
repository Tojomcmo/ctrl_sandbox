from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
from . import ilqr_utils as util
from . import gen_ctrl_funcs as gen_ctrl


class dynFuncParams:
    def __init__(self) -> None:
        pass

class nlPendParams(dynFuncParams):
    def __init__(self, g:float, b:float, l:float) -> None:
        super().__init__()
        self.g = g
        self.b = b
        self.l = l    

def pend_dyn_nl(params:nlPendParams, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])->npt.NDArray:
# continuous time dynamic equation for simple pendulum 
# time[in]       - time component, necessary prarmeter for ode integration
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
    state_dot = jnp.array([
                state[1],
                -(params.b/params.l) * state[1] - (jnp.sin(state[0]) * params.g/params.l) + control[0]
                ])
    return state_dot

def pend_dyn_lin(params:dict, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])->npt.NDArray:
# continuous time dynamic equation for simple pendulum 
# time[in]       - time component, necessary prarmeter for ode integration
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
    g:float = params['g']
    l:float = params['l']
    b:float = params['b']
    vel = state[1]
    pos = state[0]
    tq  = control[0]
    state_dot = jnp.array([
                vel,
                -((b/l) * vel) - ((g/l) * pos) + tq
                ])
    return state_dot


def lti_pend_ss_cont(params:dict) -> gen_ctrl.stateSpace:
    g = params['g']
    l = params['l']
    b = params['b']
    A = np.array([[   0,    1],
                  [-g/l, -b/l]])
    B = np.array([[0],[1]])
    C = np.eye(2)
    D = np.zeros([2,1])
    ss_cont = gen_ctrl.stateSpace(A, B, C, D)
    return ss_cont