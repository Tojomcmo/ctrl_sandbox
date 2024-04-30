from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
from numpy.linalg import inv

import ilqr_utils as util
import gen_ctrl_funcs as gen_ctrl


class dynFuncParams:
    def __init__(self) -> None:
        pass

class nlPendParams(dynFuncParams):
    def __init__(self, g:float, b:float, l:float) -> None:
        super().__init__()
        self.g = g
        self.b = b
        self.l = l

class nlDoublePendParams(dynFuncParams):
    def __init__(self, g:float, m1:float,          l1:float, 
                                m2:float,          l2:float,
                                b1:float,          b2:float,                                 
                                shoulder_act:bool, elbow_act:bool):
        super().__init__()
        self.g  = g
        self.m1 = m1
        self.l1 = l1
        self.m2 = m2
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2
        self.shoulder_act = shoulder_act
        self.elbow_act    = elbow_act
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, -1],
                                    [0,  1]],dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1],
                                    [0]],dtype=float) 
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[-1],
                                    [1]],dtype=float)
        else:
            print('double pendulum dynamics function is fully passive. Control array must be len(1,)')
            self.B_mat = jnp.zeros([2,1])     

class nlDoublePend2Params(dynFuncParams):
    def __init__(self, g:float, m1:float,        moi1:float,         
                                d1:float,          l1:float, 
                                m2:float,        moi2:float,   
                                d2:float,          l2:float,
                                b1:float,          b2:float,                                 
                                shoulder_act:bool, elbow_act:bool):
        super().__init__()
        self.g  = g
        self.m1 = m1
        self.moi1 = moi1                
        self.d1 = d1
        self.l1 = l1
        self.m2 = m2
        self.moi2 = moi2
        self.d2 = d2
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2
        self.shoulder_act = shoulder_act
        self.elbow_act    = elbow_act
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, -1],
                                    [0,  1]],dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1],
                                    [0]],dtype=float) 
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[-1],
                                    [1]],dtype=float)
        else:
            print('double pendulum dynamics function is fully passive. Control array must be len(1,)')
            self.B_mat = jnp.zeros([2,1])  

def pend_dyn_nl(params:nlPendParams, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])->npt.NDArray:
# continuous time dynamic equation for simple pendulum 
# time[in]       - time component, necessary prarmeter for ode integration
# state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
# control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
# params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
# state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
    x_jax = jnp.array(state)
    u_jax = jnp.array(control)
    state_dot = jnp.array([
                state[1],
                -(params.b/params.l) * x_jax[1] - (jnp.sin(x_jax[0]) * params.g/params.l) + u_jax[0]
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


def double_pend_no_damp_full_act_dyn(params:nlDoublePendParams, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])-> npt.NDArray:
    '''
    The state vector has form x = [th1, th2, thdot1, thdot2]'
    The control vector has form u = [tq1, tq2]'
    The output vector has form xdot = [thdot1, thdot2, thddot1, thddot2]
    helpful resource https://dassencio.org/33
    Derivation in github wiki https://github.com/Tojomcmo/ctrl_sandbox/wiki
    Thetas are both reference to intertial frame down position
    '''
    x_jax = jnp.array(state)
    u_jax = jnp.array(control)
    m1 = params.m1
    m2 = params.m2
    l1 = params.l1
    l2 = params.l2
    b1 = params.b1
    b2 = params.b2
    g  = params.g
    th1 = x_jax[0]
    th2 = x_jax[1]
    thdot1 = x_jax[2]
    thdot2 = x_jax[3]  
    c1m2 = jnp.cos(th1-th2)
    s1m2 = jnp.sin(th1-th2)
    m1pm2 = m1 + m2

    mass_matrix = jnp.array([[(m1 + m2) * l1**2            , m2 * l1 * l2 * c1m2 ],
                             [m2 * l1 * l2 * c1m2,    m2 * l2**2                 ]])
    # det_denom = 1/((m1pm2 * m2 * l1**2 * l2**2) - (m2 * l1 * l2 * c1m2)**2)
    # M_inv_mat = det_denom * jnp.array([[ m2 * l2**2                     , -m2 * l1 * l2 * c1m2],
    #                                    [-m2 * l1 * l2 * c1m2,  m1pm2 * l1**2                  ]])

    M_inv_mat = jnp.linalg.inv(mass_matrix)
    gen_dyn_mat = jnp.array([[-m2*l1*l2*thdot2**2*s1m2  - (b1+b2)*thdot1 + (b2)*thdot2],
                             [ m2*l1*l2*thdot1**2*s1m2  +    (b2)*thdot1 - (b2)*thdot2]])

    grav_mat   = jnp.array([[ - m1pm2*l1*g*jnp.sin(th1)],
                            [ -  m2  *l2*g*jnp.sin(th2)]])               

    theta_ddots = (M_inv_mat @ (gen_dyn_mat + grav_mat + params.B_mat @ u_jax.reshape(-1,1))).reshape(-1)
    state_dot = jnp.array([thdot1,thdot2,theta_ddots[0],theta_ddots[1]])
    return state_dot