import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.signal import StateSpace

import time

from numpy import typing as npt
from typing import  Callable, Tuple
from scipy.integrate import solve_ivp


class stateSpace:
    def __init__(self, a_mat:npt.NDArray, b_mat:npt.NDArray, c_mat:npt.NDArray, d_mat:npt.NDArray, time_step = None):
        if a_mat.shape[0] != a_mat.shape[1]:
            raise ValueError('A matrix must be square')
        elif a_mat.shape[0] != b_mat.shape[0]:
            raise ValueError('A and B matrices must have same n(state) dimension')
        elif a_mat.shape[0] != c_mat.shape[1]:
            raise ValueError('A and C matrices must have same m(state) dimension')
        elif b_mat.shape[1] != d_mat.shape[1]:
            raise ValueError('B and D matrices must have the same m(control) dimension')
        elif c_mat.shape[0] != d_mat.shape[0]:
            raise ValueError('C and D matrices must have the same n(measurement) dimension')
        else:
            if time_step is None:
                self.a = a_mat
                self.b = b_mat
                self.c = c_mat
                self.d = d_mat
                self.time_step = time_step
                self.type = 'continuous'
            elif isinstance(time_step, float) and time_step > 0:
                self.a = a_mat
                self.b = b_mat
                self.c = c_mat
                self.d = d_mat
                self.time_step = time_step
                self.type = 'discrete'
            else:
                raise ValueError(f'invalid time step definition {time_step}. time_step must be a positive float or None')

# def simulate_forward_dynamics_seq(
#         discrete_dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
#         x_init:npt.NDArray[np.float64], 
#         u_seq:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
#     # dyn_func[in]     - continuous dynamics function (pre-populated with *params) for integration
#     # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
#     # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
#     # time_step[in]    - time interval between sequence points (scalar > 0)
#     # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
#     # sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'    
#     # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
#     len_seq  = len(u_seq)  
#     x_seq    = np.zeros([(len_seq+1),len(x_init)], dtype=float)
#     x_seq[0] = x_init # type: ignore
#     for idx in range(len_seq):
#         x_seq[idx+1] = np.array(discrete_dyn_func(jnp.array(x_seq[idx]), jnp.array(u_seq[idx]))) # type: ignore    
#     return x_seq

def simulate_forward_dynamics_seq(
        discrete_dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        x_init:jnp.ndarray, 
        u_seq:jnp.ndarray) -> jnp.ndarray:
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - continuous dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'    
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    discrete_dyn_func_scan = lambda carry_x_k, u_seq: scan_dyn_for_sim(discrete_dyn_func, carry_x_k, u_seq)
    _, x_seq_out = lax.scan(discrete_dyn_func_scan, x_init,u_seq)
    return jnp.append(x_init.reshape(1,-1), x_seq_out, axis=0)

def scan_dyn_for_sim(discrete_dyn_func, carry_x_k, u_seq)->Tuple[jnp.ndarray, jnp.ndarray]:
    x_kp1 = discrete_dyn_func(carry_x_k, u_seq)
    return x_kp1, x_kp1

def simulate_forward_dynamics_step(
        dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        x_k:jnp.ndarray, u_k:jnp.ndarray, 
        time_step:float, 
        sim_method:str = 'rk4') -> jnp.ndarray:
    '''
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory\n
    dyn_func[in]   - continuous dynamics function (pre-populated with *params) for integration\n
    u_k[in]        - control input (scalar or vector depending on control dimension -- must match dyn_func control dim)\n
    x_k[in]        - starting state for integration (vector, must match dyn_func state dim)\n
    time_step[in]  - time interval between sequence points (scalar > 0)\n
    sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'\n
    x_kp1[out]     - array shape[1,state_dim] of state vector at next time step
    '''   
    if   sim_method == 'rk4':
        x_kp1        = step_rk4(dyn_func, time_step, x_k, u_k)
    elif sim_method == 'euler':
        x_kp1        = x_k + dyn_func(x_k, u_k) * time_step
    elif sim_method == 'solve_ivp_zoh':
        time_span    = (0,time_step)
        dyn_func_zoh = (lambda time_dyn, state: dyn_func(state, u_k.reshape(-1)))
        y_0          = (x_k).reshape(-1)
        result_ivp   = solve_ivp(dyn_func_zoh, time_span, y_0)
        x_kp1        = jnp.array(result_ivp.y[:,-1])
    else:
        raise ValueError("invalid simulation method")
    return x_kp1.reshape(-1)

def linearize_dynamics(
        dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        xu_k:jnp.ndarray, x_k_len) -> Tuple[jnp.ndarray, jnp.ndarray]:
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] dynamics function, may be continuous or discrete
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous or discrete time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous or discrete time linearization of dyn_func wrt control eval at x,u
    dyn_func_xu = lambda xu_k: dyn_func(xu_k[:x_k_len], xu_k[x_k_len:])
    ab_lin_cat  = jax.jacfwd(dyn_func_xu)(xu_k)
    a_lin       = ab_lin_cat[:x_k_len,:x_k_len]
    b_lin       = ab_lin_cat[:x_k_len,x_k_len:]
    return a_lin, b_lin

def dyn_lin_scan_func(dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],x_len:int, 
                      carry:None, x_and_u_k:jnp.ndarray) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:
    a_lin, b_lin = linearize_dynamics(dyn_func, x_and_u_k, x_len)
    return carry, (a_lin, b_lin)

def curry_dyn_lin_scan_func(discrete_dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                            num_states:int)->Callable[[None, jnp.ndarray],Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]]:
    def dyn_lin_scan_func_curried(carry:None, xu_k:jnp.ndarray):
        carry_out, A_B_k = dyn_lin_scan_func(discrete_dyn_func, num_states, carry, xu_k)
        return carry_out, A_B_k
    return dyn_lin_scan_func_curried

def step_euler_forward(
        dyn_func:Callable[[jnp.ndarray, jnp.ndarray], npt.NDArray], 
        h:float, 
        x_k:jnp.ndarray, 
        u_k:jnp.ndarray) -> jnp.ndarray:
    return x_k + dyn_func(x_k, u_k) * h

def step_rk4(
        dyn_func_ad:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        h:float, 
        x_k:jnp.ndarray, 
        u_k:jnp.ndarray) -> jnp.ndarray:
    #rk4 integration with zero-order hold on u
    f1 = dyn_func_ad(x_k            , u_k)
    f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
    f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
    f4 = dyn_func_ad(x_k + h*f3     , u_k)
    return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)

def step_solve_ivp(
        dyn_func:Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
        h:float, 
        x_k:npt.NDArray[np.float64], 
        u_k:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    time_span    = (0,h)
    dyn_func_zoh = (lambda time_dyn, state: dyn_func(state, u_k.reshape(-1)))
    y_0          = (x_k).reshape(-1)
    result_ivp   = solve_ivp(dyn_func_zoh, time_span, y_0)
    x_kp1        = (np.array([result_ivp.y[:,-1]])).reshape(-1,1)
    return x_kp1

def calculate_linearized_state_space_seq(
        lin_scan_func_curried:Callable[[None, jnp.ndarray],Tuple[None, Tuple[jnp.ndarray,jnp.ndarray]]], 
        x_seq:jnp.ndarray, 
        u_seq:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    xu_seq = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
    carry = None
    _, (a_lin_seq, b_lin_seq) = lax.scan(lin_scan_func_curried, carry, xu_seq)
    return a_lin_seq, b_lin_seq

def discretize_continuous_state_space(input_state_space:stateSpace, time_step:float, c2d_method:str = 'euler') -> stateSpace:
#   A - (nxn) - continuous state transition matrix
#   B - (nxm) - continuous control matrix
#   C - (pxn) - continuous state measurement matrix
#   D - (pxm) - continuous direct feedthrough matrix
#   Continuous state space:
#   xdot(t) = A * x(t) + B * u(t)
#   y(t)    = C * x(t) + D * u(t)
#   transformation for zohCombined:
#   e^([[A, B],[0,0]] * time_step) = [[Ad, Bd],[0,I]]
#   Discrete state space:
#   x[t+timestep] = Ad * x[t] + Bd * u[t]
#   y[t]          = Cd * x[t] + Dd * u[t]

    func_error = False

    if input_state_space.type == 'discrete':
        raise TypeError('input state space is already discrete')

    else:
        a_d = np.zeros(input_state_space.a.shape)
        b_d = np.zeros(input_state_space.b.shape)
        c_d = np.zeros(input_state_space.c.shape)
        d_d = np.zeros(input_state_space.d.shape)
        n  = input_state_space.a.shape[0]
        m  = input_state_space.b.shape[1]
        p  = input_state_space.c.shape[0]

        if c2d_method=='euler' :
            a_d = np.eye(n) + (input_state_space.a * time_step)
            b_d = input_state_space.b * time_step
            c_d = input_state_space.c
            d_d = input_state_space.d

        elif c2d_method=='zoh' :
            if scipy.linalg.det(input_state_space.a) > 10E-8:
                a_d = scipy.linalg.expm(input_state_space.a * time_step)
                b_d = np.linalg.inv(input_state_space.a) @ (a_d - np.eye(n)) @ input_state_space.b
                c_d = input_state_space.c
                d_d = input_state_space.d
            else:
                raise ValueError('determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid')

        elif c2d_method=='zohCombined':
        #   create combined A B matrix e^([[A, B],[0,0]]
            a_b_c:npt.NDArray  = np.concatenate((np.concatenate((input_state_space.a, input_state_space.b),axis=1),np.zeros((m,n + m))), axis=0)
            a_b_d:npt.NDArray  = scipy.linalg.expm(a_b_c * time_step) 
            a_d   = a_b_d[:n,:n]
            b_d   = a_b_d[:n,n:] 
            c_d   = input_state_space.c
            d_d   = input_state_space.d
        else:
            raise ValueError('invalid discretization method')
    d_state_space = stateSpace(a_d, b_d, c_d, d_d, time_step)
    return d_state_space

def combine_x_u_for_cost_scan(x_seq:jnp.ndarray, u_seq:jnp.ndarray)->jnp.ndarray:
    u_seq_end = jnp.zeros((1,u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    return jnp.concatenate([x_seq, u_seq_mod], axis=1)

def scan_cost_func_for_calc(cost_func:Callable[[jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]], 
                            x_len, carry:Tuple[int, jnp.ndarray], xu_k)-> Tuple[Tuple[int,jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    count, cost_sum = carry
    (total_cost, x_cost, u_cost) = cost_func(xu_k[:x_len], xu_k[x_len:], carry[0])
    new_carry = (count+1, cost_sum + total_cost.reshape(-1))
    return new_carry, (total_cost, x_cost, u_cost)

def calculate_total_cost(
        cost_func:Callable[[jnp.ndarray, jnp.ndarray, int],Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]], 
        x_seq:jnp.ndarray, 
        u_seq:jnp.ndarray) -> Tuple[float, jnp.ndarray,jnp.ndarray,jnp.ndarray]:
    # check state_seq and control_seq are valid lengths
    # Calculate total cost
    xu_seq                             = combine_x_u_for_cost_scan(x_seq, u_seq)
    init_carry:Tuple[int, jnp.ndarray] = (0, jnp.zeros([1], dtype=float))
    scan_cost_func_for_calc_curried    = lambda carry, xu_k : scan_cost_func_for_calc(cost_func, x_seq.shape[1], carry, xu_k)
    (_, total_cost), (total_cost_seq, x_cost_seq, u_cost_seq) = lax.scan(scan_cost_func_for_calc_curried, init_carry,xu_seq)
    return total_cost.item(), total_cost_seq, x_cost_seq, u_cost_seq

def taylor_expand_cost(
        cost_func:Callable[[jnp.ndarray, jnp.ndarray, int],jnp.ndarray], 
        xu_k:jnp.ndarray,
        x_k_len:int, k:int)-> Tuple[jnp.ndarray,
                                    jnp.ndarray,
                                    jnp.ndarray,
                                    jnp.ndarray,
                                    jnp.ndarray]:
    ''' 
    This function creates a quadratic approximation of the cost function Using taylor expansion.
    Expansion is approximated about the rolled out trajectory
    '''
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu:Callable[[jnp.ndarray],jnp.ndarray] = lambda xu_k: cost_func(xu_k[:x_k_len], 
                                                                              xu_k[x_k_len:], 
                                                                              k)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat  = (jax.jacfwd(cost_func_xu)(xu_k)).reshape(len(xu_k),1) 
    # calculate the lumped hessian matrix for the cost function at the primal point
    hes_cat  = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k)).reshape(len(xu_k),len(xu_k))
    l_x  = jac_cat[:x_k_len]
    l_u  = jac_cat[x_k_len:]
    l_xx = hes_cat[:x_k_len,:x_k_len]
    l_uu = hes_cat[x_k_len:,x_k_len:]
    l_ux = hes_cat[x_k_len:,:x_k_len]
    return l_x, l_u, l_xx, l_uu, l_ux

def taylor_expand_cost_seq(cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
                                       x_seq:jnp.ndarray,
                                       u_seq:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_seq_end = jnp.zeros((1,u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    xu_seq    = jnp.concatenate([x_seq, u_seq_mod], axis=1)
    carry:int = 0
    lin_scan_func_curried = lambda carry, xu_seq: scan_cost_func_for_exp(cost_func, x_seq.shape[1], carry, xu_seq)
    carry, expanded_mats      = lax.scan(lin_scan_func_curried, carry, xu_seq)
    return expanded_mats

def scan_cost_func_for_exp(cost_func:Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],x_len:int, 
                           carry:int, xu_k:jnp.ndarray) -> Tuple[int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, xu_k, x_len, carry)
    carry += 1
    return carry, (l_x, l_u, l_xx, l_uu, l_ux)


def prep_xu_vec_for_diff(x_k:npt.NDArray[np.float64],u_k:npt.NDArray[np.float64])->Tuple[jnp.ndarray, int, int]:
    '''x_k and u_k must each be shape (n,) and (m,) respectively'''
    xu_k       = np.concatenate([x_k, u_k])
    x_k_len    = len(x_k)
    xu_k_len   = len(xu_k)
    xu_k_jax   = jnp.array(xu_k)
    return xu_k_jax, xu_k_len, x_k_len

def ss_2_dyn_func(ss_cont:stateSpace):
    return lambda t, x, u: ss_cont.a @ x.reshape(-1,1) + ss_cont.b @ u.reshape(-1,1)

def calculate_s_xx_dot(Q_t:npt.NDArray[np.float64], 
                       A_t:npt.NDArray[np.float64], 
                       BRinvBT_t:npt.NDArray[np.float64], 
                       s_xx_t:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    s_xx_dt = -(Q_t - (s_xx_t @ BRinvBT_t @ s_xx_t) + (s_xx_t @ A_t) + (A_t.T @ s_xx_t))
    return s_xx_dt

def prep_cost_func_for_diff(cost_func:Callable[
    [jnp.ndarray, jnp.ndarray, int],
    Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]]) -> Callable[
    [jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    '''
    Function for reducing generic cost function output to single queried value.\n
    useful for preparing functions for jax differentiation
    '''
    def cost_func_for_diff(x_k:jnp.ndarray, u_k:jnp.ndarray, k_step:int) -> jnp.ndarray:
        cost_out = cost_func(x_k, u_k, k_step)
        return cost_out[0]
    return cost_func_for_diff

def prep_cost_func_for_calc(cost_func:Callable[
                            [jnp.ndarray,jnp.ndarray,int],
                            Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]]) -> Callable[
                            [npt.NDArray[np.float64],npt.NDArray[np.float64],int],
                            Tuple[float,float,float]]:
    
    def cost_func_for_calc(x_k:npt.NDArray[np.float64], 
                           u_k:npt.NDArray[np.float64], 
                           k:int) -> Tuple[float,float,float]:
        total_cost, x_cost, u_cost = cost_func(jnp.array(x_k), jnp.array(u_k), k)
        return total_cost.item(), x_cost.item(), u_cost.item() 
    return cost_func_for_calc

def c2d_AB_zoh_combined(
        A:npt.NDArray[np.float64], 
        B:npt.NDArray[np.float64], 
        time_step:float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n  = A.shape[0]
    m  = B.shape[1]
    a_b_c:npt.NDArray  = np.concatenate((np.concatenate((A, B),axis=1),np.zeros((m,n + m))), axis=0)
    a_b_d:npt.NDArray  = scipy.linalg.expm(a_b_c * time_step) 
    Ad   = a_b_d[:n,:n]
    Bd   = a_b_d[:n,n:] 
    return Ad, Bd
        
def convert_d2d_A_B(Ad_1, Bd_1, dt_1, dt_2):
    I =np.eye(Ad_1.shape[0])
    A = (1/dt_1)*scipy.linalg.logm(Ad_1)
    A_inv = np.linalg.inv(A)
    B = (np.linalg.inv(A_inv @ (np.linalg.inv(Ad_1 - I)))) @ Bd_1
    
    Ad_2 = calculate_matrix_power_similarity_transform(Ad_1, dt_2/dt_1)
    Bd_2 = A_inv @ (Ad_2 - I) @ B
    Cd_2 = 0
    Dd_2 = 0
    return Ad_2, Bd_2

def calculate_matrix_power_similarity_transform(A, exponent):
    evals, evecs = np.linalg.eig(A)
    D = np.diag(evals)
    evecs_inv = np.linalg.inv(evecs)
    A_to_exponent = evecs @ D**exponent @ evecs_inv
    return A_to_exponent

def calculate_ff_fb_u(k_fb_k:jnp.ndarray, 
                     u_ff_k:jnp.ndarray, 
                     x_des_k:jnp.ndarray,
                     x_k:jnp.ndarray) -> jnp.ndarray:
    return (u_ff_k.reshape(-1,1) + k_fb_k @ (x_k.reshape(-1,1) - x_des_k.reshape(-1,1))).reshape(-1)

def apply_cov_noise_to_vec(vec:jnp.ndarray, cov_mat:jnp.ndarray)->jnp.ndarray:
    return vec + jax.random.multivariate_normal(cov_mat.shape[0],np.zeros(cov_mat.shape[0]),cov_mat)