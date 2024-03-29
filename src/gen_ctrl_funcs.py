import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from numpy import typing as npt
from typing import Optional
from scipy.integrate import solve_ivp

class stateSpace:
    def __init__(self, a_mat, b_mat, c_mat, d_mat, time_step = None):
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

def simulate_forward_dynamics_seq(dyn_func, state_init, control_seq, time_step, sim_method = 'rk4'):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - continuous dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'    
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    len_seq: int             = len(control_seq)  
    state_seq: npt.ArrayLike = np.zeros([(len_seq+1),len(state_init), 1])
    state_seq[0]   = state_init # type: ignore
    for idx in range(len_seq):
        state_next = simulate_forward_dynamics_step(dyn_func, state_seq[idx], control_seq[idx], time_step, sim_method)
        state_seq[idx+1] = state_next # type: ignore    
    return state_seq

def simulate_forward_dynamics_step(dyn_func, x_k, u_k, time_step, sim_method = 'rk4'):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]   - continuous dynamics function (pre-populated with *params) for integration
    # u_k[in]        - control input (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # x_k[in]        - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]  - time interval between sequence points (scalar > 0)
    # sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'
    # x_kp1[out]     - array shape[1,state_dim] of state vector at next time step
    x_len      = len(x_k)  
    x_kp1      = np.zeros([x_len, 1])
    if   sim_method == 'rk4':
        x_kp1        = dynamics_rk4(dyn_func, time_step, x_k, u_k)
    elif sim_method == 'euler':
        x_kp1      = x_k + dyn_func(x_k, u_k) * time_step
    elif sim_method == 'solve_ivp_zoh':
        time_span    = (0,time_step)
        dyn_func_zoh = (lambda time_dyn, state: dyn_func(state, u_k.reshape(-1)))
        y_0          = (x_k).reshape(-1)
        result_ivp   = solve_ivp(dyn_func_zoh, time_span, y_0)
        x_kp1        = (np.array([result_ivp.y[:,-1]])).reshape(-1,1)
    else:
        raise ValueError("invalid simulation method")
    return x_kp1

def linearize_dynamics(dyn_func, x_k, u_k):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u
    xu_k_jax, xu_k_len, x_k_len = prep_xu_vec_for_diff(x_k, u_k)
    dyn_func_xu = lambda xu_k: dyn_func(xu_k[:x_k_len], xu_k[x_k_len:])
    ab_lin_cat  = jax.jacfwd(dyn_func_xu)(xu_k_jax[:,0])
    a_lin       = np.array(ab_lin_cat[:x_k_len,:x_k_len])
    b_lin       = np.array(ab_lin_cat[:x_k_len,x_k_len:])
    return a_lin, b_lin

def dynamics_rk4(dyn_func, h, x_k, u_k):
    #rk4 integration with zero-order hold on u
    f1 = dyn_func(x_k,u_k)
    f2 = dyn_func(x_k + 0.5*h*f1, u_k)
    f3 = dyn_func(x_k + 0.5*h*f2, u_k)
    f4 = dyn_func(x_k + h*f3, u_k)
    return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)

def calculate_linearized_state_space_seq(dyn_func, c2d_ss_func, state_seq, control_seq, time_seq):
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    if len(state_seq) != (len(control_seq)+1):
        raise ValueError('state and control sequences are incompatible lengths. state seq must be control seq length +1')
    else:
        len_seq     = len(time_seq)
        state_len   = jnp.shape(state_seq[0])[0]
        control_len = jnp.shape(control_seq[0])[0]
        a_lin_array = [None] * (len_seq-1)
        b_lin_array = [None] * (len_seq-1)
        c_mat_dummy = jnp.zeros((1,state_len))
        d_mat_dummy = jnp.zeros((1,control_len))
        for idx in range(len_seq-1):
            a_lin, b_lin             = linearize_dynamics(dyn_func, state_seq[idx], control_seq[idx])
            ss_pend_lin_continuous   = stateSpace(a_lin, b_lin, c_mat_dummy, d_mat_dummy)
            ss_pend_lin_discrete     = c2d_ss_func(ss_pend_lin_continuous)
            a_lin_array[idx]         = ss_pend_lin_discrete.a
            b_lin_array[idx]         = ss_pend_lin_discrete.b
    return a_lin_array, b_lin_array

def discretize_continuous_state_space(input_state_space:stateSpace, time_step:float, c2d_method='Euler'):
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
            a_b_c:npt.ArrayLike  = np.concatenate((np.concatenate((input_state_space.a, input_state_space.b),axis=1),np.zeros((m,n + m))), axis=0)
            a_b_d:npt.ArrayLike  = scipy.linalg.expm(a_b_c * time_step) # type: ignore
            a_d   = a_b_d[:n,:n] # type: ignore
            b_d   = a_b_d[:n,n:] # type: ignore
            c_d   = input_state_space.c
            d_d   = input_state_space.d
        else:
            raise ValueError('invalid discretization method')
    d_state_space = stateSpace(a_d, b_d, c_d, d_d, time_step)
    return d_state_space

def calculate_total_cost(cost_func, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # Calculate total cost
    len_seq    = len(state_seq)
    u_len      = len(control_seq[0])
    cost_seq   = np.zeros([len_seq,1])
    x_cost_seq = np.zeros([len_seq,1])    
    u_cost_seq = np.zeros([len_seq,1])
    total_cost = 0
    for idx in range(len_seq):
        if (idx == len_seq-1):
            inc_cost, inc_x_cost, inc_u_cost = cost_func(state_seq[idx], np.zeros([u_len,1]), idx, is_final_bool=True)
        else:
            inc_cost, inc_x_cost, inc_u_cost = cost_func(state_seq[idx], control_seq[idx], idx)
        total_cost += inc_cost 
        cost_seq[idx] = inc_cost
        x_cost_seq[idx] = inc_x_cost
        u_cost_seq[idx] = inc_u_cost        
    return total_cost, cost_seq, x_cost_seq, u_cost_seq

def taylor_expand_cost(cost_func, x_k:npt.ArrayLike, u_k:npt.ArrayLike, k_step:int, is_final_bool=False):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    xu_k_jax, xu_k_len, x_k_len = prep_xu_vec_for_diff(x_k, u_k)
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(xu_k[:x_k_len], xu_k[x_k_len:], k_step, is_final_bool)
    cost_func_for_diff = prep_cost_func_for_diff(cost_func_xu)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat      = (jax.jacfwd(cost_func_for_diff)(xu_k_jax)).reshape(xu_k_len)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat  = (jax.jacfwd(jax.jacrev(cost_func_for_diff))(xu_k_jax)).reshape(xu_k_len,xu_k_len)
    l_x = np.array((jac_cat[:x_k_len]).reshape(-1,1))
    l_u = np.array((jac_cat[x_k_len:]).reshape(-1,1))
    l_xx = np.array((hessian_cat[:x_k_len,:x_k_len]))
    l_uu = np.array((hessian_cat[x_k_len:,x_k_len:]))
    l_ux = np.array((hessian_cat[x_k_len:,:x_k_len]))
    return l_x, l_u, l_xx, l_uu, l_ux
    
def prep_xu_vec_for_diff(x_k,u_k):
    assert (x_k.shape)[1] == 1, 'x_vector must be column vector (n,1)'
    assert (u_k.shape)[1] == 1, 'u_vector must be column vector (m,1)'
    xu_k         = np.concatenate([x_k, u_k])
    x_k_len      = np.shape(x_k)[0]
    xu_k_len     = np.shape(xu_k)[0]
    xu_k_jax     = jnp.array(xu_k)
    return xu_k_jax, xu_k_len, x_k_len

def ss_2_dyn_func(ss_cont:stateSpace):
    return lambda t, x, u: ss_cont.a @ x + ss_cont.b @ u

def calculate_s_xx_dot(Q_t, A_t, BRinvBT_t, s_xx_t):
    s_xx_dt = -(Q_t - (s_xx_t @ BRinvBT_t @ s_xx_t) + (s_xx_t @ A_t) + (A_t.T @ s_xx_t))
    return s_xx_dt

def prep_cost_func_for_diff(cost_func):
    # Function for reducing generic cost function output to single queried value
    # useful for preparing functions for jax differentiation
    def cost_func_for_diff(vec):
        cost_out = cost_func(vec)
        return cost_out[0]
    return cost_func_for_diff


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