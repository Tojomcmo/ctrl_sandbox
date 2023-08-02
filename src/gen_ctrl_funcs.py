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

def simulate_forward_dynamics(dyn_func, state_init, control_seq, time_step, sim_method = 'euler'):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    len_seq        = len(control_seq)
    state_len      = len(state_init)
    control_len    = len(control_seq[0])    
    state_seq      = np.zeros([(len_seq+1),state_len, 1])
    state_seq[0]   = state_init
    time_seq       = np.arange(len_seq) * time_step
    if sim_method == 'euler':
        for idx in range(len_seq):
            state_dot_row    = dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next       = state_seq[idx] + state_dot_row * time_step
            state_seq[idx+1] = state_next
    elif sim_method == 'solve_ivp_zoh':
        time_span = (0,time_step)
        for idx in range(len_seq):
            dyn_func_zoh = (lambda time_dyn, state: dyn_func(time_dyn, state, control_seq[idx].reshape(-1)))
            y_0 = (state_seq[idx]).reshape(-1)
            result_ivp       = solve_ivp(dyn_func_zoh, time_span, y_0)
            state_next       = (np.array([result_ivp.y[:,-1]])).reshape(-1,1)
            state_seq[idx+1] = state_next
    else:
        raise ValueError("invalid simulation method")
    return state_seq

def linearize_dynamics(dyn_func,t_k, x_k, u_k):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u
    xu_k_jax, xu_k_len, x_k_len = prep_xu_vec_for_diff(x_k, u_k)
    dyn_func_xu = lambda xu_k: dyn_func(t_k, xu_k[:x_k_len], xu_k[x_k_len:])
    ab_lin_cat  = jax.jacfwd(dyn_func_xu)(xu_k_jax[:,0])
    a_lin       = np.array(ab_lin_cat[:x_k_len,:x_k_len])
    b_lin       = np.array(ab_lin_cat[:x_k_len,x_k_len:])
    return a_lin, b_lin

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
            a_lin, b_lin             = linearize_dynamics(dyn_func,time_seq[idx], state_seq[idx], control_seq[idx])
            ss_pend_lin_continuous   = stateSpace(a_lin, b_lin, c_mat_dummy, d_mat_dummy)
            ss_pend_lin_discrete     = c2d_ss_func(ss_pend_lin_continuous)
            a_lin_array[idx]         = ss_pend_lin_discrete.a
            b_lin_array[idx]         = ss_pend_lin_discrete.b
    return a_lin_array, b_lin_array

def discretize_state_space(input_state_space:stateSpace, time_step:float, c2d_method='Euler'):
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
            a_b_c  = np.concatenate((jnp.concatenate((input_state_space.a, input_state_space.b),axis=1),jnp.zeros((m,n + m))), axis=0)
            a_b_d  = scipy.linalg.expm(a_b_c * time_step)
            a_d   = a_b_d[:n,:n]
            b_d   = a_b_d[:n,n:]
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
    total_cost = 0
    for idx in range(len_seq):
        if (idx == len_seq-1):
            incremental_cost = np.array(cost_func(state_seq[idx], np.zeros([u_len,1]), idx, is_final_bool=True))[0][0]
        else:
            incremental_cost = np.array(cost_func(state_seq[idx], control_seq[idx], idx))[0][0]
        total_cost += incremental_cost 
        cost_seq[idx] = incremental_cost
    return total_cost, cost_seq

def taylor_expand_cost(cost_func, x_k:float, u_k:float, k_step):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    xu_k_jax, xu_k_len, x_k_len = prep_xu_vec_for_diff(x_k, u_k)
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(xu_k[:x_k_len], xu_k[x_k_len:], k_step)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat      = (jax.jacfwd(cost_func_xu)(xu_k_jax)).reshape(xu_k_len)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat  = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k_jax)).reshape(xu_k_len,xu_k_len)
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

