#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
#import jax import jacfwd
#from jax.typing import Arraylike
from scipy.integrate import solve_ivp

class ilqrControllerState:
    def __init__(self, seed_state = None, seed_contro1 = None, iter_int = 0):
        self.seed_state_vec    = seed_state
        self.seed_control_arr  = seed_contro1
        self.seed_cost_float   = 0
        self.iter_state_arr    = None
        self.iter_control_arr  = None
        self.iter_int          = iter_int


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

def calculate_forward_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step):
    # simulate forward dynamics
    # Linearize dynamics at each time step
    # calculate discretized linearized state space for each time step
    # return
    # **kwargs:
    # - sim_method('Euler', 'solve_ivp_zoh')
    # - c2d_method('Euler', 'zoh', 'zohCombined')
    state_seq, time_seq  = simulate_forward_dynamics(dyn_func_with_params, state_init, control_seq, time_step)
    total_cost, cost_seq = calculate_cost_seq_and_total_cost(Q, R, Qf, state_seq, control_seq)
    return state_seq, time_seq, cost_seq, total_cost

def calculate_initial_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step):
    state_seq, time_seq, _, _ = calculate_forward_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step)
    ad_seq, bd_seq            = calculate_linearized_state_space_seq(dyn_func_with_params, state_seq, control_seq, time_step)
    return ad_seq, bd_seq, time_seq

def calculate_backwards_pass(ilqr_config, state_seq, control_seq, time_seq, ad_seq, bd_seq):
#   du_optimal = argmin over delu of Q(delx,delu) = k + (K * delx)
#   k = - inv(q_uu) * q_u
#   K = - inv(q_uu) * q_ux
#   V_N = l(x_N)
#   Vxx = q_xx -       transpose(K) * q_uu * K
#   Vx  = q_x  -       transpose(K) * q_uu * k
#   DV  =      - 1/2 * transpose(k) * q_uu * k

#   Requires Cost function defined for final and intermediate steps (either quadratic Q, Qf >= 0 and R > 0, or function that is taylor expandable to 2nd order pos def?)
#   Requires paired vectors of state and control, along with associated linearized dynamics at each step (x_lin_k, u_lin_k, Ad_k, Bd_k)
#   Requires paired vectors of desired state and control (x_des, u_des), used to create the relative residuals for each time step:
#       -- x_bar   = x       -  x_lin_k
#       -- x_Del_k = x_des_k -  x_lin_k
#       (x - x_des_k) = (x_bar - x_del_k)   : Residual between the current state and the desired state equals the relative vector create by the difference between
#                                             the local linearized coordinate and the local description of the state residual
#   Form of minimum cost is quadratic: Jstar(x,t) = x^T @ Sxx(t) @ x + 2 * x^T @ sx(t) + s0(t)
#   Form of optimal control is u_star_k = u_des_k  - inv(R) @ B^T @[Sxx_k @ x_bar_k + sx_k] where x_bar_k = x_k - x_lin_k

#   Can formulate the Value function V_k = min_u{cost_func(x_k, u_k) + V_k+1(f(x_k,u_k))}
#   

    unew = 0
    v = 0
    print("this is the backward pass 2")
    return unew, v


def simulate_forward_dynamics(dyn_func, state_init, control_seq, time_step, sim_method = 'Euler'):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    int_iter       = len(control_seq)
    state_seq      = state_init
    time_seq       = jnp.arange(int_iter + 1) * time_step
    time_span      = (0, time_step)
    if sim_method == 'Euler':
        for idx in range(int_iter):
            state_dot  = dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next = jnp.array([state_seq[idx] + state_dot * time_step])
            state_seq  = jnp.append(state_seq, state_next, axis=0)
    elif sim_method == 'solve_ivp_zoh':
        for idx in range(int_iter):
            dyn_func_zoh = (lambda time_dyn, state: dyn_func(time_dyn, state, control_seq[idx]))
            result_ivp   = solve_ivp(dyn_func_zoh, time_span, state_seq[idx])
            state_next   = jnp.array([result_ivp.y[:,-1]])
            state_seq    = jnp.append(state_seq, state_next, axis=0)
    else:
        raise ValueError("invalid simulation method")
    return state_seq, time_seq

def calculate_cost_seq_and_total_cost(Q, R, Qf, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # calculate each discrete cost
    # Calculate total cost

    total_cost = 0
    cost_seq = 0
    return total_cost, cost_seq

def taylor_expand_cost(cost_func, cost_func_params, x_k, u_k):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    xu_k         = np.concatenate([x_k, u_k])
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(cost_func_params, xu_k[:len(x_k)], xu_k[len(x_k):])
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat     = jax.jacfwd(cost_func_xu)(xu_k)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat = jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k)
    l_x = jac_cat[:len(x_k)]
    l_u = jac_cat[len(x_k):]
    l_xx = hessian_cat[:len(x_k),:len(x_k)]
    l_uu = hessian_cat[len(x_k):,len(x_k):]
    l_ux = hessian_cat[len(x_k):,:len(x_k)]
    # print('jac_cat: '       , jac_cat)
    # print('hessian_cat: '   , hessian_cat)
    # print('l_x: '           , l_x)
    # print('l_x check: ', cost_func_params['Q'] @ x_k)
    # print('l_u: '           , l_u)
    # print('l_xx: '          , l_xx)
    # print('l_uu: '          , l_uu)
    # print('l_ux: '          , l_ux)
    return l_x, l_u, l_xx, l_uu, l_ux

def taylor_expand_pseudo_hamiltonian(cost_func, cost_func_params, A_lin_k, B_lin_k, x_k, u_k, P_kp1, p_kp1):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'

    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, cost_func_params, x_k, u_k)
    A_lin_k_T = jnp.transpose(A_lin_k)
    B_lin_k_T = jnp.transpose(B_lin_k)

    q_x  = l_x  + A_lin_k_T @ p_kp1
    q_u  = l_u  + B_lin_k_T @ p_kp1
    q_xx = l_xx + A_lin_k_T @ P_kp1 @ A_lin_k
    q_uu = l_uu + B_lin_k_T @ P_kp1 @ B_lin_k   
    q_ux = l_ux + B_lin_k_T @ P_kp1 @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu

def calculate_final_cost_to_go_approximation(cost_func, cost_func_params, x_N, len_u):
    # create concatenated state and control vector of primal points
    xu_N         = np.concatenate([x_N, jnp.zeros(len_u)])
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_N: cost_func(cost_func_params, xu_N[:len(x_N)], xu_N[len(x_N):], is_final_bool=True)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat     = jax.jacfwd(cost_func_xu)(xu_N)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat = jax.jacfwd(jax.jacrev(cost_func_xu))(xu_N)
    l_x = jac_cat[:len(x_N)]
    l_xx = hessian_cat[:len(x_N),:len(x_N)]
    print(hessian_cat)
    P_N = l_xx
    p_N = l_x
    return P_N, p_N

def linearize_dynamics(dyn_func, x_primal, u_primal, t_primal='None'):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u
    a_lin = jax.jacfwd(lambda x: dyn_func(t_primal, x       , u_primal))(x_primal)
    b_lin = jax.jacfwd(lambda u: dyn_func(t_primal, x_primal, u       ))(u_primal)
    return a_lin, b_lin

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
        a_d = jnp.zeros(input_state_space.a.shape)
        b_d = jnp.zeros(input_state_space.b.shape)
        c_d = jnp.zeros(input_state_space.c.shape)
        d_d = jnp.zeros(input_state_space.d.shape)
        n  = input_state_space.a.shape[0]
        m  = input_state_space.b.shape[1]
        p  = input_state_space.c.shape[0]

        if c2d_method=='Euler' :
            a_d = jnp.eye(n) + (input_state_space.a * time_step)
            b_d = input_state_space.b * time_step
            c_d = input_state_space.c
            d_d = input_state_space.d

        elif c2d_method=='zoh' :
            if jscipy.linalg.det(input_state_space.a) > 10E-8:
                a_d = jscipy.linalg.expm(input_state_space.a * time_step)
                b_d = jnp.linalg.inv(input_state_space.a) @ (a_d - jnp.eye(n)) @ input_state_space.b
                c_d = input_state_space.c
                d_d = input_state_space.d
            else:
                raise ValueError('determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid')

        elif c2d_method=='zohCombined':
        #   create combined A B matrix e^([[A, B],[0,0]]
            a_b_c  = jnp.concatenate((jnp.concatenate((input_state_space.a, input_state_space.b),axis=1),jnp.zeros((m,n + m))), axis=0)
            a_b_d  = jscipy.linalg.expm(a_b_c * time_step)
            a_d   = a_b_d[:n,:n]
            b_d   = a_b_d[:n,n:]
            c_d   = input_state_space.c
            d_d   = input_state_space.d
        else:
            raise ValueError('invalid discretization method')
    d_state_space = stateSpace(a_d, b_d, c_d, d_d, time_step)
    return d_state_space

def calculate_linearized_state_space_seq(dyn_func_with_params, state_seq, control_seq, time_step):
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    if len(state_seq.shape) != 2 or len(control_seq.shape) != 2:
        raise ValueError('state or control sequence is incorrect array dimension. sequences must be 2d arrays')
    elif len(state_seq) != len(control_seq)+1:
        raise ValueError('state and control sequences are incompatible lengths. state seq must be control seq length +1')
    elif not isinstance(time_step, float) or not time_step > 0:
        raise ValueError('time_step is invalid. Must be positive float')
    else:
        state_dim   = len(state_seq[1,:])
        control_dim = len(control_seq[1,:])
        a_lin_array = jnp.zeros((state_dim, state_dim,1))
        b_lin_array = jnp.zeros((state_dim, control_dim,1))
        c_mat_dummy = jnp.zeros((1,state_dim))
        d_mat_dummy = jnp.zeros((1,control_dim))
        for idx,_ in enumerate(control_seq):
            a_lin, b_lin             = linearize_dynamics(dyn_func_with_params, state_seq[idx], control_seq[idx])
            ss_pend_lin_continuous   = stateSpace(a_lin, b_lin, c_mat_dummy, d_mat_dummy)
            ss_pend_lin_discrete     = discretize_state_space(ss_pend_lin_continuous, time_step)
            if idx == 0:
                a_lin_array = jnp.array([ss_pend_lin_discrete.a])
                b_lin_array = jnp.array([ss_pend_lin_discrete.b])
            else:
                a_lin_array = jnp.append(a_lin_array, jnp.array([ss_pend_lin_discrete.a]), axis=0)
                b_lin_array = jnp.append(b_lin_array, jnp.array([ss_pend_lin_discrete.b]), axis=0)
    return a_lin_array, b_lin_array