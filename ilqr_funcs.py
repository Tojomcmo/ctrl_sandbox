#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from typing import Optional
#import jax import jacfwd
#from jax.typing import Arraylike
from scipy.integrate import solve_ivp

class ilqrControllerState:
    def __init__(self, seed_state, seed_contro1):
        self.seed_state_vec    = seed_state
        self.seed_state_seq    = []
        self.seed_control_seq  = seed_contro1
        self.seed_cost_float   = 0
        self.len_seq           = len(seed_contro1)
        self.iter_state_seq    = []
        self.iter_control_seq  = []
        self.iter_cost_float   = 0
        self.iter_int          = 0
        self.time_seq          = []
        
    def create_time_sequence(self, time_step):
        self.time_seq = jnp.arange(len(self.seed_control_seq) +1) * time_step

class ilqrConfiguredFuncs:
    def __init__(self, ilqr_config):    
        self.cost_func          = self.create_curried_cost_func(ilqr_config)
        self.state_trans_func   = self.create_curried_state_trans_func(ilqr_config) 
        self.simulate_dyn_func  = self.create_curried_simulate_dyn_func(ilqr_config)
        self.c2d_ss_func           = self.create_curried_c2d_ss_func(ilqr_config)

    def create_curried_cost_func(self, ilqr_config):
        cost_func         = ilqr_config['cost_func']
        cost_func_params  = ilqr_config['cost_func_params']
        cost_func_curried = lambda x, u, is_final_bool=False: cost_func(cost_func_params, x, u, is_final_bool)
        return cost_func_curried 
    
    def create_curried_state_trans_func(self, ilqr_config):
        state_trans_func         = ilqr_config['state_trans_func']
        state_trans_func_params  = ilqr_config['state_trans_func_params']
        state_trans_func_curried = lambda x, u, t=None: state_trans_func(state_trans_func_params, x, u, t)
        return state_trans_func_curried
    
    def create_curried_simulate_dyn_func(self, ilqr_config):
        sim_method  = ilqr_config['cost_func_params']
        time_step   = ilqr_config['time_step']
        sim_func_curried = lambda x_init, u_seq: simulate_forward_dynamics(self.state_trans_func, x_init, u_seq, time_step, sim_method)
        return sim_func_curried 
    
    def create_curried_c2d_ss_func(self, ilqr_config):
        c2d_method = ilqr_config['c2d_method']
        time_step  = ilqr_config['time_step']
        descritize_func_curried  = lambda state_space: discretize_state_space(state_space, time_step, c2d_method):
        return descritize_func_curried


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

def ilqr_controller(ilqr_config, ctrl_state:ilqrControllerState):
    # initialize controller state?
    # calculate initial rollout
    # 
    config_funcs = ilqrConfiguredFuncs(ilqr_config)
    ctrl_state.seed_state_seq = config_funcs.simulate_dyn_func(ctrl_state.seed_state_vec,
                                                               ctrl_state.seed_control_seq)
    
    ctrl_state.seed_cost_float = calculate_total_cost(config_funcs.cost_func, 
                                                    ctrl_state.seed_state_seq, 
                                                    ctrl_state.seed_control_seq)
    prev_cost = 10000

    ctrl_state.iter_control_seq = ctrl_state.seed_control_seq
    ctrl_state.iter_state_seq   = ctrl_state.seed_state_seq

    while iter < ilqr_config['max_iter'] and jnp.abs(ctrl_state.seed_cost_float - ctrl_state.iter_cost_float) > ilqr_config['converge_crit']:
        ctrl_state.seed_cost_float = ctrl_state.iter_cost_float

        Ad_seq, Bd_seq = calculate_linearized_state_space_seq(config_funcs.state_trans_func,
                                                              config_funcs.c2d_ss_func,
                                                              ctrl_state.iter_state_seq,
                                                              ctrl_state.iter_control_seq,
                                                              ctrl_state.time_seq)

        K_seq, d_seq,  = calculate_backwards_pass(config_funcs, ctrl_state)

    return 
def calculate_forward_rollout(state_init, control_seq, time_step, 
                              config_funcs: Optional[ilqrConfiguredFuncs] = None):
    # simulate forward dynamics
    # Linearize dynamics at each time step
    # calculate discretized linearized state space for each time step
    # return
    # **kwargs:
    # - sim_method('Euler', 'solve_ivp_zoh')
    # - c2d_method('Euler', 'zoh', 'zohCombined')
    # state_seq, time_seq  = simulate_forward_dynamics(state_trans_func, state_init, control_seq, time_step)
    state_seq  = config_funcs.simulate_dyn_func(state_init, control_seq)
    total_cost = config_funcs.calculate_total_cost(state_seq, control_seq)
    return state_seq, time_seq, cost_seq, total_cost

def calculate_initial_rollout(state_trans_func, cost_func, state_init, control_seq, time_step):
    state_seq, time_seq, _, _ = calculate_forward_rollout(state_trans_func, cost_func, state_init, control_seq, time_step)
    ad_seq, bd_seq            = calculate_linearized_state_space_seq(state_trans_func, state_seq, control_seq, time_step)
    return state_seq, time_seq, ad_seq, bd_seq

def calculate_backwards_pass(config_funcs:ilqrConfiguredFuncs, ctrl_state:ilqrControllerState):
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
    Ad_seq, Bd_seq = calculate_linearized_state_space_seq(config_funcs.state_trans_func,
                                                        config_funcs.c2d_ss_func,
                                                        ctrl_state.iter_state_seq,
                                                        ctrl_state.iter_control_seq,
                                                        ctrl_state.time_seq)
    
    P_N, p_N = calculate_final_cost_to_go_approximation(config_funcs.cost_func, 
                                                        ctrl_state.iter_state_seq[-1], 
                                                        len(ctrl_state.iter_control_seq[0]))
    
    P_kp1 = P_N
    p_kp1 = p_N 

    K_seq = []
    d_seq = []

    ro = 0.1
    is_valid_q_uu = False

    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq):
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu = taylor_expand_pseudo_hamiltonian(config_funcs.cost_func, 
                                                                    Ad_seq[-idx], Bd_seq[-idx], 
                                                                    ctrl_state.iter_state_seq,[-idx],
                                                                    ctrl_state.iter_control_seq[-idx], 
                                                                    P_kp1, p_kp1)
            q_uu_reg = reqularize_q_uu(q_uu, ro)
            if is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                K_seq = []
                d_seq = []
                P_kp1 = P_N
                p_kp1 = p_N
                ro = ro + 0.1
                break
            else:
                K_k, d_k = calculate_optimal_gains(q_uu_reg, q_ux, q_u)
                K_seq.extend(K_k)
                d_seq.extend(d_k)
                P_kp1, p_kp1 = calculate_ctg_approx()

    return K_seq, d_seq

def calculate_forwards_pass(config_funcs: ilqrConfiguredFuncs, state_init, control_seq, K_gain_seq, d_gain_seq):

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
            dyn_func_zoh = (lambda time_dyn, state: dyn_func(state, control_seq[idx], time_dyn))
            result_ivp   = solve_ivp(dyn_func_zoh, time_span, state_seq[idx])
            state_next   = jnp.array([result_ivp.y[:,-1]])
            state_seq    = jnp.append(state_seq, state_next, axis=0)
    else:
        raise ValueError("invalid simulation method")
    return state_seq

def calculate_total_cost(cost_func, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # calculate each discrete cost
    # Calculate total cost
    total_cost = 0
    return total_cost

def taylor_expand_cost(cost_func, x_k, u_k):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    xu_k         = np.concatenate([x_k, u_k])
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(xu_k[:len(x_k)], xu_k[len(x_k):])
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat     = jax.jacfwd(cost_func_xu)(xu_k)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat = jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k)
    l_x = jac_cat[:len(x_k)]
    l_u = jac_cat[len(x_k):]
    l_xx = hessian_cat[:len(x_k),:len(x_k)]
    l_uu = hessian_cat[len(x_k):,len(x_k):]
    l_ux = hessian_cat[len(x_k):,:len(x_k)]
    return l_x, l_u, l_xx, l_uu, l_ux

def taylor_expand_pseudo_hamiltonian(cost_func, A_lin_k, B_lin_k, x_k, u_k, P_kp1, p_kp1):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'
    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, x_k, u_k)
    A_lin_k_T = jnp.transpose(A_lin_k)
    B_lin_k_T = jnp.transpose(B_lin_k)

    q_x  = l_x  + A_lin_k_T @ p_kp1
    q_u  = l_u  + B_lin_k_T @ p_kp1
    q_xx = l_xx + A_lin_k_T @ P_kp1 @ A_lin_k
    q_uu = l_uu + B_lin_k_T @ P_kp1 @ B_lin_k   
    q_ux = l_ux + B_lin_k_T @ P_kp1 @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu

def calculate_final_cost_to_go_approximation(cost_func, x_N, len_u):
    # create concatenated state and control vector of primal points
    xu_N         = np.concatenate([x_N, jnp.zeros(len_u)])
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_N: cost_func(xu_N[:len(x_N)], xu_N[len(x_N):], is_final_bool=True)
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

def linearize_dynamics(dyn_func, x_primal, u_primal, t_primal):
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

def calculate_linearized_state_space_seq(dyn_func, c2d_ss_func, state_seq, control_seq, time_seq):
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    if len(state_seq.shape) != 2 or len(control_seq.shape) != 2:
        raise ValueError('state or control sequence is incorrect array dimension. sequences must be 2d arrays')
    elif len(state_seq) != len(control_seq)+1:
        raise ValueError('state and control sequences are incompatible lengths. state seq must be control seq length +1')
    else:
        state_dim   = len(state_seq[1,:])
        control_dim = len(control_seq[1,:])
        a_lin_array = jnp.zeros((state_dim, state_dim,1))
        b_lin_array = jnp.zeros((state_dim, control_dim,1))
        c_mat_dummy = jnp.zeros((1,state_dim))
        d_mat_dummy = jnp.zeros((1,control_dim))
        for idx,_ in enumerate(control_seq):
            a_lin, b_lin             = linearize_dynamics(dyn_func, state_seq[idx], control_seq[idx], time_seq[idx])
            ss_pend_lin_continuous   = stateSpace(a_lin, b_lin, c_mat_dummy, d_mat_dummy)
            ss_pend_lin_discrete     = c2d_ss_func(ss_pend_lin_continuous)
            if idx == 0:
                a_lin_array = jnp.array([ss_pend_lin_discrete.a])
                b_lin_array = jnp.array([ss_pend_lin_discrete.b])
            else:
                a_lin_array = jnp.append(a_lin_array, jnp.array([ss_pend_lin_discrete.a]), axis=0)
                b_lin_array = jnp.append(b_lin_array, jnp.array([ss_pend_lin_discrete.b]), axis=0)
    return a_lin_array, b_lin_array

def is_pos_def(x):
    return jnp.all(jnp.linalg.eigvals(x) > 0)

def reqularize_q_uu(q_uu, ro):
    return q_uu + ro * jnp.eye(q_uu.shape[0], q_uu.shape[1])






#-------------- Obsolete functions -------------------#






def calculate_forward_rollout(state_init, control_seq, time_step, 
                              config_funcs: Optional[ilqrConfiguredFuncs] = None):
    # simulate forward dynamics
    # Linearize dynamics at each time step
    # calculate discretized linearized state space for each time step
    # return
    # **kwargs:
    # - sim_method('Euler', 'solve_ivp_zoh')
    # - c2d_method('Euler', 'zoh', 'zohCombined')
    # state_seq, time_seq  = simulate_forward_dynamics(state_trans_func, state_init, control_seq, time_step)
    state_seq  = config_funcs.simulate_dyn_func(state_init, control_seq)
    total_cost = config_funcs.calculate_total_cost(state_seq, control_seq)
    return state_seq, time_seq, cost_seq, total_cost

def calculate_initial_rollout(state_trans_func, cost_func, state_init, control_seq, time_step):
    state_seq, time_seq, _, _ = calculate_forward_rollout(state_trans_func, cost_func, state_init, control_seq, time_step)
    ad_seq, bd_seq            = calculate_linearized_state_space_seq(state_trans_func, state_seq, control_seq, time_step)
    return state_seq, time_seq, ad_seq, bd_seq