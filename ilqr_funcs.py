#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
#https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from typing import Optional
from scipy.integrate import solve_ivp

import ilqr_utils as util

class ilqrControllerState:
    def __init__(self, seed_state_vec, seed_contro1_seq, time_step, state_des_seq = None, control_des_seq = None):
        self.seed_state_vec        = seed_state_vec
        self.seed_control_seq      = seed_contro1_seq
        self.len_seq               = len(seed_contro1_seq)
        if state_des_seq == None:
            self.state_des_seq     = jnp.zeros([self.len_seq, self.num_states()])
        else:
            self.state_des_seq     = state_des_seq   
        if control_des_seq == None:
            self.control_des_seq   = jnp.zeros([self.len_seq, self.num_controls()])
        else:
            self.control_des_seq   = control_des_seq 
        self.state_seq             = []
        self.control_seq           = seed_contro1_seq
        self.des_state_diff_seq    = []
        self.des_control_diff_seq  = []
        self.state_seq_minus_des   = []
        self.control_seq_minus_des = []
        self.cost_float            = 0.0
        self.prev_cost_float       = 0.0
        self.K_seq                 = []
        self.d_seq                 = []
        self.Del_V_vec_seq         = []
        self.iter_int              = 0
        self.time_step             = time_step
        self.time_seq              = jnp.arange(len(self.seed_control_seq) +1) * time_step
        
    def update_time_sequence(self, new_time_step):
        self.time_step = new_time_step
        self.time_seq  = jnp.arange(len(self.seed_control_seq) +1) * new_time_step

    def update_des_traj_diff_seq(self):  
            self.des_state_traj_diff_seq   = self.state_des_seq - self.state_seq
            self.des_control_traj_diff_seq = self.control_des_seq - self.control_seq

    def num_states(self):
        return len(self.seed_state_vec)
    
    def num_controls(self):
        return len(self.seed_control_seq[0])

class ilqrConfiguredFuncs:
    def __init__(self, ilqr_config, controller_state:ilqrControllerState):
        self.cost_func               = self.create_curried_cost_func(ilqr_config, controller_state)
        self.state_trans_func        = self.create_curried_state_trans_func(ilqr_config)
        self.simulate_dyn_func       = self.create_curried_simulate_dyn_func(ilqr_config)
        self.c2d_ss_func             = self.create_curried_c2d_ss_func(ilqr_config)
        self.analyze_cost_dec_func   = self.create_curried_analyze_cost_dec_func(ilqr_config)

    def create_curried_cost_func(self, ilqr_config, controller_state):
        cost_func         = ilqr_config['cost_func']
        cost_func_params  = ilqr_config['cost_func_params']
        state_des_seq     = controller_state.state_des_seq
        control_des_seq   = controller_state.control_des_seq
        cost_func_curried = lambda x, u, k, is_final_bool=False: cost_func(cost_func_params, x, u, k, state_des_seq, control_des_seq, is_final_bool)
        return cost_func_curried 
    
    def create_curried_state_trans_func(self, ilqr_config):
        state_trans_func         = ilqr_config['state_trans_func']
        state_trans_func_params  = ilqr_config['state_trans_func_params']
        state_trans_func_curried = lambda t,x,u=None: state_trans_func(t,x,u,state_trans_func_params)
        return state_trans_func_curried
    
    def create_curried_simulate_dyn_func(self, ilqr_config):
        sim_method       = ilqr_config['cost_func_params']
        time_step        = ilqr_config['time_step']
        sim_func_curried = lambda x_init, u_seq: simulate_forward_dynamics(self.state_trans_func, x_init, u_seq, time_step, sim_method)
        return sim_func_curried 
    
    def create_curried_c2d_ss_func(self, ilqr_config):
        c2d_method              = ilqr_config['c2d_method']
        time_step               = ilqr_config['time_step']
        descritize_func_curried = lambda state_space: discretize_state_space(state_space, time_step, c2d_method)
        return descritize_func_curried
    
    def create_curried_analyze_cost_dec_func(self, ilqr_config):
        cost_ratio_bounds = ilqr_config['cost_ratio_bounds']
        analyze_cost_dec_func_curried  = lambda cost_decrease_ratio: analyze_cost_decrease(cost_decrease_ratio, cost_ratio_bounds)
        return analyze_cost_dec_func_curried
    
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

    # create object containing preconfigured functions with static parameters (currying and parameter encapsulation)
    config_funcs = ilqrConfiguredFuncs(ilqr_config, ctrl_state)
    # populate state sequence from seed control sequence
    ctrl_state.state_seq = config_funcs.simulate_dyn_func(ctrl_state.seed_state_vec,
                                                          ctrl_state.seed_control_seq)

    # calculate initial cost
    ctrl_state.cost_float = calculate_total_cost(config_funcs.cost_func, ctrl_state.state_seq, ctrl_state.control_seq)
    # ensure seed_cost is initialized with larger differential than convergence bound
    ctrl_state.prev_cost_float = ctrl_state.cost_float + (1 + ilqr_config['converge_crit'])
    # while within iteration limit and while cost has not converged
    while (ctrl_state.iter_int < ilqr_config['max_iter']) and (jnp.abs(ctrl_state.prev_cost_float - ctrl_state.cost_float) > ilqr_config['converge_crit']):
        # retain previous cost value for convergence check
        ctrl_state.prev_cost_float = ctrl_state.cost_float
        # perform backwards pass to calculate gains and expected value function change
        ctrl_state.K_seq,     ctrl_state.d_seq,       ctrl_state.Del_V_vec_seq  = calculate_backwards_pass(config_funcs, ctrl_state)
        # perform forwards pass to calculate new trajectory and trajectory cost
        ctrl_state.state_seq, ctrl_state.control_seq, ctrl_state.cost_float     = calculate_forwards_pass(config_funcs, ctrl_state)
        # increment iteration counter
        ctrl_state.iter_int += ctrl_state.iter_int
    return ctrl_state

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

    Ad_seq, Bd_seq = calculate_linearized_state_space_seq(config_funcs.state_trans_func,
                                                        config_funcs.c2d_ss_func,
                                                        ctrl_state.state_seq,
                                                        ctrl_state.control_seq,
                                                        ctrl_state.time_seq)
    P_N, p_N = calculate_final_cost_to_go_approximation(config_funcs.cost_func, 
                                                        ctrl_state.state_seq_minus_des[-1], 
                                                        len(ctrl_state.control_seq[0]))
    
    P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N)
    reg_factor = 0
    is_valid_q_uu = False
    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq - 1):
            k_step = -(idx + 1)
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu = taylor_expand_pseudo_hamiltonian(config_funcs.cost_func,
                                                                        Ad_seq[k_step],
                                                                        Bd_seq[k_step],
                                                                        ctrl_state.state_seq_minus_des[k_step],
                                                                        ctrl_state.control_seq_minus_des[k_step],
                                                                        P_kp1,
                                                                        p_kp1,
                                                                        k_step)
            q_uu_reg = util.reqularize_mat(q_uu, reg_factor)
            # if q_uu_reg is not positive definite, reset loop to initial state and increase ro_reg
            if util.is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                reg_factor    = reg_factor + 0.1
                P_kp1, p_kp1, k_seq, d_seq, Del_V_seq = initialize_backwards_pass(P_N, p_N)
                break
            else:
                K_k,   d_k                = calculate_optimal_gains(q_uu_reg, q_ux, q_u)
                P_kp1, p_kp1, Del_V_vec_k = calculate_backstep_ctg_approx(q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k)
                k_seq.extend(K_k)
                d_seq.extend(d_k)
                Del_V_vec_seq.extend(Del_V_vec_k)

    return k_seq, d_seq, Del_V_vec_seq

def calculate_forwards_pass(config_funcs:ilqrConfiguredFuncs, ctrl_state:ilqrControllerState):
    #initialize updated state vec, updated control_vec, line search factor
    state_seq_new,control_seq_new,cost_float_new,in_bounds_bool = initialize_forwards_pass(ctrl_state.seed_state_vec)
    line_search_factor = 1
    line_search_scale_param = 0.5
    while not in_bounds_bool:
        for k_step in range(ctrl_state.len_seq-1):
            # calculate updated control value
            u_k_new = calculate_u_k_new(ctrl_state.control_seq[k_step],
                                        ctrl_state.state_seq[k_step],
                                        state_seq_new[k_step], 
                                        ctrl_state.K_seq[k_step], 
                                        ctrl_state.d_seq[k_step], 
                                        line_search_factor)
            # calculate updated next state with dynamics function
            x_kp1_new = config_funcs.state_trans_func(ctrl_state.time_seq[k_step], state_seq_new[k_step], u_k_new)
            # populate state and control sequences
            control_seq_new.append(u_k_new)
            state_seq_new.append(x_kp1_new)
        # shift sequences for cost calculation wrt desired trajectories
        # calculate new trajectory cost    
        cost_float_new = calculate_total_cost(config_funcs.cost_func, state_seq_new, control_seq_new)
        # calculate the ratio between the expected cost decrease and actual cost decrease
        cost_decrease_ratio = calculate_cost_decrease_ratio(ctrl_state.cost_float, cost_float_new, ctrl_state.Del_V_vec_seq, line_search_factor) 
        in_bounds_bool = config_funcs.analyze_cost_dec_func(cost_decrease_ratio)
        # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
        if not in_bounds_bool:
            initialize_forwards_pass(ctrl_state.seed_state_vec)
            line_search_factor = line_search_factor * line_search_scale_param
    return state_seq_new, control_seq_new, cost_float_new

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
            state_dot_row = dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next    = jnp.array([state_seq[idx] + state_dot_row * time_step])
            state_seq     = jnp.append(state_seq, state_next, axis=0)
    elif sim_method == 'solve_ivp_zoh':
        for idx in range(int_iter):
            dyn_func_zoh = (lambda time_dyn, state: dyn_func(time_dyn, state, control_seq[idx]))
            result_ivp   = solve_ivp(dyn_func_zoh, time_span, state_seq[idx])
            state_next   = jnp.array([result_ivp.y[:,-1]])
            state_seq    = jnp.append(state_seq, state_next, axis=0)
    else:
        raise ValueError("invalid simulation method")
    return state_seq

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

def initialize_backwards_pass(P_N, p_N):
    K_seq=[]
    d_seq=[]
    Del_V_vec_seq=[]
    P_kp1 = P_N
    p_kp1 = p_N
    return P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq

def initialize_forwards_pass(seed_state_vec):
    control_seq_updated = []
    state_seq_updated = []
    cost_float_updated = 0
    state_seq_updated.append(seed_state_vec)
    in_bounds_bool = False
    return state_seq_updated, control_seq_updated, cost_float_updated, in_bounds_bool

def calculate_total_cost(cost_func, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # Calculate total cost
    seq_len    = len(control_seq)
    total_cost = 0
    for idx in range(seq_len):
        if (idx == seq_len-1):
            incremental_cost = cost_func(state_seq[idx], jnp.zeros(len(control_seq[0])), idx, is_final_bool=True)
        incremental_cost = cost_func(state_seq[idx], control_seq[idx], idx)
        total_cost = total_cost + incremental_cost
    return total_cost

def taylor_expand_cost(cost_func, x_k, u_k, k_step):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    xu_k         = np.concatenate([x_k, u_k])
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(xu_k[:len(x_k)], xu_k[len(x_k):], k_step)
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

def taylor_expand_pseudo_hamiltonian(cost_func, A_lin_k, B_lin_k, x_k, u_k, P_kp1, p_kp1, k_step):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'
    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, x_k, u_k, k_step)
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
    cost_func_xu = lambda xu_N: cost_func(xu_N[:len(x_N)], xu_N[len(x_N):], k_step=-1, is_final_bool=True)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat      = jax.jacfwd(cost_func_xu)(xu_N)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat  = jax.jacfwd(jax.jacrev(cost_func_xu))(xu_N)
    l_x          = jac_cat[:len(x_N)]
    l_xx         = hessian_cat[:len(x_N),:len(x_N)]
    P_N = l_xx
    p_N = l_x
    return P_N, p_N

def calculate_backstep_ctg_approx(q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k):
    # P_k is the kth hessian approximation of the cost-to-go
    # p_k is the kth jacobian approximation of the cost-to-go
    # Del_V_k is the kth expected change in the value function
    K_kt        = K_k.transpose()
    d_kt        = d_k.transpose()
    q_xu        = q_ux.transpose()
    P_k         = (q_xx) + (K_kt @ q_uu @ K_k) + (K_kt @ q_ux) + (q_xu @ K_k)
    p_k         = (q_x ) + (K_kt @ q_uu @ d_k) + (K_kt @ q_u ) + (q_xu @ d_k)
    Del_V_vec_k = jnp.array([[(d_kt @ q_u)],[(0.5) * (d_kt @ q_uu @ d_k)]])
    return P_k, p_k, Del_V_vec_k

def calculate_optimal_gains(q_uu_reg, q_ux, q_u):
    # K_k is a linear feedback term related to del_x from nominal trajectory
    # d_k is a feedforward term related to trajectory correction
    inverted_q_uu = jnp.linalg.inv(q_uu_reg)
    K_k           = -inverted_q_uu @ q_ux
    d_k           = -inverted_q_uu @ q_u
    return K_k, d_k

def calculate_u_k_new(u_nom_k ,x_nom_k ,x_new_k, K_k, d_k, line_search_factor):
    u_nom_k_col     = util.vec_1D_array_to_col(u_nom_k)
    x_nom_k_col     = util.vec_1D_array_to_col(x_nom_k)
    x_new_k_col     = util.vec_1D_array_to_col(x_new_k)
    u_k_updated_col = u_nom_k_col + K_k @ (x_new_k_col - x_nom_k_col) + line_search_factor * d_k
    u_k_updated_row = util.vec_1D_array_to_row(u_k_updated_col)
    print(K_k)
    print(u_k_updated_row)
    return u_k_updated_row

def calculate_cost_decrease_ratio(prev_cost_float, cost_float_updated, Del_V_vec_seq, line_search_factor):
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
    cost_decrease_ratio = (1 / Del_V_sum) * (prev_cost_float - cost_float_updated)
    return cost_decrease_ratio

def calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor):
    Del_V_sum = 0
    for idx in range(len(Del_V_vec_seq)):
        Del_V_sum = Del_V_sum + ((line_search_factor * Del_V_vec_seq[idx][0]) + (line_search_factor**2 * Del_V_vec_seq[idx][1]))
    return Del_V_sum    

def analyze_cost_decrease(cost_decrease_ratio, cost_ratio_bounds):
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (cost_decrease_ratio < cost_ratio_bounds[1]):
        in_bounds_bool = True
    else:
        in_bounds_bool = False            
    return in_bounds_bool


    x_vec = util.vec_1D_array_to_col(x)
    u_vec = util.vec_1D_array_to_col(u)
    return x_vec, u_vec