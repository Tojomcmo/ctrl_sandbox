#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
#https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import copy
import numpy as np
from typing import Optional
from scipy.integrate import solve_ivp

import ilqr_utils as util

class ilqrControllerState:
    def __init__(self,ilqr_config, seed_state_vec, seed_contro1_seq, state_des_seq = None, control_des_seq = None):
        # direct parameter population
        self.seed_state_vec        = seed_state_vec
        self.seed_control_seq      = seed_contro1_seq
        self.time_step             = ilqr_config['time_step']
        self.cost_float            = 0.0
        self.prev_cost_float       = 0.0
        self.iter_int              = 0 
        self.ro_reg                = 0.0
        self.ro_reg_change_bool    = False    
        # dependent parameter population
        self.len_seq               = len(seed_contro1_seq)+1
        self.seed_state_seq        = [None] * (self.len_seq)
        self.state_len             = self.get_num_states()
        self.control_len           = self.get_num_controls()
        self.state_seq             = [None] * self.len_seq
        self.control_seq           = seed_contro1_seq
        self.time_seq              = self.create_time_sequence()
        self.state_des_seq         = self.populate_desired_state_seq(state_des_seq)
        self.control_des_seq       = self.populate_desired_control_seq(control_des_seq)
        self.K_seq                 = [None] * (self.len_seq-1)
        self.d_seq                 = [None] * (self.len_seq-1)
        self.Del_V_vec_seq         = [None] * (self.len_seq-1)

    def create_time_sequence(self):
        time_id        = list(range(self.len_seq))
        time_seq       = [i * self.time_step for i in time_id]
        return time_seq

    def populate_desired_state_seq(self, state_des_seq):
        if state_des_seq == None:
            state_des_seq = [jnp.zeros([self.get_num_states(), 1])] * self.len_seq
        else:
            assert(len(state_des_seq) == len(self.state_seq))
        return state_des_seq  

    def populate_desired_control_seq(self, control_des_seq):
        if control_des_seq == None:
            control_des_seq = [jnp.zeros([self.get_num_controls(), 1])] * (self.len_seq-1)
        else:
            assert(len(control_des_seq) == len(self.control_seq))
        return control_des_seq          

    def get_num_states(self):
        return jnp.shape(util.vec_1D_array_to_col(self.seed_state_vec))[0]
    
    def get_num_controls(self):
        return jnp.shape(util.vec_1D_array_to_col(self.seed_control_seq[0]))[0]

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
        state_trans_func_curried = lambda t,x,u=None: state_trans_func(state_trans_func_params,t,x,u)
        return state_trans_func_curried
    
    def create_curried_simulate_dyn_func(self, ilqr_config):
        sim_method       = ilqr_config['sim_method']
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

def initialize_ilqr_controller(ilqr_config, ctrl_state:ilqrControllerState):
    # create object containing preconfigured functions with static parameters (currying and parameter encapsulation)
    config_funcs = ilqrConfiguredFuncs(ilqr_config, ctrl_state)
    # populate state sequence from seed control sequence
    state_seq = config_funcs.simulate_dyn_func(ctrl_state.seed_state_vec,
                                               ctrl_state.seed_control_seq)

    # calculate initial cost
    cost_float = calculate_total_cost(config_funcs.cost_func, state_seq, ctrl_state.control_seq)
    # ensure seed_cost is initialized with larger differential than convergence bound
    prev_cost_float = cost_float + (1 + ilqr_config['converge_crit'])
    # while within iteration limit and while cost has not converged    
    return config_funcs, state_seq, cost_float, prev_cost_float

def run_ilqr_controller(ilqr_config, config_funcs, ctrl_state:ilqrControllerState):
    local_ctrl_state = copy.deepcopy(ctrl_state)
    converge_measure = ilqr_config['converge_crit'] + 1
    converge_reached = False
    while (local_ctrl_state.iter_int < ilqr_config['max_iter']) and (converge_reached is False):
        # retain previous cost value for convergence check
        local_ctrl_state.iter_int += 1
        local_ctrl_state.prev_cost_float = local_ctrl_state.cost_float
        # perform backwards pass to calculate gains and expected value function change
        local_ctrl_state.K_seq, local_ctrl_state.d_seq, local_ctrl_state.Del_V_vec_seq, local_ctrl_state.ro_reg = calculate_backwards_pass(ilqr_config, config_funcs, local_ctrl_state)
        # perform forwards pass to calculate new trajectory and trajectory cost
        local_ctrl_state.state_seq, local_ctrl_state.control_seq, local_ctrl_state.cost_float, local_ctrl_state.ro_reg_change_bool = calculate_forwards_pass(ilqr_config, config_funcs, local_ctrl_state)
        # increment iteration counter
        converge_measure = jnp.abs(local_ctrl_state.prev_cost_float - local_ctrl_state.cost_float)
        print('iteration number: ', local_ctrl_state.iter_int)
        print('converge_measure: ', converge_measure)
        print('ro_reg_value: ', local_ctrl_state.ro_reg)
        print('need to increase ro?: ', local_ctrl_state.ro_reg_change_bool)
        if (converge_measure < ilqr_config['converge_crit']) and (local_ctrl_state.ro_reg_change_bool is False):
            converge_reached = True
            print('controller converged')
        elif local_ctrl_state.iter_int == ilqr_config['max_iter']:  
            print('controller reached iteration limit')
    return local_ctrl_state

def calculate_backwards_pass(ilqr_config, config_funcs:ilqrConfiguredFuncs, ctrl_state:ilqrControllerState):
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
                                                        ctrl_state.state_seq[-1], 
                                                        ctrl_state.control_len)
    
    P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N,
                                                                          ctrl_state.len_seq)
    if ctrl_state.ro_reg_change_bool is True:
        ro_reg = ctrl_state.ro_reg + ilqr_config['ro_reg_change']
    else:
        ro_reg = ilqr_config['ro_reg_start']
    is_valid_q_uu = False
    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq - 1):
            k_step = -(idx + 1)
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = taylor_expand_pseudo_hamiltonian(
                                                                        config_funcs.cost_func,
                                                                        Ad_seq[k_step],
                                                                        Bd_seq[k_step],
                                                                        ctrl_state.state_seq[k_step],
                                                                        ctrl_state.control_seq[k_step],
                                                                        P_kp1,
                                                                        p_kp1,
                                                                        ro_reg,
                                                                        k_step)
            # if q_uu_reg is not positive definite, reset loop to initial state and increase ro_reg
            if util.is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                ro_reg    = ro_reg + ilqr_config['ro_reg_change']
                P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N,
                                                                                      ctrl_state.len_seq)
                break
            else:
                K_k,   d_k                = calculate_optimal_gains(q_uu_reg, q_ux_reg, q_u)
                P_kp1, p_kp1, Del_V_vec_k = calculate_backstep_ctg_approx(q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k)
                k_seq[k_step]         = K_k
                d_seq[k_step]         = d_k
                Del_V_vec_seq[k_step] = Del_V_vec_k
    return k_seq, d_seq, Del_V_vec_seq, ro_reg

def calculate_forwards_pass(ilqr_config, config_funcs:ilqrConfiguredFuncs, ctrl_state:ilqrControllerState):
    #initialize updated state vec, updated control_vec, line search factor
    state_seq_new,control_seq_new,cost_float_new,in_bounds_bool = initialize_forwards_pass(ctrl_state.seed_state_vec,
                                                                                           ctrl_state.len_seq)
    line_search_factor = 1
    line_search_scale_param = 0.5
    iter_count = 0
    max_iter   = ilqr_config['fp_max_iter']
    ro_reg_change_bool = False
    while not in_bounds_bool and (iter_count < max_iter):
        for k_step in range(ctrl_state.len_seq-1):
            # calculate updated control value
            u_k_new = calculate_u_k_new(ctrl_state.control_seq[k_step],
                                        ctrl_state.state_seq[k_step],
                                        state_seq_new[k_step], 
                                        ctrl_state.K_seq[k_step], 
                                        ctrl_state.d_seq[k_step], 
                                        line_search_factor)
            # calculate updated next state with dynamics function
            x_kp1_new = (config_funcs.simulate_dyn_func(state_seq_new[k_step], u_k_new))[-1]
            
            # populate state and control sequences
            control_seq_new[k_step]  = u_k_new
            state_seq_new[k_step+1]  = x_kp1_new 

        # shift sequences for cost calculation wrt desired trajectories
        # calculate new trajectory cost    
        cost_float_new = calculate_total_cost(config_funcs.cost_func, state_seq_new, control_seq_new)
        # calculate the ratio between the expected cost decrease and actual cost decrease
        expected_cost_decrease = calculate_expected_cost_decrease(ctrl_state.Del_V_vec_seq, line_search_factor)
        actual_cost_decrease = cost_float_new - ctrl_state.cost_float
        cost_decrease_ratio = calculate_cost_decrease_ratio(ctrl_state.cost_float, cost_float_new, ctrl_state.Del_V_vec_seq, line_search_factor) 
        in_bounds_bool = config_funcs.analyze_cost_dec_func(cost_decrease_ratio)
        # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
        if in_bounds_bool:
            break
        elif (iter_count == max_iter-1):
            iter_count     += 1  
            state_seq_new   = ctrl_state.state_seq
            control_seq_new = ctrl_state.control_seq
            cost_float_new  = ctrl_state.cost_float
            ro_reg_change_bool = True
        else:
            state_seq_new,control_seq_new,cost_float_new,in_bounds_bool = initialize_forwards_pass(ctrl_state.seed_state_vec,
                                                                                                   ctrl_state.len_seq)
            line_search_factor *= line_search_scale_param
            iter_count += 1
    return state_seq_new, control_seq_new, cost_float_new, ro_reg_change_bool

def simulate_forward_dynamics(dyn_func, state_init, control_seq, time_step, sim_method = 'euler'):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    len_seq        = len(control_seq)
    state_seq      = [None] * (len_seq + 1)
    state_seq[0]   = state_init
    state_len      = jnp.shape(state_init)[0]
    control_len    = jnp.shape(control_seq[0])[0]
    time_seq       = jnp.arange(len_seq) * time_step
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
            state_next       = (jnp.array([result_ivp.y[:,-1]])).reshape(-1,1)
            state_seq[idx+1] = state_next
    else:
        raise ValueError("invalid simulation method")
    return state_seq

def linearize_dynamics(dyn_func,t_primal, x_primal, u_primal):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u
    x_len = jnp.shape(x_primal)[0]
    u_len = jnp.shape(u_primal)[0]
    a_lin = jax.jacfwd(lambda x: dyn_func(t_primal, x       , u_primal))(x_primal).reshape(x_len,x_len)
    b_lin = jax.jacfwd(lambda u: dyn_func(t_primal, x_primal, u       ))(u_primal).reshape(x_len,u_len)
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
        a_d = jnp.zeros(input_state_space.a.shape)
        b_d = jnp.zeros(input_state_space.b.shape)
        c_d = jnp.zeros(input_state_space.c.shape)
        d_d = jnp.zeros(input_state_space.d.shape)
        n  = input_state_space.a.shape[0]
        m  = input_state_space.b.shape[1]
        p  = input_state_space.c.shape[0]

        if c2d_method=='euler' :
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

def initialize_backwards_pass(P_N, p_N, len_seq):
    K_seq         = [None] * (len_seq-1)
    d_seq         = [None] * (len_seq-1)
    Del_V_vec_seq = [None] * (len_seq-1)
    P_kp1 = P_N
    p_kp1 = p_N
    return P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq

def initialize_forwards_pass(seed_state_vec, len_seq):
    control_seq_updated = [None] * (len_seq-1)
    state_seq_updated   = [None] * len_seq
    seed_state_vec.reshape(1,-1)
    state_seq_updated[0] = seed_state_vec
    cost_float_updated = 0
    in_bounds_bool = False
    return state_seq_updated, control_seq_updated, cost_float_updated, in_bounds_bool

def calculate_total_cost(cost_func, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # Calculate total cost
    seq_len    = len(state_seq)
    total_cost = 0
    for idx in range(seq_len):
        if (idx == seq_len-1):
            incremental_cost = cost_func(state_seq[idx], jnp.zeros([1,1]), idx, is_final_bool=True)
        else:
            incremental_cost = cost_func(state_seq[idx], control_seq[idx], idx)
        total_cost = total_cost + incremental_cost 
    return total_cost

def taylor_expand_cost(cost_func, x_k, u_k, k_step):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    # create concatenated state and control vector of primal points
    x_k_col      = x_k.reshape(-1,1)
    x_k_len      = jnp.shape(x_k_col)[0]
    u_k_col      = u_k.reshape(-1,1)
    xu_k         = np.concatenate([x_k_col, u_k_col])
    xu_k_len     = jnp.shape(xu_k)[0]
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_k: cost_func(xu_k[:x_k_len], xu_k[x_k_len:], k_step)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat     = (jax.jacfwd(cost_func_xu)(xu_k)).reshape(xu_k_len)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k)).reshape(xu_k_len,xu_k_len)
    l_x = (jac_cat[:x_k_len]).reshape(-1,1)
    l_u = (jac_cat[x_k_len:]).reshape(-1,1)
    l_xx = hessian_cat[:x_k_len,:x_k_len]
    l_uu = hessian_cat[x_k_len:,x_k_len:]
    l_ux = hessian_cat[x_k_len:,:x_k_len]
    return l_x, l_u, l_xx, l_uu, l_ux

def taylor_expand_pseudo_hamiltonian(cost_func, A_lin_k, B_lin_k, x_k, u_k, P_kp1, p_kp1, ro_reg, k_step):

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
    P_kp1_reg = util.reqularize_mat(P_kp1, ro_reg)

    q_x      = l_x  + A_lin_k_T @ p_kp1
    q_u      = l_u  + B_lin_k_T @ p_kp1
    q_xx     = l_xx + A_lin_k_T @ P_kp1     @ A_lin_k
    q_uu     = l_uu + B_lin_k_T @ P_kp1     @ B_lin_k
    q_ux     = l_ux + B_lin_k_T @ P_kp1     @ A_lin_k
    q_uu_reg = l_uu + B_lin_k_T @ P_kp1_reg @ B_lin_k
    q_ux_reg = l_ux + B_lin_k_T @ P_kp1_reg @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg

def calculate_final_cost_to_go_approximation(cost_func, x_N, len_u):
    # create concatenated state and control vector of primal points
    x_N_col      = x_N.reshape(-1,1)
    x_N_len      = jnp.shape(x_N_col)[0]
    xu_N         = jnp.concatenate([x_N_col, jnp.zeros([len_u,1])])
    xu_N_len     = jnp.shape(xu_N)[0]
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_N: cost_func(xu_N[:x_N_len], xu_N[x_N_len:], -1, is_final_bool=True)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat      = (jax.jacfwd(cost_func_xu)(xu_N)).reshape(xu_N_len)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat  = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_N)).reshape(xu_N_len,xu_N_len)
    l_x          = (jac_cat[:(len(x_N))]).reshape(-1,1)
    l_xx         = hessian_cat[:(len(x_N)),:(len(x_N))]
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
    # TODO verify Del_V_vec_k calculation signs, absolute vs actual, actual gives different signs...
    # Del_V_vec_k = jnp.array([[-jnp.abs((d_kt @ q_u))],[-jnp.abs((0.5) * (d_kt @ q_uu @ d_k))]]).reshape(1,-1)
    Del_V_vec_k = jnp.array([[(d_kt @ q_u)],[((0.5) * (d_kt @ q_uu @ d_k))]]).reshape(1,-1)
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
    return u_k_updated_col

def calculate_cost_decrease_ratio(prev_cost_float, new_cost_float, Del_V_vec_seq, line_search_factor):
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
    cost_decrease_ratio = (1 / -Del_V_sum) * (prev_cost_float - new_cost_float)
    return cost_decrease_ratio

def calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor):
    Del_V_sum = 0
    for idx in range(len(Del_V_vec_seq)):
        Del_V_vec = (Del_V_vec_seq[idx]).reshape(-1)
        Del_V_sum = Del_V_sum + ((line_search_factor * Del_V_vec[0]) + (line_search_factor**2 * Del_V_vec[1]))
    return Del_V_sum    

def analyze_cost_decrease(cost_decrease_ratio, cost_ratio_bounds):
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (cost_decrease_ratio < cost_ratio_bounds[1]):
        in_bounds_bool = True
    else:
        in_bounds_bool = False            
    return in_bounds_bool


