#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
#https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import copy
import numpy as np
from numpy import typing as npt
from typing import Optional
import scipy 
from scipy.integrate import solve_ivp

from . import ilqr_utils as util
from . import gen_ctrl_funcs as gen_ctrl

class ilqrControllerState:
    def __init__(self,ilqr_config, seed_x_vec:npt.ArrayLike, seed_u_seq:npt.ArrayLike, x_des_seq:npt.ArrayLike = np.zeros([1,1]), u_des_seq:npt.ArrayLike = np.zeros([1,1])):
        # direct parameter population
        seed_x_des_seq, seed_u_des_seq = self.populate_des_seqs(x_des_seq, u_des_seq, seed_x_vec, seed_u_seq)
        self.validate_input_data(ilqr_config, seed_x_vec, seed_u_seq, seed_x_des_seq, seed_u_des_seq)          
        self.seed_x_vec         = seed_x_vec     
        self.seed_u_seq         = seed_u_seq     
        self.time_step          = ilqr_config['time_step']
        self.cost_float         = 0.0
        self.prev_cost_float    = 0.0
        self.iter_int           = 0 
        self.ff_gain_avg        = 0.0        
        self.ro_reg             = 0.0
        self.lsf_float          = 1.0
        self.ro_reg_change_bool = False    
        
        # dependent parameter population
        self.u_seq         = seed_u_seq        
        self.len_seq       = self.get_len_seq()
        self.x_len         = self.get_num_states()
        self.u_len         = self.get_num_controls()
        self.seed_x_seq    = np.zeros([self.len_seq,self.x_len,1])        
        self.x_seq         = np.zeros([self.len_seq,self.x_len,1])
        self.time_seq      = np.arange(self.len_seq) * self.time_step
        self.x_des_seq     = seed_x_des_seq
        self.u_des_seq     = seed_u_des_seq
        self.K_seq         = np.zeros([self.len_seq-1, self.u_len, self.x_len])
        self.d_seq         = np.zeros([self.len_seq-1, self.u_len,          1])
        self.Del_V_vec_seq = np.zeros([self.len_seq-1, 2])
        self.cost_seq      = np.zeros([self.len_seq])
        if ilqr_config['log_ctrl_history'] is True:
            self.u_seq_history = []
            self.x_seq_history = []
            self.K_seq_history = []
            self.d_seq_history = []
            self.Del_V_vec_seq_history = []
            self.cost_seq_history = []
            self.cost_float_history = []
            self.lsf_float_history = []

    def validate_input_data(self,ilqr_config, seed_x_vec:npt.ArrayLike, seed_u_seq:npt.ArrayLike, x_des_seq:npt.ArrayLike, u_des_seq:npt.ArrayLike):
        if seed_x_vec.ndim != 2 or (seed_x_vec.shape)[1] != 1: # type: ignore
            raise ValueError('seed state vector has invalid dimension, must be float array of (1,n,1)')
        if seed_u_seq.ndim != 3 or (seed_u_seq.shape)[2] != 1: # type: ignore
            raise ValueError('seed control vector has invalid dimension, must be float array of (len_seq-1,m,1)')
        if u_des_seq.tolist() != (np.zeros([1,1])).tolist():  # type: ignore
                if  u_des_seq.shape != seed_u_seq.shape: # type: ignore
                    raise ValueError('desired control vector has invalid dimension, must match seed control sequence (len_seq-1,m,1)')
        if x_des_seq.tolist() != (np.zeros([1,1])).tolist():  # type: ignore
                if  x_des_seq.shape != (len(seed_u_seq)+1, len(seed_x_vec), 1): # type: ignore
                    raise ValueError('desired state vector has invalid dimension, must match seed control sequence length and state vector length (len_seq,n,1)')

    def create_time_sequence(self):
        time_id        = list(range(self.len_seq))
        time_seq       = [i * self.time_step for i in time_id]
        return time_seq

    def populate_des_seqs(self, x_des_seq, u_des_seq, x_vec, u_vec):
        # if the desired x and u are initialized to zeros 1 x 1, 
        # the sequences are filled to fit the input x and u arrays.
        if x_des_seq.tolist() == (np.zeros([1,1])).tolist():
            x_des_seq = np.zeros([len(u_vec[0])+1,len(x_vec[0]), 1])
        if u_des_seq.tolist() == (np.zeros([1,1])).tolist():
            u_des_seq = np.zeros_like(u_vec)           
        return x_des_seq, u_des_seq

    def get_len_seq(self):
        return len(self.seed_u_seq)+1 # type: ignore
    
    def get_num_states(self):
        return len(self.seed_x_vec) # type: ignore
    
    def get_num_controls(self):
        return len(self.seed_u_seq[0]) # type: ignore
    
    def append_controller_seqs(self):
        self.u_seq_history.append(self.u_seq)
        self.x_seq_history.append(self.x_seq)
        self.cost_seq_history.append(self.cost_seq) 
        self.cost_float_history.append(self.cost_float)
        self.lsf_float_history.append(self)
        self.K_seq_history.append(self.K_seq) 
        self.d_seq_history.append(self.d_seq)
        self.Del_V_vec_seq_history.append(self.Del_V_vec_seq)   


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
        state_des_seq     = controller_state.x_des_seq
        control_des_seq   = controller_state.u_des_seq
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
        sim_func_curried = lambda x_init, u_seq: gen_ctrl.simulate_forward_dynamics_seq(self.state_trans_func, x_init, u_seq, time_step, sim_method)
        return sim_func_curried 
    
    def create_curried_c2d_ss_func(self, ilqr_config):
        c2d_method              = ilqr_config['c2d_method']
        time_step               = ilqr_config['time_step']
        descritize_func_curried = lambda state_space: gen_ctrl.discretize_state_space(state_space, time_step, c2d_method)
        return descritize_func_curried
    
    def create_curried_analyze_cost_dec_func(self, ilqr_config):
        cost_ratio_bounds = ilqr_config['cost_ratio_bounds']
        analyze_cost_dec_func_curried  = lambda cost_decrease_ratio: analyze_cost_decrease(cost_decrease_ratio, cost_ratio_bounds)
        return analyze_cost_dec_func_curried
    
def initialize_ilqr_controller(ilqr_config, ctrl_state:ilqrControllerState):
    # create object containing preconfigured functions with static parameters (currying and parameter encapsulation)
    config_funcs = ilqrConfiguredFuncs(ilqr_config, ctrl_state)
    # populate state sequence from seed control sequence
    state_seq = config_funcs.simulate_dyn_func(ctrl_state.seed_x_vec,
                                               ctrl_state.seed_u_seq)

    # calculate initial cost
    cost_float, cost_seq = gen_ctrl.calculate_total_cost(config_funcs.cost_func, state_seq, ctrl_state.u_seq)
    # ensure seed_cost is initialized with larger differential than convergence bound
    prev_cost_float = cost_float + (1 + ilqr_config['converge_crit'])
    # while within iteration limit and while cost has not converged    
    return config_funcs, state_seq, cost_float, prev_cost_float, cost_seq

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
        local_ctrl_state.x_seq, local_ctrl_state.u_seq, local_ctrl_state.cost_float, local_ctrl_state.cost_seq, local_ctrl_state.lsf_float, local_ctrl_state.ro_reg_change_bool = calculate_forwards_pass(ilqr_config, config_funcs, local_ctrl_state)
        
        # log iteration output
        if ilqr_config['log_ctrl_history'] is True and local_ctrl_state.ro_reg_change_bool is False:
            local_ctrl_state.append_controller_seqs()                  
        # calculate convergence measures
        avg_ff_gain = calculate_avg_ff_gains(local_ctrl_state.len_seq, local_ctrl_state.d_seq, local_ctrl_state.u_seq)
        converge_measure = jnp.abs(local_ctrl_state.prev_cost_float - local_ctrl_state.cost_float)/ctrl_state.cost_float
        # print controller info
        print('iteration number: ', local_ctrl_state.iter_int)
        print('converge_measure: ', converge_measure)
        print('ff_gain_avg_measure: ', avg_ff_gain)        
        print('ro_reg_value: ', local_ctrl_state.ro_reg)
        print('need to increase ro?: ', local_ctrl_state.ro_reg_change_bool)
        if (converge_measure < ilqr_config['converge_crit']) and (local_ctrl_state.ro_reg_change_bool is False):
            converge_reached = True
            print('controller converged')
        elif avg_ff_gain < ilqr_config['ff_gain_tol']:
            converge_reached = True    
            print('controller ff gains below convergence tolerance')            
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

    Ad_seq, Bd_seq = gen_ctrl.calculate_linearized_state_space_seq(config_funcs.state_trans_func,
                                                          config_funcs.c2d_ss_func,
                                                          ctrl_state.x_seq,
                                                          ctrl_state.u_seq,
                                                          ctrl_state.time_seq)
    
    P_N, p_N = calculate_final_ctg_approx(config_funcs.cost_func, 
                                                        ctrl_state.x_seq[-1], 
                                                        ctrl_state.u_len)
    
    P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N, ctrl_state.x_len,
                                                                          ctrl_state.u_len, ctrl_state.len_seq)
    if ctrl_state.ro_reg_change_bool is True:
        ro_reg = ctrl_state.ro_reg + ilqr_config['ro_reg_change']
    else:
        ro_reg = ilqr_config['ro_reg_start']
    is_valid_q_uu = False
    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq - 1):
            # k_step = -(idx + 1)
            k_step = ctrl_state.len_seq - (idx + 2)
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = taylor_expand_pseudo_hamiltonian(
                                                                        config_funcs.cost_func,
                                                                        Ad_seq[k_step],
                                                                        Bd_seq[k_step],
                                                                        ctrl_state.x_seq[k_step],
                                                                        ctrl_state.u_seq[k_step], # type: ignore
                                                                        P_kp1,
                                                                        p_kp1,
                                                                        ro_reg,
                                                                        k_step)
            # if q_uu_reg is not positive definite, reset loop to initial state and increase ro_reg
            if util.is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                ro_reg    = ro_reg + ilqr_config['ro_reg_change']
                P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N, ctrl_state.x_len,
                                                                                      ctrl_state.u_len, ctrl_state.len_seq)
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
    state_seq_new,control_seq_new,cost_float_new, cost_seq_new,in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
    line_search_factor = 1
    line_search_scale_param = 0.5
    iter_count = 0
    max_iter   = ilqr_config['fp_max_iter']
    ro_reg_change_bool = False
    norm_cost_decrease = ilqr_config['converge_crit'] + 1
    while not in_bounds_bool and (iter_count < max_iter):
        for k_step in range(ctrl_state.len_seq-1):
            # calculate updated control value
            u_k_new = calculate_u_k_new(ctrl_state.u_seq[k_step], # type: ignore
                                        ctrl_state.x_seq[k_step],
                                        state_seq_new[k_step], 
                                        ctrl_state.K_seq[k_step], 
                                        ctrl_state.d_seq[k_step], 
                                        line_search_factor)
            # calculate updated next state with dynamics function
            x_kp1_new = (config_funcs.simulate_dyn_func(state_seq_new[k_step], u_k_new))[-1] # type: ignore
            
            # populate state and control sequences
            control_seq_new[k_step]  = u_k_new # type: ignore
            state_seq_new[k_step+1]  = x_kp1_new 

        # shift sequences for cost calculation wrt desired trajectories
        # calculate new trajectory cost    
        cost_float_new, cost_seq_new = gen_ctrl.calculate_total_cost(config_funcs.cost_func, state_seq_new, control_seq_new)
        # calculate the ratio between the expected cost decrease and actual cost decrease
        actual_cost_decrease = cost_float_new - ctrl_state.cost_float
        norm_cost_decrease = jnp.abs(actual_cost_decrease)/ctrl_state.cost_float
        expected_cost_decrease = calculate_expected_cost_decrease(ctrl_state.Del_V_vec_seq, line_search_factor)
        cost_decrease_ratio = calculate_cost_decrease_ratio(ctrl_state.cost_float, cost_float_new, ctrl_state.Del_V_vec_seq, line_search_factor) 
        in_bounds_bool = config_funcs.analyze_cost_dec_func(cost_decrease_ratio)

        # if (norm_cost_decrease < ilqr_config['converge_crit']) and (iter_count < max_iter-1):
        #     # TODO check statement. if cost decrease is sufficiently small?
        #     break
        if in_bounds_bool:
            break    
        elif (iter_count == max_iter-1):
            iter_count     += 1  
            state_seq_new   = ctrl_state.x_seq
            control_seq_new = ctrl_state.u_seq
            cost_float_new  = ctrl_state.cost_float
            ro_reg_change_bool = True
        else:        # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
            state_seq_new,control_seq_new,cost_float_new, cost_seq_new, in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
            line_search_factor *= line_search_scale_param
            iter_count += 1
    return state_seq_new, control_seq_new, cost_float_new, cost_seq_new, line_search_factor, ro_reg_change_bool

def initialize_backwards_pass(P_N, p_N, x_len, u_len, len_seq):
    K_seq         = np.zeros([(len_seq-1), u_len, x_len])
    d_seq         = np.zeros([(len_seq-1), u_len, 1])
    Del_V_vec_seq = np.zeros([(len_seq-1), 2])
    P_kp1 = P_N
    p_kp1 = p_N
    return P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq

def initialize_forwards_pass(x_len:int, u_len:int, seed_state_vec:npt.ArrayLike, len_seq:int):
    control_seq_updated     = np.zeros([(len_seq-1),u_len, 1])
    state_seq_updated       = np.zeros([(len_seq),  x_len, 1])
    state_seq_updated[0][:] = seed_state_vec
    cost_float_updated      = 0.0
    cost_seq_updated            = np.zeros([len_seq, 1])
    in_bounds_bool          = False
    return state_seq_updated, control_seq_updated, cost_float_updated, cost_seq_updated, in_bounds_bool

def taylor_expand_pseudo_hamiltonian(cost_func, A_lin_k, B_lin_k, x_k, u_k, P_kp1, p_kp1, ro_reg, k_step):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'
    l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost(cost_func, x_k, u_k, k_step)
    P_kp1_reg = util.reqularize_mat(P_kp1, ro_reg)
    q_x      = l_x  + A_lin_k.T @ p_kp1
    q_u      = l_u  + B_lin_k.T @ p_kp1
    q_xx     = l_xx + A_lin_k.T @ P_kp1     @ A_lin_k
    q_uu     = l_uu + B_lin_k.T @ P_kp1     @ B_lin_k
    q_ux     = l_ux + B_lin_k.T @ P_kp1     @ A_lin_k
    q_uu_reg = l_uu + B_lin_k.T @ P_kp1_reg @ B_lin_k
    q_ux_reg = l_ux + B_lin_k.T @ P_kp1_reg @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg

def calculate_final_ctg_approx(cost_func, x_N, u_len):
    # approximates the final step cost-to-go as only the state with no control input
    # create concatenated state and control vector of primal points
    xu_N_jax, xu_N_len, x_N_len = gen_ctrl.prep_xu_vec_for_diff(x_N, np.zeros([u_len,1]))
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu = lambda xu_N: cost_func(xu_N[:x_N_len], xu_N[x_N_len:], -1, is_final_bool=True)
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat      = (jax.jacfwd(cost_func_xu)(xu_N_jax)).reshape(xu_N_len)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hessian_cat  = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_N_jax)).reshape(xu_N_len,xu_N_len)
    p_N          = np.array((jac_cat[:(len(x_N))]).reshape(-1,1))
    P_N          = np.array(hessian_cat[:(len(x_N)),:(len(x_N))])
    return P_N, p_N

def calculate_backstep_ctg_approx(q_x:npt.ArrayLike, q_u:npt.ArrayLike, q_xx:npt.ArrayLike, q_uu:npt.ArrayLike, q_ux:npt.ArrayLike, K_k:npt.ArrayLike, d_k:npt.ArrayLike):
    # P_k is the kth hessian approximation of the cost-to-go
    # p_k is the kth jacobian approximation of the cost-to-go
    # Del_V_k is the kth expected change in the value function
    assert q_x.shape  == (len(q_x), 1), 'q_x must be column vector (n,1)' # type: ignore
    assert q_u.shape  == (len(q_u), 1), 'q_u must be column vector (n,1)' # type: ignore
    assert K_k.shape  == (len(q_u), len(q_x)), "K_k incorrect shape, must be (len(q_u), len(q_x))" # type: ignore
    assert d_k.shape  == (len(q_u), 1       ), "d_k incorrect shape, must be (len(q_u), 1)" # type: ignore
    assert q_xx.shape == (len(q_x), len(q_x)), "q_xx incorrect shape, must be (len(q_x), len(q_x))" # type: ignore
    assert q_uu.shape == (len(q_u), len(q_u)), "q_uu incorrect shape, must be (len(q_u), len(q_u))"    # type: ignore
    assert q_ux.shape == (len(q_u), len(q_x)), "q_ux incorrect shape, must be (len(q_u), len(q_x))" # type: ignore
    K_kt        = K_k.T # type: ignore
    d_kt        = d_k.T # type: ignore
    q_xu        = q_ux.T # type: ignore
    P_k         = (q_xx) + (K_k.T @ q_uu @ K_k) + (K_k.T @ q_ux) + (q_xu @ K_k) # type: ignore
    p_k         = (q_x ) + (K_k.T @ q_uu @ d_k) + (K_k.T @ q_u ) + (q_xu @ d_k) # type: ignore
    # TODO verify Del_V_vec_k calculation signs, absolute vs actual, actual gives different signs...
    # Del_V_vec_k = jnp.array([[-jnp.abs((d_kt @ q_u))],[-jnp.abs((0.5) * (d_kt @ q_uu @ d_k))]]).reshape(1,-1)
    Del_V_vec_k = np.array([(d_k.T @ q_u),((0.5) * (d_k.T @ q_uu @ d_k))]).reshape(1,-1) # type: ignore
    return P_k, p_k, Del_V_vec_k[0]

def calculate_optimal_gains(q_uu_reg, q_ux, q_u):
    # K_k is a linear feedback term related to del_x from nominal trajectory
    # d_k is a feedforward term related to trajectory correction
    inverted_q_uu = np.linalg.inv(q_uu_reg)
    K_k           = -inverted_q_uu @ q_ux
    d_k           = -inverted_q_uu @ q_u
    return K_k, d_k

def calculate_u_k_new(u_nom_k ,x_nom_k ,x_new_k, K_k, d_k, line_search_factor):
    # check valid dimensions
    assert K_k.shape    == (len(u_nom_k), len(x_nom_k)), "K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))"
    assert d_k.shape    == (len(u_nom_k), 1)           , "d_k incorrect shape, must be (len(u_nom_k), 1)"
    assert len(x_new_k) ==  len(x_nom_k)               , "x_nom_k and x_new_k must be the same length"
    assert (x_nom_k.shape) == (len(x_nom_k), 1), 'x_nom_k must be column vector (n,1)'
    assert (x_new_k.shape) == (len(x_new_k), 1), 'x_new_k must be column vector (n,1)'    
    assert (u_nom_k.shape) == (len(u_nom_k), 1), 'u_nom_k must be column vector (m,1)'    
    # form column vectors
    u_k_updated = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
    return u_k_updated

def calculate_cost_decrease_ratio(prev_cost_float, new_cost_float, Del_V_vec_seq, line_search_factor):
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
    cost_decrease_ratio = (1 / -Del_V_sum) * (prev_cost_float - new_cost_float)
    return cost_decrease_ratio

def calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor):
    assert (Del_V_vec_seq.shape)[1] == 2, "Del_V_vec_seq is not the right shape, must be (:,2)" 
    Del_V_sum = 0
    for idx in range(len(Del_V_vec_seq)):
        Del_V_sum = Del_V_sum + ((line_search_factor * Del_V_vec_seq[idx,0]) + (line_search_factor**2 * Del_V_vec_seq[idx,1]))
    return Del_V_sum    

def analyze_cost_decrease(cost_decrease_ratio, cost_ratio_bounds):
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (cost_decrease_ratio < cost_ratio_bounds[1]):
        in_bounds_bool = True
    else:
        in_bounds_bool = False            
    return in_bounds_bool

def calculate_avg_ff_gains(iter_int, d_seq, U_seq):

    # TODO check whether U_seq norm is norm 1 or norm 2
    norm_factor = 1 / (iter_int - 1)
    norm_gain_sum = 0.0
    for k in range(iter_int-1):
        norm_gain_sum += (abs(max(d_seq[k])) / (np.linalg.norm(U_seq[k]) + 1)).item()
    avg_ff_gains = norm_factor * norm_gain_sum
    return avg_ff_gains