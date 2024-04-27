#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
#https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax.numpy as jnp
import copy
import numpy as np
from numpy import typing as npt
from typing import Sequence, Callable, Tuple
import mujoco as mujoco

import ilqr_utils as util
import gen_ctrl_funcs as gen_ctrl
import mujoco_funcs as mj_funcs
import cost_functions as cost
import dyn_functions as dyn


class ilqrConfigStruct:
    def __init__(self, num_states:int, num_controls:int, len_seq:int, time_step:float)-> None:  
        self.num_states:int             = num_states
        self.num_controls:int           = num_controls
        self.len_seq:int                = len_seq
        self.time_step:float            = time_step
        self.max_iter:int               = 20
        self.converge_crit:float        = 1e-5
        self.ff_gain_tol:float          = 1e-4
        self.cost_ratio_bounds:Tuple[float,float] = (1e-8, 10)
        self.ro_reg_start:float         = 0.0
        self.ro_reg_change:float        = 0.5
        self.fp_max_iter:int            = 10
        self.ls_scale_alpha_param:float = 0.5
        self.log_ctrl_history:bool      = True
        self.is_dyn_configured:bool     = False
        self.is_cost_configured:bool    = False
        self.is_curried:bool            = False

    def config_cost_func(self,
            cost_func:Callable[
                [cost.costFuncParams, npt.NDArray[np.float64], npt.NDArray[np.float64], int, bool],
                Tuple[npt.NDArray,npt.NDArray,npt.NDArray]], 
            cost_func_params:cost.costFuncParams)-> None:
        '''
        This function receives the proposed cost function and parameters for the cost function, and generates
        The specific cost function for the controller to use with curried inputs
        ''' 
        self.cost_func_curried:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64], int, bool],
            Tuple[npt.NDArray,npt.NDArray,npt.NDArray]] = lambda x, u, k, is_final_bool: cost_func(cost_func_params, x, u, k, is_final_bool)
        
        self.cost_func_for_calc = gen_ctrl.prep_cost_func_for_calc(self.cost_func_curried)
        self.cost_func_for_diff = gen_ctrl.prep_cost_func_for_diff(self.cost_func_curried)
        self.is_cost_configured = True
        self.cost_func_params   = cost_func_params

    def config_for_dyn_func(self,
                            cont_dyn_func:Callable[[dyn.dynFuncParams, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
                            dyn_func_params:dyn.dynFuncParams,
                            integrate_func:Callable[[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
                                                    float, npt.NDArray[np.float64], npt.NDArray[np.float64]], 
                                                    npt.NDArray[np.float64]]) -> None:
        '''
        This function receives python functions for dynamics and prepares the curried functions for the controller to use.\n
        dyn_func[in] - callable that returns the continuous dynamics (state_dot) given current state, control, and a set of paramters\n
        dyn_func_params[in] - struct of dynamics function parameters to specify dynamic function values (mass, damping, springrate etc.)\n
        integrate_func[in] - callable that recieves the continuous dynamic function and returns the integrated value at t + timestep.
        Can be viewed as a discretizer (rk4, euler etc.)
        '''
        # check whether struct already configured for dynamics
        if self.is_dyn_configured is True:
            ValueError("struct already configured for dynamics")
        # Populate discrete dynamics function
        self.cont_dyn_func   = cont_dyn_func
        self.dyn_func_params = dyn_func_params
        self.integrate_func  = integrate_func
        self.is_cost_configured = True
        self.mj_ctrl:bool       = False
        # TODO create update-timestep script to update currying if timestep is changed after func config

    def config_for_mujoco(self, mjcf_model)->None: 
        # check whether struct already configured for dynamics
        if self.is_dyn_configured is True:
            ValueError("struct already configured for dynamics")
        self.mj_model, _, self.mj_data = mj_funcs.create_mujoco_model(mjcf_model, self.time_step)
        self.mjcf_model                = mjcf_model
        mujoco.mj_forward(self.mj_model,self.mj_data)        
        self.mj_ctrl:bool              = True   

    def curry_funcs_for_mujoco(self)->None:
        self.discrete_dyn_func:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
             npt.NDArray[np.float64]] = lambda x, u : mj_funcs.fwd_sim_mj_w_ctrl_step(self.mj_model, self.mj_data, x, u)
        
        self.simulate_fwd_dyn_seq:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
            npt.NDArray[np.float64]] = lambda x_init, u_seq: mj_funcs.fwd_sim_mj_w_ctrl(self.mj_model, self.mj_data, x_init, u_seq)

        self.linearize_dyn_seq:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = lambda x_seq, u_seq: mj_funcs.linearize_mj_seq(self.mj_model, self.mj_data,
                                                                                                                      x_seq, u_seq)
        
    def curry_funcs_for_dyn_funcs(self)->None:

        self.cont_dyn_func_curried:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray] = lambda x, u: self.cont_dyn_func(self.dyn_func_params, x, u)
        
        self.discrete_dyn_func:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
            npt.NDArray[np.float64]] = lambda x, u: self.integrate_func(self.cont_dyn_func_curried, self.time_step, x, u)

        self.simulate_fwd_dyn_seq:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
            npt.NDArray[np.float64]] = lambda x_init, u_seq: gen_ctrl.simulate_forward_dynamics_seq(self.discrete_dyn_func, x_init, u_seq)
        
        self.linearize_dyn_seq:Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], 
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = lambda x_seq, u_seq: gen_ctrl.calculate_linearized_state_space_seq(
                                                                                                                        self.discrete_dyn_func,
                                                                                                                        x_seq, u_seq) 

    def create_curried_analyze_cost_dec_func(self)-> None:
        self.analyze_cost_dec = lambda cost_decrease_ratio: analyze_cost_decrease(cost_decrease_ratio, self.cost_ratio_bounds)

    
    def create_curried_funcs(self):
        if self.is_cost_configured is False or self.is_dyn_configured is False:
            ValueError("Controller is not configured yet")
        self.create_curried_analyze_cost_dec_func()
        if self.mj_ctrl is True:
            self.curry_funcs_for_mujoco()
        else:
            self.curry_funcs_for_dyn_funcs()
        self.is_curried = True   

class ilqrControllerState:
    def __init__(self,
                 ilqr_config:ilqrConfigStruct, 
                 seed_x_vec:npt.NDArray[np.float64], 
                 seed_u_seq:npt.NDArray[np.float64]):
        # direct parameter population
        self.seed_x_vec         = seed_x_vec     
        self.seed_u_seq         = seed_u_seq           
        self.validate_input_data(ilqr_config, seed_x_vec, seed_u_seq)            
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
        self.seed_x_seq    = np.zeros([self.len_seq,self.x_len])      
        self.x_seq         = np.zeros([self.len_seq,self.x_len])
        self.time_seq      = np.arange(self.len_seq) * ilqr_config.time_step
        self.K_seq         = np.zeros([self.len_seq-1, self.u_len, self.x_len])
        self.d_seq         = np.zeros([self.len_seq-1, self.u_len,          1])
        self.Del_V_vec_seq = np.zeros([self.len_seq-1, 2])
        self.cost_seq      = np.zeros([self.len_seq])
        self.x_cost_seq    = np.zeros([self.len_seq])
        self.u_cost_seq    = np.zeros([self.len_seq])        

        if ilqr_config.log_ctrl_history is True:
            self.u_seq_history:list         = []
            self.x_seq_history:list         = []
            self.K_seq_history:list         = []
            self.d_seq_history:list         = []
            self.Del_V_vec_seq_history:list = []
            self.cost_seq_history:list      = []
            self.x_cost_seq_history:list    = []                  
            self.u_cost_seq_history:list    = []
            self.cost_float_history:list    = []
            self.lsf_float_history:list     = []

    def validate_input_data(self,ilqr_config:ilqrConfigStruct, seed_x_vec:npt.NDArray, seed_u_seq:npt.NDArray):
        if self.get_len_seq() != ilqr_config.len_seq:
            ValueError("sequences are the wrong length")
        if self.get_num_states() != ilqr_config.num_states:
            ValueError("incorrect state vector length")
        if self.get_num_controls() != ilqr_config.num_controls:
            ValueError("incorrect control vector length")    
        # TODO create validation criteria

    def get_len_seq(self):
        return len(self.seed_u_seq)+1
    
    def get_num_states(self):
        return len(self.seed_x_vec) 
    
    def get_num_controls(self):
        return len(self.seed_u_seq[0])
    
    def append_controller_seqs(self):
        self.u_seq_history.append(self.u_seq)
        self.x_seq_history.append(self.x_seq)
        self.cost_seq_history.append(self.cost_seq) 
        self.x_cost_seq_history.append(self.x_cost_seq) 
        self.u_cost_seq_history.append(self.u_cost_seq) 
        self.cost_float_history.append(self.cost_float)
        self.lsf_float_history.append(self)
        self.K_seq_history.append(self.K_seq) 
        self.d_seq_history.append(self.d_seq)
        self.Del_V_vec_seq_history.append(self.Del_V_vec_seq)   

def initialize_ilqr_controller(ilqr_config:ilqrConfigStruct, 
                               ctrl_state:ilqrControllerState)-> Tuple[
                                   npt.NDArray[np.float64], float, float, npt.NDArray[np.float64]]:
    # create object containing preconfigured functions with static parameters (currying and parameter encapsulation)

    # populate state sequence from seed control sequence
    x_seq = ilqr_config.simulate_fwd_dyn_seq(ctrl_state.seed_x_vec,ctrl_state.seed_u_seq)

    # calculate initial cost
    cost_float, cost_seq, _, _ = gen_ctrl.calculate_total_cost(ilqr_config.cost_func_for_calc, x_seq, ctrl_state.u_seq)
    # ensure seed_cost is initialized with larger differential than convergence bound

    prev_cost_float = cost_float + (1 + ilqr_config.converge_crit)
    # while within iteration limit and while cost has not converged    
    return x_seq, cost_float, prev_cost_float, cost_seq

def run_ilqr_controller(ilqr_config:ilqrConfigStruct, ctrl_state:ilqrControllerState):
    local_ctrl_state = copy.deepcopy(ctrl_state)
    converge_measure = ilqr_config.converge_crit + 1
    converge_reached = False
    while (local_ctrl_state.iter_int < ilqr_config.max_iter) and (converge_reached is False):
        # retain previous cost value for convergence check
        local_ctrl_state.iter_int += 1
        local_ctrl_state.prev_cost_float = local_ctrl_state.cost_float
        # perform backwards pass to calculate gains and expected value function change
        local_ctrl_state.K_seq, local_ctrl_state.d_seq, local_ctrl_state.Del_V_vec_seq, local_ctrl_state.ro_reg = calculate_backwards_pass(ilqr_config, local_ctrl_state)
        # perform forwards pass to calculate new trajectory and trajectory cost
        local_ctrl_state.x_seq, local_ctrl_state.u_seq, local_ctrl_state.cost_float, local_ctrl_state.cost_seq,local_ctrl_state.x_cost_seq, local_ctrl_state.u_cost_seq, local_ctrl_state.lsf_float, local_ctrl_state.ro_reg_change_bool = calculate_forwards_pass(ilqr_config, 
                                                                                                                                                                                                                                                                   local_ctrl_state)
        
        # log iteration output
        if ilqr_config.log_ctrl_history is True and local_ctrl_state.ro_reg_change_bool is False:
            local_ctrl_state.append_controller_seqs()                  
        # calculate convergence measures

        avg_ff_gain = calculate_avg_ff_gains(local_ctrl_state.len_seq, local_ctrl_state.d_seq, local_ctrl_state.u_seq)
        converge_measure = np.abs((local_ctrl_state.prev_cost_float - local_ctrl_state.cost_float)/ctrl_state.cost_float).item()
        # determine convergence criteria
        converge_reached = determine_convergence(ilqr_config, local_ctrl_state, converge_measure, avg_ff_gain)

        # print controller info
        print('iteration number: ', local_ctrl_state.iter_int)
        print('converge_measure: ', converge_measure)
        print('ff_gain_avg_measure: ', avg_ff_gain)        
        print('ro_reg_value: ', local_ctrl_state.ro_reg)
        print('need to increase ro?: ', local_ctrl_state.ro_reg_change_bool)


    return local_ctrl_state

def calculate_backwards_pass(ilqr_config:ilqrConfigStruct, ctrl_state:ilqrControllerState):
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

    Ad_seq, Bd_seq = ilqr_config.linearize_dyn_seq(ctrl_state.x_seq, ctrl_state.u_seq)    
    
    P_N, p_N = calculate_final_ctg_approx(ilqr_config.cost_func_for_diff, 
                                                        ctrl_state.x_seq[-1], 
                                                        ctrl_state.u_len,
                                                        ctrl_state.len_seq)
    
    P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N, ctrl_state.x_len,
                                                                          ctrl_state.u_len, ctrl_state.len_seq)
    if ctrl_state.ro_reg_change_bool is True:
        ro_reg = ctrl_state.ro_reg + ilqr_config.ro_reg_change
    else:
        ro_reg = ilqr_config.ro_reg_start
    is_valid_q_uu = False
    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq - 1):
            # solve newton-step condition backwards from sequence end
            k_step = ctrl_state.len_seq - (idx + 2) 
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = taylor_expand_pseudo_hamiltonian(
                                                                        ilqr_config.cost_func_for_diff,
                                                                        Ad_seq[k_step],
                                                                        Bd_seq[k_step],
                                                                        ctrl_state.x_seq[k_step],
                                                                        ctrl_state.u_seq[k_step],
                                                                        P_kp1,
                                                                        p_kp1,
                                                                        ro_reg,
                                                                        k_step)
            # if q_uu_reg is not positive definite, reset loop to initial state and increase ro_reg
            if util.is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                ro_reg    = ro_reg + ilqr_config.ro_reg_change
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

def calculate_forwards_pass(ilqr_config:ilqrConfigStruct, ctrl_state:ilqrControllerState):
    #initialize updated state vec, updated control_vec, line search factor
    x_seq_new,u_seq_new,cost_float_new, cost_seq_new,x_cost_seq_new,u_cost_seq_new,in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
    line_search_factor = 1.0
    line_search_scale_param = ilqr_config.ls_scale_alpha_param
    iter_count = 0
    ro_reg_change_bool = False
    while not in_bounds_bool and (iter_count < ilqr_config.fp_max_iter):
        for k in range(ctrl_state.len_seq-1):
            # calculate updated control value
            u_seq_new[k] = calculate_u_k_new(ctrl_state.u_seq[k], 
                                        ctrl_state.x_seq[k],
                                        x_seq_new[k], 
                                        ctrl_state.K_seq[k], 
                                        ctrl_state.d_seq[k], 
                                        line_search_factor)
            # calculate updated next state with dynamics function
            x_seq_new[k+1] = ilqr_config.discrete_dyn_func(x_seq_new[k], u_seq_new[k])

        # shift sequences for cost calculation wrt desired trajectories
        # calculate new trajectory cost    
        cost_float_new, cost_seq_new, x_cost_seq_new, u_cost_seq_new = gen_ctrl.calculate_total_cost(ilqr_config.cost_func_for_calc, x_seq_new, u_seq_new)
        # calculate the ratio between the expected cost decrease and actual cost decrease
        cost_decrease_ratio = calculate_cost_decrease_ratio(ctrl_state.cost_float, cost_float_new, ctrl_state.Del_V_vec_seq, line_search_factor) 
        in_bounds_bool = ilqr_config.analyze_cost_dec(cost_decrease_ratio)

        # if (norm_cost_decrease < ilqr_config['converge_crit']) and (iter_count < max_iter-1):
        #     # TODO check statement. if cost decrease is sufficiently small?
        #     break
        if in_bounds_bool:
            break    
        elif (iter_count == ilqr_config.fp_max_iter-1):
            iter_count     += 1  
            x_seq_new   = ctrl_state.x_seq
            u_seq_new = ctrl_state.u_seq
            cost_float_new  = ctrl_state.cost_float
            ro_reg_change_bool = True
        else:        # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
            x_seq_new,u_seq_new,cost_float_new, cost_seq_new,x_cost_seq_new,u_cost_seq_new, in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
            line_search_factor *= line_search_scale_param
            iter_count += 1
    return x_seq_new, u_seq_new, cost_float_new, cost_seq_new,x_cost_seq_new, u_cost_seq_new, line_search_factor, ro_reg_change_bool

def initialize_backwards_pass(P_N:npt.NDArray, p_N:npt.NDArray, x_len:int, u_len:int, len_seq:int):
    K_seq         = np.zeros([(len_seq-1), u_len, x_len])
    d_seq         = np.zeros([(len_seq-1), u_len, 1])
    Del_V_vec_seq = np.zeros([(len_seq-1), 2])
    P_kp1 = P_N
    p_kp1 = p_N
    return P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq

def initialize_forwards_pass(x_len:int, u_len:int, seed_state_vec:npt.NDArray, len_seq:int):
    control_seq_updated     = np.zeros([(len_seq-1),u_len])
    state_seq_updated       = np.zeros([(len_seq),  x_len])
    state_seq_updated[0][:] = seed_state_vec
    cost_float_updated      = 0.0
    cost_seq_updated        = np.zeros([len_seq])
    x_cost_seq_updated      = np.zeros([len_seq])   
    u_cost_seq_updated      = np.zeros([len_seq]) 
    in_bounds_bool          = False
    return state_seq_updated, control_seq_updated, cost_float_updated, cost_seq_updated, x_cost_seq_updated,u_cost_seq_updated,in_bounds_bool

def taylor_expand_pseudo_hamiltonian(cost_func_for_diff:Callable[
                                        [npt.NDArray[np.float64], npt.NDArray[np.float64], int, bool],
                                        npt.NDArray], 
                                     A_lin_k:npt.NDArray[np.float64],
                                     B_lin_k:npt.NDArray[np.float64], 
                                     x_k:npt.NDArray[np.float64], 
                                     u_k:npt.NDArray[np.float64], 
                                     P_kp1:npt.NDArray[np.float64], 
                                     p_kp1:npt.NDArray[np.float64], 
                                     ro_reg:float, 
                                     k_step:int):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'
    l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost(cost_func_for_diff, x_k, u_k, k_step)
    if ro_reg > 0:
        P_kp1_reg = util.reqularize_mat(P_kp1, ro_reg)
    else:
        P_kp1_reg = P_kp1    
    q_x      = l_x  + A_lin_k.T @ p_kp1
    q_u      = l_u  + B_lin_k.T @ p_kp1
    q_xx     = l_xx + A_lin_k.T @ P_kp1     @ A_lin_k
    q_uu     = l_uu + B_lin_k.T @ P_kp1     @ B_lin_k
    q_ux     = l_ux + B_lin_k.T @ P_kp1     @ A_lin_k
    q_uu_reg = l_uu + B_lin_k.T @ P_kp1_reg @ B_lin_k
    q_ux_reg = l_ux + B_lin_k.T @ P_kp1_reg @ A_lin_k
    return q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg

def calculate_final_ctg_approx(cost_func_for_diff:Callable[
                                        [npt.NDArray[np.float64], npt.NDArray[np.float64], int, bool],
                                        npt.NDArray], 
                                x_N:npt.NDArray[np.float64], u_len:int, len_seq:int) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    # approximates the final step cost-to-go as only the state with no control input
    # create concatenated state and control vector of primal points
    u_N = np.zeros([u_len], dtype=float)
    k_step = len_seq - 1
    p_N, _, P_N, _, _ = gen_ctrl.taylor_expand_cost(cost_func_for_diff, x_N, u_N, k_step, is_final_bool=True)
    return P_N, p_N

def calculate_backstep_ctg_approx(q_x:npt.NDArray[np.float64], 
                                  q_u:npt.NDArray[np.float64], 
                                  q_xx:npt.NDArray[np.float64], 
                                  q_uu:npt.NDArray[np.float64], 
                                  q_ux:npt.NDArray[np.float64], 
                                  K_k:npt.NDArray[np.float64], 
                                  d_k:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    # P_k is the kth hessian approximation of the cost-to-go
    # p_k is the kth jacobian approximation of the cost-to-go
    # Del_V_k is the kth expected change in the value function
    assert q_x.shape  == (len(q_x), 1), 'q_x must be column vector (n,1)' 
    assert q_u.shape  == (len(q_u), 1), 'q_u must be column vector (n,1)' 
    assert K_k.shape  == (len(q_u), len(q_x)), "K_k incorrect shape, must be (len(q_u), len(q_x))" 
    assert d_k.shape  == (len(q_u), 1       ), "d_k incorrect shape, must be (len(q_u), 1)"
    assert q_xx.shape == (len(q_x), len(q_x)), "q_xx incorrect shape, must be (len(q_x), len(q_x))" 
    assert q_uu.shape == (len(q_u), len(q_u)), "q_uu incorrect shape, must be (len(q_u), len(q_u))"    
    assert q_ux.shape == (len(q_u), len(q_x)), "q_ux incorrect shape, must be (len(q_u), len(q_x))" 
    q_xu        = q_ux.T 
    P_k         = (q_xx) + (K_k.T @ q_uu @ K_k) + (K_k.T @ q_ux) + (q_xu @ K_k) 
    p_k         = (q_x ) + (K_k.T @ q_uu @ d_k) + (K_k.T @ q_u ) + (q_xu @ d_k) 
    # TODO verify Del_V_vec_k calculation signs, absolute vs actual, actual gives different signs...
    Del_V_vec_k = np.array([(d_k.T @ q_u),((0.5) * (d_k.T @ q_uu @ d_k))]).reshape(1,-1) 
    return P_k, p_k, Del_V_vec_k[0]

def calculate_optimal_gains(q_uu_reg:npt.NDArray[np.float64], 
                            q_ux:npt.NDArray[np.float64], 
                            q_u:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    # K_k is a linear feedback term related to del_x from nominal trajectory
    # d_k is a feedforward term related to trajectory correction
    inverted_q_uu = np.linalg.inv(q_uu_reg)
    K_k           = -inverted_q_uu @ q_ux
    d_k           = -inverted_q_uu @ q_u
    return K_k, d_k

def calculate_u_k_new(u_nom_k:npt.NDArray[np.float64],
                      x_nom_k:npt.NDArray[np.float64],
                      x_new_k:npt.NDArray[np.float64], 
                      K_k:npt.NDArray[np.float64], 
                      d_k:npt.NDArray[np.float64], line_search_factor:float) -> npt.NDArray[np.float64]:
    # check valid dimensions
    assert K_k.shape    == (len(u_nom_k), len(x_nom_k)), "K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))"
    assert d_k.shape    == (len(u_nom_k), 1)           , "d_k incorrect shape, must be (len(u_nom_k), 1)"
    assert len(x_new_k) ==  len(x_nom_k)               , "x_nom_k and x_new_k must be the same length"
    assert (x_nom_k.shape) == (len(x_nom_k),), 'x_nom_k must be column vector (n,)'
    assert (x_new_k.shape) == (len(x_new_k),), 'x_new_k must be column vector (n,)'    
    assert (u_nom_k.shape) == (len(u_nom_k),), 'u_nom_k must be column vector (m,)'    
    # form column vectors
    return (u_nom_k.reshape(-1,1) + K_k @ (x_new_k.reshape(-1,1) - x_nom_k.reshape(-1,1)) + line_search_factor * d_k).reshape(-1)

def calculate_cost_decrease_ratio(prev_cost_float:float, new_cost_float:float, Del_V_vec_seq:npt.NDArray[np.float64], line_search_factor:float) -> float:
    """
    Calculates the ratio between the expected cost decrease from approximated dynamics and cost functions, and the calculated
    cost from the forward pass shooting method. The cost decrease is expected to be positive, as the Del_V_sum should be a negative
    expected value, and the new cost should be smaller than the previous cost.
    """
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
    cost_decrease_ratio = (1 / Del_V_sum) * (new_cost_float - prev_cost_float)
    return cost_decrease_ratio

def calculate_expected_cost_decrease(Del_V_vec_seq:npt.NDArray[np.float64], line_search_factor:float) -> float:
    assert (Del_V_vec_seq.shape)[1] == 2, "Del_V_vec_seq is not the right shape, must be (:,2)" 
    Del_V_sum = 0.0
    for idx in range(len(Del_V_vec_seq)):
        Del_V_sum += ((line_search_factor * Del_V_vec_seq[idx,0]) + (line_search_factor**2 * Del_V_vec_seq[idx,1]))
    if Del_V_sum > 0:
        a = 1
        # TODO include some logic or statement here that flags a failed decrease? may be unnecessary given the line search factor catch
    return Del_V_sum    

def analyze_cost_decrease(cost_decrease_ratio:float, cost_ratio_bounds:Tuple[float, float]) -> bool:
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (cost_decrease_ratio < cost_ratio_bounds[1]):
        in_bounds_bool = True
    else:
        in_bounds_bool = False            
    return in_bounds_bool

def calculate_avg_ff_gains(iter_int:int, d_seq:npt.NDArray[np.float64], U_seq:npt.NDArray[np.float64]) -> float:
    # TODO check whether U_seq norm is norm 1 or norm 2
    norm_factor = 1 / (iter_int - 1)
    norm_gain_sum = 0.0
    for k in range(iter_int-1):
        norm_gain_sum += (abs(max(d_seq[k])) / (np.linalg.norm(U_seq[k]) + 1)).item()
    avg_ff_gains = norm_factor * norm_gain_sum
    return avg_ff_gains


def calculate_u_star(k_fb_k:npt.NDArray[np.float64], 
                     u_des_k:npt.NDArray[np.float64], 
                     x_des_k:npt.NDArray[np.float64], 
                     x_k:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (u_des_k.reshape(-1,1) + k_fb_k @ (x_k.reshape(-1,1) - x_des_k.reshape(-1,1))).reshape(-1)


def simulate_ilqr_output(sim_dyn_func_step:Callable[
        [npt.NDArray[np.float64],npt.NDArray[np.float64]],  
        npt.NDArray[np.float64]], 
        ctrl_out:ilqrControllerState, x_sim_init:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    x_sim_seq  = np.zeros([ctrl_out.len_seq, ctrl_out.x_len])
    u_sim_seq  = np.zeros([ctrl_out.len_seq-1, ctrl_out.u_len])
    x_sim_seq[0] = x_sim_init    
    for k in range(ctrl_out.len_seq-1):
        u_sim_seq[k]     = calculate_u_star(ctrl_out.K_seq[k], ctrl_out.u_seq[k], ctrl_out.x_seq[k], x_sim_seq[k])
        x_sim_seq[k+1]   = sim_dyn_func_step(x_sim_seq[k], u_sim_seq[k])
    return x_sim_seq,u_sim_seq

def determine_convergence(ilqr_config:ilqrConfigStruct, ctrl_state:ilqrControllerState, converge_measure:float, avg_ff_gain:float) -> bool:
    converge_reached = False
    if (converge_measure < ilqr_config.converge_crit) and (ctrl_state.ro_reg_change_bool is False):
        converge_reached = True
        print('controller converged')
    elif avg_ff_gain < ilqr_config.ff_gain_tol:
        converge_reached = True    
        print('controller ff gains below convergence tolerance')            
    elif ctrl_state.iter_int == ilqr_config.max_iter:  
        print('controller reached iteration limit')
    return converge_reached
