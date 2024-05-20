#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
#https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

# This script provides functions for the ilqr algorithm outlined in the above links.
# iLQR is an algorithm for trajectory optimization.

import jax.numpy as jnp
from jax import lax
import copy
import numpy as np
from numpy import typing as npt
from typing import Sequence, Callable, Tuple, Union
import mujoco as mujoco
import time

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
        self.converge_crit:float        = 1e-4
        self.ff_gain_tol:float          = 1e-4
        self.cost_ratio_bounds:Tuple[float,float] = (1e-7, 10)
        self.ro_reg_start:float         = 0.0
        self.ro_reg_change:float        = 0.5
        self.fp_max_iter:int            = 5
        self.ls_scale_alpha_param:float = 0.5
        self.log_ctrl_history:bool      = False
        self.is_dyn_configured:bool     = False
        self.is_cost_configured:bool    = False
        self.is_curried:bool            = False

    def config_cost_func(self,
            cost_func_calc:Callable[[jnp.ndarray, jnp.ndarray, int],Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]],
            cost_func_diff:Callable[[jnp.ndarray, jnp.ndarray, int],Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]])-> None:
        '''
        This function receives the proposed cost function and parameters for the cost function, and generates
        The specific cost function for the controller to use with curried inputs
        ''' 
        # self.cost_func_for_calc = gen_ctrl.prep_cost_func_for_calc(cost_func_calc)
        self.cost_func_for_calc = cost_func_calc
        self.cost_func_for_diff = gen_ctrl.prep_cost_func_for_diff(cost_func_diff)
        self.quad_exp_cost_seq  = lambda x_seq, u_seq: gen_ctrl.taylor_expand_cost_seq(self.cost_func_for_diff,
                                                                                        x_seq, u_seq)
        self.is_cost_configured = True


    def config_for_dyn_func(self,
                            cont_dyn_func:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                            integrate_func:Callable[[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                                                    float, jnp.ndarray, jnp.ndarray], 
                                                    jnp.ndarray]) -> None:
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
        mujoco.mj_forward(self.mj_model,self.mj_data)         # type: ignore
        self.mj_ctrl:bool              = True   

    def _curry_funcs_for_mujoco(self)->None:
        self.discrete_dyn_mj_func:Callable[
            [jnp.ndarray, jnp.ndarray], 
             jnp.ndarray] = lambda x, u : mj_funcs.fwd_sim_mj_w_ctrl_step(self.mj_model, self.mj_data, x, u)
        
        self.simulate_fwd_dyn_seq:Callable[
            [jnp.ndarray, jnp.ndarray], 
            jnp.ndarray] = lambda x_init, u_seq: mj_funcs.fwd_sim_mj_w_ctrl(self.mj_model, self.mj_data, x_init, u_seq)

        self.linearize_dyn_seq:Callable[
            [jnp.ndarray, jnp.ndarray], 
            Tuple[jnp.ndarray, jnp.ndarray]] = lambda x_seq, u_seq: mj_funcs.linearize_mj_seq(self.mj_model, self.mj_data,
                                                                                                                      x_seq, u_seq)
        
    def _curry_funcs_for_dyn_funcs(self)->None:
        self.discrete_dyn_func:Callable[
            [jnp.ndarray, jnp.ndarray], 
            jnp.ndarray] = lambda x, u: self.integrate_func(self.cont_dyn_func, self.time_step, x, u)

        self.simulate_fwd_dyn_seq:Callable[
            [jnp.ndarray, jnp.ndarray], 
            jnp.ndarray] = lambda x_init, u_seq: gen_ctrl.simulate_forward_dynamics_seq(self.discrete_dyn_func, x_init, u_seq)
        
        scan_lin_dyn_func = lambda carry, xu_seq: gen_ctrl.scan_dyn_func_For_lin(self.discrete_dyn_func,self.num_states, carry, xu_seq)
        
        self.linearize_dyn_seq:Callable[
            [jnp.ndarray, jnp.ndarray], 
            Tuple[jnp.ndarray, jnp.ndarray]] = lambda x_seq, u_seq: gen_ctrl.calculate_linearized_state_space_seq(scan_lin_dyn_func,
                                                                                                                        x_seq, u_seq) 

        fp_scan_func = lambda lsf, carry, seqs: scan_func_forward_pass(self.discrete_dyn_func,
                                                                       self.cost_func_for_diff,
                                                                       lsf, carry, seqs)

        self.simulate_forward_pass = lambda x_seq, u_seq, K_seq, d_seq, lsf : simulate_forward_pass(fp_scan_func,self.cost_func_for_diff, x_seq, u_seq, K_seq,d_seq, lsf)
    
    def _create_curried_analyze_cost_dec_func(self)-> None:
        self.analyze_cost_dec = lambda cost_decrease_ratio: analyze_cost_decrease(cost_decrease_ratio, self.cost_ratio_bounds)
  
    def create_curried_funcs(self):
        if self.is_cost_configured is False or self.is_dyn_configured is False:
            ValueError("Controller is not configured yet")
        self._create_curried_analyze_cost_dec_func()
        if self.mj_ctrl is True:
            self._curry_funcs_for_mujoco()
        else:
            self._curry_funcs_for_dyn_funcs()
        self.is_curried = True   

class ilqrControllerState:
    def __init__(self,
                 ilqr_config:ilqrConfigStruct, 
                 seed_x_vec:Union[jnp.ndarray, npt.NDArray[np.float64]], 
                 seed_u_seq:Union[jnp.ndarray, npt.NDArray[np.float64]]):
        # direct parameter population
        self.seed_x_vec         = jnp.array(seed_x_vec)     
        self.seed_u_seq         = jnp.array(seed_u_seq)           
        self.validate_input_data(ilqr_config, self.seed_x_vec, self.seed_u_seq)            
        self.cost_float         = 0.0
        self.prev_cost_float    = 0.0
        self.iter_int           = 0 
        self.ff_gain_avg        = 0.0        
        self.ro_reg             = 0.0
        self.lsf_float          = 1.0
        self.ro_reg_change_bool = False    
        
        # dependent parameter population
        self.u_seq         = jnp.array(seed_u_seq)   
        self.len_seq       = self.get_len_seq()
        self.x_len         = self.get_num_states()
        self.u_len         = self.get_num_controls()
        self.seed_x_seq    = jnp.zeros([self.len_seq,self.x_len])      
        self.x_seq         = jnp.zeros([self.len_seq,self.x_len])
        self.time_seq      = jnp.arange(self.len_seq) * ilqr_config.time_step
        self.K_seq         = jnp.zeros([self.len_seq-1, self.u_len, self.x_len])
        self.d_seq         = jnp.zeros([self.len_seq-1, self.u_len,          1])
        self.Del_V_vec_seq = jnp.zeros([self.len_seq-1, 2])
        self.cost_seq      = jnp.zeros([self.len_seq])
        self.x_cost_seq    = jnp.zeros([self.len_seq])
        self.u_cost_seq    = jnp.zeros([self.len_seq])        

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

    def validate_input_data(self,ilqr_config:ilqrConfigStruct, seed_x_vec:jnp.ndarray, seed_u_seq:jnp.ndarray):
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
    
    def log_ctrl_state_to_history(self):
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
                                   jnp.ndarray, float, float, jnp.ndarray]:
    # create object containing preconfigured functions with static parameters (currying and parameter encapsulation)

    # populate state sequence from seed control sequence
    x_seq = ilqr_config.simulate_fwd_dyn_seq(ctrl_state.seed_x_vec,ctrl_state.seed_u_seq)

    # calculate initial cost
    cost_float, cost_seq, _, _ = gen_ctrl.calculate_total_cost(ilqr_config.cost_func_for_calc, jnp.array(x_seq), jnp.array(ctrl_state.u_seq))
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
        local_ctrl_state.x_seq, local_ctrl_state.u_seq, local_ctrl_state.cost_float, local_ctrl_state.lsf_float, local_ctrl_state.ro_reg_change_bool = calculate_forwards_pass(ilqr_config, 
                                                                                                                                                                                local_ctrl_state)
        # log iteration output
        if ilqr_config.log_ctrl_history is True and local_ctrl_state.ro_reg_change_bool is False:
            local_ctrl_state.log_ctrl_state_to_history()                  
        # calculate convergence measures

        avg_ff_gain = calculate_avg_ff_gains(local_ctrl_state.d_seq, local_ctrl_state.u_seq)
        converge_measure = jnp.abs((local_ctrl_state.prev_cost_float - local_ctrl_state.cost_float)/ctrl_state.cost_float).item()
        # determine convergence criteria
        converge_reached = determine_convergence(ilqr_config, local_ctrl_state, converge_measure, avg_ff_gain)
        # print controller info
        print('iteration number: ', local_ctrl_state.iter_int)
        print('converge_measure: ', converge_measure)
        print('max_control_effort: ', max(local_ctrl_state.u_seq))
        print('ff_gain_avg_measure: ', avg_ff_gain)        
        print('ro_reg_value: ', local_ctrl_state.ro_reg)
        print('need to increase ro?: ', local_ctrl_state.ro_reg_change_bool)
        local_ctrl_state.cost_float, local_ctrl_state.cost_seq, local_ctrl_state.x_cost_seq, local_ctrl_state.u_cost_seq = gen_ctrl.calculate_total_cost(ilqr_config.cost_func_for_calc, 
                                                                                                                                                        local_ctrl_state.x_seq, 
                                                                                                                                                        local_ctrl_state.u_seq)


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
    # Ad_seq, Bd_seq = ilqr_config.linearize_dyn_seq(ctrl_state.x_seq, ctrl_state.u_seq)    
    dyn_lin_approx_seq   = ilqr_config.linearize_dyn_seq(ctrl_state.x_seq, ctrl_state.u_seq)
    cost_quad_approx_seq = ilqr_config.quad_exp_cost_seq(ctrl_state.x_seq, ctrl_state.u_seq)
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
    xu_seq = jnp.concatenate((ctrl_state.x_seq[:-1],ctrl_state.u_seq), axis=1)
    while is_valid_q_uu is False:
        for idx in range(ctrl_state.len_seq - 1):
            # solve newton-step condition backwards from sequence end
            k = ctrl_state.len_seq - (idx + 2) 
            is_valid_q_uu = True
            q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = taylor_expand_pseudo_hamiltonian(
                                                                        dyn_lin_approx_seq,
                                                                        cost_quad_approx_seq,
                                                                        P_kp1,
                                                                        p_kp1,
                                                                        ro_reg,
                                                                        k)
            # if q_uu_reg is not positive definite, reset loop to initial state and increase ro_reg
            if util.is_pos_def(q_uu_reg) is False:
                is_valid_q_uu = False
                ro_reg    = ro_reg + ilqr_config.ro_reg_change
                P_kp1, p_kp1, k_seq, d_seq, Del_V_vec_seq = initialize_backwards_pass(P_N, p_N, ctrl_state.x_len,
                                                                                      ctrl_state.u_len, ctrl_state.len_seq)
                print('pos_def_fail!')
                break
                
            else:
                K_k,   d_k                = calculate_optimal_gains(q_uu_reg, q_ux_reg, q_u)
                P_kp1, p_kp1, Del_V_vec_k = calculate_backstep_ctg_approx(q_x, q_u, q_xx, q_uu, q_ux, K_k, d_k)
                k_seq = k_seq.at[k].set(K_k)
                d_seq = d_seq.at[k].set(d_k)
                Del_V_vec_seq = Del_V_vec_seq.at[k].set(Del_V_vec_k)  
    return k_seq, d_seq, Del_V_vec_seq, ro_reg

def calculate_forwards_pass(ilqr_config:ilqrConfigStruct, ctrl_state:ilqrControllerState):
    '''lsf is the line serach factor, scaling the size of the newton step'''
    #initialize forward pass state vec, control_vec, cost and cost seqs, line search factor, armijo parameter bool
    x_seq_fp,u_seq_fp,cost_float_fp, cost_seq_fp,x_cost_seq_fp,u_cost_seq_fp,in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
    lsf = 1.0
    line_search_scale_param = ilqr_config.ls_scale_alpha_param
    iter_count = 0
    ro_reg_change_bool = False
    while not in_bounds_bool and (iter_count < ilqr_config.fp_max_iter):
        if ilqr_config.mj_ctrl is True:
            for k in range(ctrl_state.len_seq-1):
                # calculate updated control value
                u_seq_fp = u_seq_fp.at[k].set(calculate_u_k_new(ctrl_state.u_seq[k], 
                                                                ctrl_state.x_seq[k],
                                                                x_seq_fp[k], 
                                                                ctrl_state.K_seq[k], 
                                                                ctrl_state.d_seq[k], 
                                                                lsf))
                x_seq_fp = x_seq_fp.at[k+1].set(ilqr_config.discrete_dyn_mj_func(x_seq_fp[k], u_seq_fp[k]))
            # shift sequences for cost calculation wrt desired trajectories
            # calculate new trajectory cost    
            cost_float_fp, _, _, _ = gen_ctrl.calculate_total_cost(ilqr_config.cost_func_for_calc, jnp.array(x_seq_fp), jnp.array(u_seq_fp))
        else:
            x_seq_fp, u_seq_fp, cost_float_fp = ilqr_config.simulate_forward_pass(ctrl_state.x_seq, ctrl_state.u_seq, ctrl_state.K_seq, ctrl_state.d_seq, lsf)

        # calculate the ratio between the expected cost decrease and actual cost decrease
        cost_decrease_ratio = calculate_cost_decrease_ratio(ctrl_state.cost_float, cost_float_fp, ctrl_state.Del_V_vec_seq, lsf) 
        in_bounds_bool = ilqr_config.analyze_cost_dec(cost_decrease_ratio)

        if in_bounds_bool:
            break    
        elif (iter_count == ilqr_config.fp_max_iter-1):
            iter_count     += 1  
            x_seq_fp   = ctrl_state.x_seq
            u_seq_fp   = ctrl_state.u_seq
            cost_float_fp  = ctrl_state.cost_float
            ro_reg_change_bool = True
        else:        # if decrease is outside of bounds, reinitialize and run pass again with smaller feedforward step
            x_seq_fp,u_seq_fp,cost_float_fp, cost_seq_fp,x_cost_seq_fp,u_cost_seq_fp, in_bounds_bool = initialize_forwards_pass(ctrl_state.x_len, ctrl_state.u_len, 
                                                                                           ctrl_state.seed_x_vec, ctrl_state.len_seq)
            lsf *= line_search_scale_param
            iter_count += 1
    return x_seq_fp, u_seq_fp, cost_float_fp, lsf, ro_reg_change_bool

def simulate_forward_pass(fp_scan_func:Callable[[float, 
                                                 Tuple[int, jnp.ndarray,jnp.ndarray], 
                                                 Tuple[jnp.ndarray, jnp.ndarray,jnp.ndarray,jnp.ndarray]],
                                                 Tuple[int, jnp.ndarray, jnp.ndarray]], 
                            cost_func_for_calc: Callable[[jnp.ndarray, jnp.ndarray, int],jnp.ndarray],                      
                            x_seq:jnp.ndarray, u_seq:jnp.ndarray, 
                            K_seq:jnp.ndarray, d_seq:jnp.ndarray, lsf:float)->Tuple[jnp.ndarray, 
                                                                                    jnp.ndarray, 
                                                                                    float]:
    scan_func_curried = lambda carry, seqs: fp_scan_func(lsf, carry, seqs)
    seqs = (x_seq[:-1], u_seq, K_seq, d_seq)
    carry_init = (int(0),jnp.zeros((1),dtype=float), x_seq[0])
    (_,cost_float,_),(x_seq_new, u_seq_new) = lax.scan(scan_func_curried, carry_init, seqs)
    x_seq_new = jnp.append((x_seq[0]).reshape(1,-1),x_seq_new, axis=0)
    cost_out = cost_float + (cost_func_for_calc(x_seq_new[-1], jnp.zeros((len(u_seq[1]))),  len(x_seq_new)-1))
    return x_seq_new, u_seq_new, cost_out.item()

def scan_func_forward_pass(discrete_dyn_func,
                           cost_func_for_calc,
                           lsf:float, 
                           carry:Tuple[int, float, jnp.ndarray], 
                           seqs:Tuple[jnp.ndarray, jnp.ndarray,jnp.ndarray,jnp.ndarray])->Tuple[
                            Tuple[int, float,jnp.ndarray],
                            Tuple[jnp.ndarray, jnp.ndarray]]:
    x_k, u_k, K_k, d_k    = seqs
    k, cost_float, x_k_new = carry
    u_k_new     = calculate_u_k_new(u_k, x_k, x_k_new, K_k, d_k, lsf)
    cost_float += (cost_func_for_calc(x_k_new, u_k_new, k)).reshape(-1)
    x_kp1_new   = discrete_dyn_func(x_k_new, u_k_new)
    k += 1
    return (k, cost_float, x_kp1_new), (x_kp1_new, u_k_new)

def initialize_backwards_pass(P_N:jnp.ndarray, p_N:jnp.ndarray, x_len:int, u_len:int, len_seq:int):
    K_seq         = jnp.zeros([(len_seq-1), u_len, x_len])
    d_seq         = jnp.zeros([(len_seq-1), u_len, 1])
    Del_V_vec_seq = jnp.zeros([(len_seq-1), 2])
    P_kp1 = P_N
    p_kp1 = p_N
    return P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq

def initialize_forwards_pass(x_len:int, u_len:int, seed_state_vec:jnp.ndarray, len_seq:int):
    control_seq_fp  = jnp.zeros([(len_seq-1),u_len])
    x_seq_fp        = jnp.zeros([(len_seq),  x_len])
    x_seq_fp.at[0].set(seed_state_vec)
    cost_float_fp   = 0.0
    cost_seq_fp     = jnp.zeros([len_seq])
    x_cost_seq_fp   = jnp.zeros([len_seq])   
    u_cost_seq_fp   = jnp.zeros([len_seq]) 
    in_bounds_bool  = False
    return x_seq_fp, control_seq_fp, cost_float_fp, cost_seq_fp, x_cost_seq_fp,u_cost_seq_fp,in_bounds_bool

def taylor_expand_pseudo_hamiltonian(dyn_lin_approx_k: Tuple[jnp.ndarray, jnp.ndarray],
                                     cost_quad_approx_k: Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray,jnp.ndarray,jnp.ndarray], 
                                     P_kp1:jnp.ndarray, 
                                     p_kp1:jnp.ndarray, 
                                     ro_reg:float, 
                                     k:int):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
#  https://alkzar.cl/blog/2022-01-13-taylor-approximation-and-jax/
# q_xx = l_xx + A^T P' A
# q_uu = l_uu + B^T P' B
# q_ux = l_ux + B^T P' A
# q_x  = l_x  + A^T p'
# q_u  = l_u  + B^T p'
    A_lin_k = (dyn_lin_approx_k[0])[k]
    B_lin_k = (dyn_lin_approx_k[1])[k]
    l_x     = (cost_quad_approx_k[0])[k]
    l_u     = (cost_quad_approx_k[1])[k]
    l_xx    = (cost_quad_approx_k[2])[k]
    l_uu    = (cost_quad_approx_k[3])[k]
    l_ux    = (cost_quad_approx_k[4])[k]
    if ro_reg > 0:
        P_kp1_reg = util.reqularize_mat(P_kp1, ro_reg)
    else:
        P_kp1_reg = P_kp1    
    q_x      = (l_x  + A_lin_k.T @ p_kp1)
    q_u      = (l_u  + B_lin_k.T @ p_kp1)
    q_xx     = (l_xx + A_lin_k.T @ P_kp1     @ A_lin_k)
    q_uu     = (l_uu + B_lin_k.T @ P_kp1     @ B_lin_k)
    q_ux     = (l_ux + B_lin_k.T @ P_kp1     @ A_lin_k)
    q_uu_reg = (l_uu + B_lin_k.T @ P_kp1_reg @ B_lin_k)
    q_ux_reg = (l_ux + B_lin_k.T @ P_kp1_reg @ A_lin_k)
    return q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg

def calculate_final_ctg_approx(cost_func_for_diff:Callable[[jnp.ndarray, jnp.ndarray, int],jnp.ndarray], 
                                x_N:jnp.ndarray, u_len:int, len_seq:int) -> Tuple[jnp.ndarray,jnp.ndarray]:
    # approximates the final step cost-to-go as only the state with no control input
    # create concatenated state and control vector of primal points
    u_N = np.zeros([u_len], dtype=float)
    xu_N = jnp.concatenate((x_N, u_N))
    k_step = len_seq - 1
    p_N, _, P_N, _, _ = gen_ctrl.taylor_expand_cost(cost_func_for_diff, xu_N,len(x_N), k_step)
    return P_N, p_N

def calculate_backstep_ctg_approx(q_x:jnp.ndarray, 
                                  q_u:jnp.ndarray,
                                  q_xx:jnp.ndarray,
                                  q_uu:jnp.ndarray,
                                  q_ux:jnp.ndarray,
                                  K_k:jnp.ndarray, 
                                  d_k:jnp.ndarray) -> Tuple[jnp.ndarray,
                                                            jnp.ndarray,
                                                            jnp.ndarray]:
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
    Del_V_vec_k = jnp.array([(d_k.T @ q_u),((0.5) * (d_k.T @ q_uu @ d_k))]).reshape(-1) 
    return P_k, p_k, Del_V_vec_k

def calculate_optimal_gains(q_uu_reg:jnp.ndarray, 
                            q_ux:jnp.ndarray, 
                            q_u:jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
    # K_k is a linear feedback term related to del_x from nominal trajectory
    # d_k is a feedforward term related to trajectory correction
    inverted_q_uu = jnp.linalg.inv(q_uu_reg)
    K_k           = -inverted_q_uu @ q_ux
    d_k           = -inverted_q_uu @ q_u
    return K_k, d_k

def calculate_u_k_new(u_nom_k:jnp.ndarray,
                      x_nom_k:jnp.ndarray,
                      x_new_k:jnp.ndarray, 
                      K_k:jnp.ndarray, 
                      d_k:jnp.ndarray, line_search_factor:float) -> jnp.ndarray:
    # check valid dimensions
    assert K_k.shape    == (len(u_nom_k), len(x_nom_k)), "K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))"
    assert d_k.shape    == (len(u_nom_k), 1)           , "d_k incorrect shape, must be (len(u_nom_k), 1)"
    assert len(x_new_k) ==  len(x_nom_k)               , "x_nom_k and x_new_k must be the same length"
    assert (x_nom_k.shape) == (len(x_nom_k),), 'x_nom_k must be column vector (n,)'
    assert (x_new_k.shape) == (len(x_new_k),), 'x_new_k must be column vector (n,)'    
    assert (u_nom_k.shape) == (len(u_nom_k),), 'u_nom_k must be column vector (m,)'    
    return (u_nom_k.reshape(-1,1) + K_k @ (x_new_k.reshape(-1,1) - x_nom_k.reshape(-1,1)) + line_search_factor * d_k).reshape(-1)

def calculate_cost_decrease_ratio(prev_cost_float:float, new_cost_float:float, Del_V_vec_seq:jnp.ndarray, lsf:float) -> float:
    """
    Calculates the ratio between the expected cost decrease from approximated dynamics and cost functions, and the calculated
    cost from the forward pass shooting method. The cost decrease is expected to be positive, as the Del_V_sum should be a negative
    expected value, and the new cost should be smaller than the previous cost.
    """
    Del_V_sum = calculate_expected_cost_decrease(Del_V_vec_seq, lsf)
    cost_decrease_ratio = (1 / Del_V_sum) * (new_cost_float - prev_cost_float)
    return cost_decrease_ratio

def calculate_expected_cost_decrease(Del_V_vec_seq:jnp.ndarray, lsf:float) -> float:
    Del_V_sum = jnp.sum((lsf * Del_V_vec_seq[:,0]) + (lsf**2 * Del_V_vec_seq[:,1]))
    return Del_V_sum.item()    

def analyze_cost_decrease(cost_decrease_ratio:float, cost_ratio_bounds:Tuple[float, float]) -> bool:
    if (cost_decrease_ratio > cost_ratio_bounds[0]) and (cost_decrease_ratio < cost_ratio_bounds[1]):
        return True
    else:
        return False            

def calculate_avg_ff_gains(d_seq:jnp.ndarray, u_seq:jnp.ndarray) -> float:
    norm_factor:float   = 1 / (len(u_seq) - 1)
    norm_gain_sum:float = jnp.sum(jnp.abs(d_seq.reshape(-1).max()) / (jnp.linalg.norm(u_seq, axis=1)+1)).item()  
    return norm_factor * norm_gain_sum


def calculate_u_star(k_fb_k:jnp.ndarray, 
                     u_des_k:jnp.ndarray, 
                     x_des_k:jnp.ndarray,
                     x_k:jnp.ndarray) -> jnp.ndarray:
    return (u_des_k.reshape(-1,1) + k_fb_k @ (x_k.reshape(-1,1) - x_des_k.reshape(-1,1))).reshape(-1)


def simulate_ilqr_output(sim_dyn_func_step:Callable[[jnp.ndarray,jnp.ndarray], jnp.ndarray], 
    ctrl_out:ilqrControllerState, x_sim_init:jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
    x_sim_seq  = jnp.zeros([ctrl_out.len_seq, ctrl_out.x_len])
    u_sim_seq  = jnp.zeros([ctrl_out.len_seq-1, ctrl_out.u_len])
    x_sim_seq  = x_sim_seq.at[0].set(x_sim_init)    
    for k in range(ctrl_out.len_seq-1):
        u_sim_seq = u_sim_seq.at[k].set(calculate_u_star(ctrl_out.K_seq[k], ctrl_out.u_seq[k], ctrl_out.x_seq[k], x_sim_seq[k]))
        x_sim_seq = x_sim_seq.at[k+1].set(sim_dyn_func_step(x_sim_seq[k], u_sim_seq[k]))
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
