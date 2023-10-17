import unittest
import numpy as np
from jax import numpy as jnp

import src.ilqr_funcs as ilqr
import src.ilqr_utils as util
import src.gen_ctrl_funcs as gen_ctrl

# for each input
#   test one known case
#   if discrete change of behavior (abs value add, check pos and neg)
#   check edge cases (NaN, overflow, string, divide by zero)

class shared_unit_test_data_and_funcs:
    def __init__(self) -> None:
        self.pend_unit_cost_func_params = {
                        'Q'  : np.array([[1.,0],[0,1.]]), # type: ignore
                        'R'  : np.array([[1.]]),
                        'Qf' : np.array([[1.,0],[0,1.]]) # type: ignore
                      } 
        self.pend_cost_func_params = {
                        'Q'  : np.array([[1.,0],[0,1.]]), # type: ignore
                        'R'  : np.array([[0.001]]),
                        'Qf' : np.array([[100.,0],[0,100.]]) # type: ignore
                      } 
        self.pend_unit_state_trans_params = {
                        'b'  : 1.0,
                        'l'  : 1.0,
                        'g'  : 1.0
                      } 
        self.gen_cost_func_params = {
                        'Q'  : np.array([[1.,0,0],[0,10.,0],[0,0,1.]]), # type: ignore
                        'R'  : np.array([[1.,0],[0,10.]]), # type: ignore
                        'Qf' : np.array([[1.,0,0],[0,10.,0],[0,0,10.]]) # type: ignore
                      } 
        self.gen_state_trans_params = {
                        'b'  : 1.0,
                        'l'  : 2.0,
                        'g'  : 3.0
                      } 
        self.ilqr_config_lin_pend_unit_cost = {
            'state_trans_func'          : self.pend_lin_dyn_func_full,
            'state_trans_func_params'   : self.pend_unit_state_trans_params,
            'cost_func'                 : self.cost_func_quad_state_and_control_full,
            'cost_func_params'          : self.pend_unit_cost_func_params,
            'sim_method'                : 'solve_ivp_zoh',      # 'euler', "solve_ivp_zoh"
            'c2d_method'                : 'zoh',      # 'euler', "zoh", 'zohCombined'
            'max_iter'                  : 20,
            'time_step'                 : 0.1,
            'converge_crit'             : 1e-5,
            'ff_gain_tol'               : 1e-5,
            'cost_ratio_bounds'         : [1e-6, 20],
            'ro_reg_start'              : 0.25,
            'ro_reg_change'             : 0.25,
            'fp_max_iter'               : 5,
            'log_ctrl_history'          : True
            }
        self.ilqr_config_nl_pend = {
            'state_trans_func'          : self.pend_dyn_func_full,
            'state_trans_func_params'   : self.pend_unit_state_trans_params,
            'cost_func'                 : self.cost_func_quad_state_and_control_full,
            'cost_func_params'          : self.pend_cost_func_params,
            'sim_method'                : 'solve_ivp_zoh',      # 'euler', "solve_ivp_zoh"
            'c2d_method'                : 'zoh',      # 'euler', "zoh", 'zohCombined'
            'max_iter'                  : 20,
            'time_step'                 : 0.1,
            'converge_crit'             : 1e-5,
            'ff_gain_tol'               : 1e-5,            
            'cost_ratio_bounds'         : [1e-6, 20],
            'ro_reg_start'              : 0.25,
            'ro_reg_change'             : 0.25,
            'fp_max_iter'               : 5,
            'log_ctrl_history'          : True
            }
        
        self.len_seq          = 10
        self.seed_x_vec       = np.ones([2,1]) 
        self.seed_u_seq       = np.ones([self.len_seq-1,1,1])

        self.x_len            = len(self.seed_x_vec)
        self.u_len            = len(self.seed_u_seq[0])

        self.x_seq            = np.ones([self.len_seq,2,1])
        self.x_seq[:,1]      *= 2.0
        self.u_seq            = np.ones([self.len_seq-1,1,1])

        self.x_des_seq        = np.zeros([self.len_seq, 2,1])
        self.u_des_seq        = np.zeros([self.len_seq-1, 1,1])

        self.K_seq            = np.ones([self.len_seq, self.u_len, self.x_len])
        self.d_seq            = np.ones([self.len_seq, self.u_len, 1         ])
        self.Del_V_vec_seq    = np.ones([self.len_seq, 1,          2         ])

        self.time_step        = self.ilqr_config_lin_pend_unit_cost['time_step']
        self.time_seq         = np.arange(self.len_seq) * self.time_step

    def gen_dyn_func_full(self, x, u, t=None):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y
    
    def gen_dyn_func_curried(self, x, u, t=None):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y
    
    def pend_dyn_func_full(self, params, time, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        state.reshape(-1)
        g = params['g']
        l = params['l']
        b = params['b']
        state_dot = jnp.array([
                    state[1],
                    -(b/l) * state[1] - (jnp.sin(state[0]) * g/l) + control[0]
                    ])
        return state_dot
    
    def pend_unit_dyn_func_curried(self, time, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        params = self.pend_unit_state_trans_params
        # TODO assert dimensions of incoming vectors
        state.reshape(-1)
        g = params['g']
        l = params['l']
        b = params['b']
        pos = state[0]
        vel = state[1]
        tq  = control[0]
        state_dot = jnp.array([
                    vel,
                    -(b/l) * vel - (jnp.sin(pos) * g/l) + tq
                    ])
        return state_dot  

    def pend_lin_dyn_func_full(self, params, time, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        g = params['g']
        l = params['l']
        b = params['b']
        state_dot = jnp.array([
                    state[1],
                    -((b/l) * state[1]) - ((state[0]) * (g/l)) + control[0]
                    ])
        return state_dot
    
    def pend_unit_lin_dyn_func_curried(self, time, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([
                    state[1],
                    -((b/l) * state[1]) - ((state[0]) * (g/l)) + control[0]
                    ])
        return state_dot
    
    def cost_func_quad_state_and_control_full(self,cost_func_params:dict, state_vec, control_vec, k_step, state_des_seq, control_des_seq, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array(cost_func_params['Q'])
        R  = jnp.array(cost_func_params['R'])
        Qf = jnp.array(cost_func_params['Qf'])
        # check that dimensions match [TODO]
        x_k_corr     = util.calc_and_shape_array_diff(state_vec, state_des_seq[k_step])
        x_k_corr_col = util.array_to_col(x_k_corr)
        if is_final_bool:
            cost_float = (0.5) * (jnp.transpose(x_k_corr_col)  @ Qf @ x_k_corr_col)
        else:
            u_k_corr     = util.calc_and_shape_array_diff(control_vec, control_des_seq[k_step])
            u_k_corr_col = util.array_to_col(u_k_corr)
            cost_float = (0.5) * ( (jnp.transpose(x_k_corr_col) @ Q @ x_k_corr_col)
                                  +(jnp.transpose(u_k_corr_col) @ R @ u_k_corr_col))
        return cost_float      

    def unit_cost_func_quad_state_and_control_pend_curried(self, x_k, u_k, k_step, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        cost_func         = self.cost_func_quad_state_and_control_full
        cost_func_params  = self.ilqr_config_lin_pend_unit_cost['cost_func_params']
        state_des_seq     = self.x_des_seq
        control_des_seq   = self.u_des_seq
        cost_func_curried = lambda x, u, k, is_final_bool=False: cost_func(cost_func_params, x, u, k, state_des_seq, control_des_seq, is_final_bool)
        cost_float = cost_func_curried(x_k, u_k, k_step, is_final_bool)
        return cost_float   
    
class stateSpace_tests(unittest.TestCase):
    def test_state_space_class_accepts_valid_continuous_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss = gen_ctrl.stateSpace(A,B,C,D)
        self.assertEqual(ss.a.tolist(), A.tolist())
        self.assertEqual(ss.b.tolist(), B.tolist())
        self.assertEqual(ss.c.tolist(), C.tolist())
        self.assertEqual(ss.d.tolist(), D.tolist())
        self.assertEqual(ss.type, 'continuous')
        self.assertEqual(ss.time_step, None)

    def test_state_space_class_accepts_valid_discrete_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss = gen_ctrl.stateSpace(A,B,C,D, time_step = 0.1)
        self.assertEqual(ss.a.tolist(), A.tolist())
        self.assertEqual(ss.b.tolist(), B.tolist())
        self.assertEqual(ss.c.tolist(), C.tolist())
        self.assertEqual(ss.d.tolist(), D.tolist())
        self.assertEqual(ss.type, 'discrete')
        self.assertEqual(ss.time_step, 0.1)

    def test_state_space_rejects_nonsquare_A_matrix(self):
        A = jnp.array([[1,1,1],[0,1,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = gen_ctrl.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A matrix must be square')

    def test_state_space_rejects_A_and_B_matrix_n_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = gen_ctrl.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A and B matrices must have same n(state) dimension')        

    def test_state_space_rejects_A_and_C_matrix_m_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = gen_ctrl.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A and C matrices must have same m(state) dimension')        

    def test_state_space_rejects_B_and_D_matrix_m_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = gen_ctrl.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'B and D matrices must have the same m(control) dimension')  
 
    def test_state_space_rejects_C_and_D_matrix_n_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,],[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = gen_ctrl.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'C and D matrices must have the same n(measurement) dimension')  

    def test_state_space_rejects_invalid_timestep_input(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,]])
        with self.assertRaises(Exception) as assert_error_negative:
            ss = gen_ctrl.stateSpace(A,B,C,D, time_step = -0.1) 
        self.assertEqual(str(assert_error_negative.exception), 'invalid time step definition -0.1. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_string:
            ss = gen_ctrl.stateSpace(A,B,C,D, time_step = 'lettuce') 
        self.assertEqual(str(assert_error_string.exception), 'invalid time step definition lettuce. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_zero:
            ss = gen_ctrl.stateSpace(A,B,C,D, time_step = 0) 
        self.assertEqual(str(assert_error_zero.exception), 'invalid time step definition 0. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_NaN:
            ss = gen_ctrl.stateSpace(A,B,C,D, time_step = float("nan")) 
        self.assertEqual(str(assert_error_NaN.exception), 'invalid time step definition nan. time_step must be a positive float or None')      

class ilqrConfiguredFuncs_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        self.assertEqual(True,True)  

class ilqrControllerState_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        self.assertEqual(True,True)

class initialize_ilqr_controller_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config_lin_pend_unit_cost
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        config_funcs, x_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state) 
        self.assertEqual((np.abs(cost_float-prev_cost_float) > ilqr_config['converge_crit']), True)   

class run_ilqr_controller_tests(unittest.TestCase):
    def test_accepts_valid_linear_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config_lin_pend_unit_cost
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        config_funcs, state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        ctrl_state.x_seq = state_seq
        ctrl_state.cost_float = cost_float
        ctrl_state.prev_cost_float = prev_cost_float
        ctrl_state.cost_seq = cost_seq
        ctrl_state = ilqr.run_ilqr_controller(ilqr_config, config_funcs, ctrl_state)
        self.assertEqual(True,True)

    def test_accepts_valid_nl_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config_nl_pend
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_nl_pend,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        config_funcs, state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        ctrl_state.x_seq = state_seq
        ctrl_state.cost_float = cost_float
        ctrl_state.prev_cost_float = prev_cost_float
        ctrl_state = ilqr.run_ilqr_controller(ilqr_config, config_funcs, ctrl_state)
        self.assertEqual(True,True)

class calculate_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_linear_dyn_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        ctrl_state.x_seq = data_and_funcs.x_seq
        config_funcs = ilqr.ilqrConfiguredFuncs(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                                ctrl_state)
        
        k_seq, d_seq, Del_V_vec_seq, ro_reg = ilqr.calculate_backwards_pass(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                                                            config_funcs, ctrl_state)
        self.assertEqual(len(k_seq),data_and_funcs.len_seq-1)
        self.assertEqual(len(d_seq),data_and_funcs.len_seq-1)
        self.assertEqual(len(Del_V_vec_seq),data_and_funcs.len_seq-1) 
        self.assertEqual(jnp.shape(k_seq[0]),(data_and_funcs.u_len, data_and_funcs.x_len))                  
        self.assertEqual(jnp.shape(d_seq[0]),(data_and_funcs.u_len, 1))                       
        self.assertEqual(jnp.shape(Del_V_vec_seq[0]),(2,)) 

class calculate_forwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_linear_dyn_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config_lin_pend_unit_cost
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_lin_pend_unit_cost,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        config_funcs, state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        ctrl_state.x_seq = state_seq
        ctrl_state.cost_float = cost_float
        ctrl_state.prev_cost_float = prev_cost_float
        k_seq, d_seq, Del_V_vec_seq, ro_reg = ilqr.calculate_backwards_pass(ilqr_config, config_funcs, ctrl_state)
        ctrl_state.K_seq         = k_seq
        ctrl_state.d_seq         = d_seq
        ctrl_state.Del_V_vec_seq = Del_V_vec_seq                
        # ctrl_state.K_seq         = data_and_funcs.K_seq
        # ctrl_state.d_seq         = data_and_funcs.d_seq
        # ctrl_state.Del_V_vec_seq = data_and_funcs.Del_V_vec_seq
        state_seq_new, control_seq_new, cost_float_new,cost_seq_new,line_search_factor, ro_reg_change_bool = ilqr.calculate_forwards_pass(ilqr_config, config_funcs, ctrl_state)
        self.assertEqual(True,True)

    def test_accepts_valid_nl_dyn_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config_lin_pend_unit_cost
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config_nl_pend,
                                              data_and_funcs.seed_x_vec,
                                              data_and_funcs.u_seq,
                                              data_and_funcs.x_des_seq,
                                              data_and_funcs.u_des_seq)
        config_funcs, state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        ctrl_state.x_seq = state_seq
        ctrl_state.cost_float = cost_float
        ctrl_state.prev_cost_float = prev_cost_float
        k_seq, d_seq, Del_V_vec_seq, ro_reg = ilqr.calculate_backwards_pass(ilqr_config, config_funcs, ctrl_state)
        ctrl_state.K_seq         = k_seq
        ctrl_state.d_seq         = d_seq
        ctrl_state.Del_V_vec_seq = Del_V_vec_seq                
        # ctrl_state.K_seq         = data_and_funcs.K_seq
        # ctrl_state.d_seq         = data_and_funcs.d_seq
        # ctrl_state.Del_V_vec_seq = data_and_funcs.Del_V_vec_seq
        state_seq_new, control_seq_new, cost_float_new,cost_seq_new,line_search_factor, ro_reg_change_bool = ilqr.calculate_forwards_pass(ilqr_config, config_funcs, ctrl_state)
        self.assertEqual(True,True)

class simulate_forward_dynamics_tests(unittest.TestCase):
    def pend_unit_dyn_func(self, time, state, control):
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([
                    x_k_jax[1],
                    -(b/l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g/l) + u_k_jax[0]
                    ])
        return state_dot  

    def test_accepts_valid_inputs_euler(self):
        len_seq         = 4
        time_step       = 0.1
        control_seq     = np.ones([4,1,1])
        time_step       = 0.1
        state_init      = np.array([[1.],[1.]])
        state_seq       = gen_ctrl.simulate_forward_dynamics_seq(self.pend_unit_dyn_func, 
                                                         state_init, control_seq, time_step, 
                                                         sim_method = 'euler')
        # calculate expected output
        time_seq = [0,0.1,0.2,0.3]
        state_seq_expected    = np.zeros([5,2,1])
        state_seq_expected[0] = state_init
        for idx in range(len_seq):
            state_dot_row    = self.pend_unit_dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next       = state_seq[idx] + state_dot_row * time_step
            state_seq_expected[idx+1] = state_next
        self.assertEqual(len(state_seq), len_seq +1)
        self.assertEqual(state_seq[-1].tolist(), state_seq_expected[-1].tolist())

    def test_accepts_valid_inputs_euler_from_shared(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        dyn_func        = data_and_funcs.pend_unit_dyn_func_curried
        time_step       = data_and_funcs.time_step
        control_seq     = data_and_funcs.u_seq
        state_init      = data_and_funcs.seed_x_vec
        x_len           = len(data_and_funcs.x_seq[0])
        u_len           = len(data_and_funcs.u_seq[0])
        len_seq         = data_and_funcs.len_seq
        state_seq       = gen_ctrl.simulate_forward_dynamics_seq(dyn_func, state_init, 
                                                         control_seq, time_step, 
                                                         sim_method = 'euler')
        self.assertEqual(len(state_seq), len_seq)

    def test_accepts_valid_inputs_solve_ivp_zoh(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        dyn_func        = data_and_funcs.pend_unit_dyn_func_curried
        time_step       = data_and_funcs.time_step
        control_seq     = data_and_funcs.u_seq
        state_init      = data_and_funcs.seed_x_vec
        x_len           = len(data_and_funcs.x_seq[0])
        u_len           = len(data_and_funcs.u_seq[0])
        len_seq         = data_and_funcs.len_seq
        state_seq       = gen_ctrl.simulate_forward_dynamics_seq(dyn_func, state_init, 
                                                         control_seq, time_step, 
                                                         sim_method = 'solve_ivp_zoh')
        self.assertEqual(len(state_seq), len_seq)
        self.assertEqual(np.shape(state_seq[0]), (x_len,1))        
        self.assertEqual(True,True)        

class linearize_dynamics_tests(unittest.TestCase):

    def test_linearize_dynamics_accepts_valid_case(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        time_seq        = data_and_funcs.time_seq
        control_seq     = data_and_funcs.u_seq
        state_seq       = data_and_funcs.x_seq
        x_len           = data_and_funcs.x_len
        u_len           = data_and_funcs.u_len
        k_step          = 1
        a_lin, b_lin    = gen_ctrl.linearize_dynamics(data_and_funcs.pend_unit_dyn_func_curried,
                                                  time_seq[k_step],
                                                  state_seq[k_step],
                                                  control_seq[k_step])
        
        g = data_and_funcs.pend_unit_state_trans_params['g']
        l = data_and_funcs.pend_unit_state_trans_params['l']
        b = data_and_funcs.pend_unit_state_trans_params['b']
        state_pos = state_seq[k_step][0][0]
        a_lin_expected = np.array([[0, 1], 
                                    [-(g/l)*jnp.cos(state_pos), -(b/l)]])
        b_lin_expected = np.array([[0],[1]])
        # print('a_lin: ', a_lin)
        # print('a_lin_expected: ', a_lin_expected)
        # print('b_lin: ', b_lin)
        # print('b_lin_expected: ', b_lin_expected)
        self.assertEqual(jnp.shape(a_lin), (x_len, x_len))
        self.assertEqual(jnp.shape(b_lin), (x_len, u_len))
        self.assertEqual(a_lin.tolist(), a_lin_expected.tolist())
        self.assertEqual(b_lin.tolist(), b_lin_expected.tolist())

    def test_linearize_dynamics_accepts_valid_linear_case(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        time_seq        = data_and_funcs.time_seq
        control_seq     = data_and_funcs.u_seq
        state_seq       = data_and_funcs.x_seq
        x_len           = data_and_funcs.x_len
        u_len           = data_and_funcs.u_len
        k_step          = 1
        a_lin, b_lin    = gen_ctrl.linearize_dynamics(data_and_funcs.pend_unit_lin_dyn_func_curried,
                                                  time_seq[k_step],
                                                  state_seq[k_step],
                                                  control_seq[k_step])
        
        g = data_and_funcs.pend_unit_state_trans_params['g']
        l = data_and_funcs.pend_unit_state_trans_params['l']
        b = data_and_funcs.pend_unit_state_trans_params['b']
        state_pos = state_seq[k_step][0][0]
        a_lin_expected = jnp.array([[0, 1], 
                                    [-(g/l), -(b/l)]])
        b_lin_expected = jnp.array([[0],[1]])
        # print('a_lin: ', a_lin)
        # print('a_lin_expected: ', a_lin_expected)
        # print('b_lin: ', b_lin)
        # print('b_lin_expected: ', b_lin_expected)
        self.assertEqual(jnp.shape(a_lin), (x_len, x_len))
        self.assertEqual(jnp.shape(b_lin), (x_len, u_len))
        self.assertEqual(a_lin.tolist(), a_lin_expected.tolist())
        self.assertEqual(b_lin.tolist(), b_lin_expected.tolist())

class calculate_linearized_state_space_seq_tests(unittest.TestCase):

    def discretize_ss_simple(self, input_state_space:gen_ctrl.stateSpace):
        time_step = 0.1
        n  = input_state_space.a.shape[0] 
        a_d = jnp.eye(n) + (input_state_space.a * time_step)
        b_d = input_state_space.b * time_step
        c_d = input_state_space.c
        d_d = input_state_space.d
        d_state_space = gen_ctrl.stateSpace(a_d, b_d, c_d, d_d, time_step)
        return d_state_space
    
    def pend_unit_dyn_func(self, time, state, control):
        state.reshape(-1)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([
                    state[1],
                    -(b/l) * state[1] - (jnp.sin(state[0]) * g/l) + control[0]
                    ])
        return state_dot     

    def test_calculate_linearized_state_space_seq_accepts_valid_system(self):
        time_seq       = [0.1, 0.2, 0.3]
        len_seq        = 3
        time_step      = 0.1
        control_seq    = [jnp.array([[1.]])] * (len_seq-1)
        state_seq      = [jnp.array([[1.],[1.]])] * (len_seq)
        A_lin_array, B_lin_array = gen_ctrl.calculate_linearized_state_space_seq(self.pend_unit_dyn_func,
                                                                             self.discretize_ss_simple, 
                                                                             state_seq, 
                                                                             control_seq, 
                                                                             time_seq)
        # calculate expected output
        state_pos = state_seq[0][0]
        a_lin_expected = jnp.array([[0, 1], 
                                    [-(1./1.) * jnp.cos(state_pos[0]), -(1./1.)]])
        b_lin_expected = jnp.array([[0],[1]])
        A_lin_d_expected = jnp.eye(2) + (a_lin_expected * time_step)
        B_lin_d_expected = b_lin_expected * time_step
        #assert validity
        self.assertEqual(len(A_lin_array), len_seq-1)
        self.assertEqual(len(B_lin_array), len_seq-1)
        self.assertEqual((A_lin_array[0]).tolist(), A_lin_d_expected.tolist()) # type: ignore
        self.assertEqual((B_lin_array[0]).tolist(), B_lin_d_expected.tolist())         # type: ignore

    def test_calculate_linearized_state_space_seq_accepts_valid_system_from_shared(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        time_seq      = data_and_funcs.time_seq
        len_seq        = data_and_funcs.len_seq
        x_len          = len(data_and_funcs.x_seq[0])
        u_len          = len(data_and_funcs.u_seq[0])
        control_seq    = data_and_funcs.u_seq
        state_seq      = data_and_funcs.x_seq
        A_lin_array, B_lin_array = gen_ctrl.calculate_linearized_state_space_seq(data_and_funcs.pend_unit_dyn_func_curried,
                                                                             self.discretize_ss_simple, 
                                                                             state_seq, 
                                                                             control_seq, 
                                                                             time_seq)
        self.assertEqual(len(A_lin_array), len_seq-1)
        self.assertEqual(len(B_lin_array), len_seq-1)
    # def test_calculate_linearized_state_space_seq_rejects_invalid_sequence_dimensions(self):
    #     time_step    = 0.1
    #     num_steps    = 20
    #     num_states   = 3
    #     num_controls = 2
    #     control_seq  = jnp.ones((num_steps, num_controls, 2))
    #     state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
    #     with self.assertRaises(Exception) as assert_seq_dim_error:
    #         A_lin_array, B_lin_array = gen_ctrl.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_dim_error.exception), 'state or control sequence is incorrect array dimension. sequences must be 2d arrays')

    # def test_calculate_linearized_state_space_seq_rejects_invalid_sequence_lengths(self):
    #     time_step    = 0.1
    #     num_steps    = 20
    #     num_states   = 3
    #     num_controls = 2
    #     control_seq  = jnp.ones((num_steps, num_controls))
    #     state_seq    = jnp.ones((num_steps+2, num_states)) * 0.1
    #     with self.assertRaises(Exception) as assert_seq_len_error:
    #         A_lin_array, B_lin_array = gen_ctrl.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_len_error.exception), 'state and control sequences are incompatible lengths. state seq must be control seq length +1')

    # def test_calculate_linearized_state_space_seq_rejects_invalid_time_step(self):
    #     time_step    = -0.1
    #     num_steps    = 20
    #     num_states   = 3
    #     num_controls = 2
    #     control_seq  = jnp.ones((num_steps, num_controls))
    #     state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
    #     with self.assertRaises(Exception) as assert_seq_len_error:
    #         A_lin_array, B_lin_array = gen_ctrl.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_len_error.exception), 'time_step is invalid. Must be positive float')

class discretize_state_space_tests(unittest.TestCase):

    def test_discretize_state_space_accepts_valid_system(self):
        A = np.array([[1,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        ss_c = gen_ctrl.stateSpace(A,B,C,D)
        time_step = 0.1
        c2d_methods = ['euler', 'zoh', 'zohCombined']
        for c2d_method  in c2d_methods:
            ss = gen_ctrl.discretize_state_space(ss_c,time_step,c2d_method)
            self.assertEqual(ss.a.shape, A.shape)
            self.assertEqual(ss.b.shape, B.shape)
            self.assertEqual(ss.c.shape, C.shape)
            self.assertEqual(ss.d.shape, D.shape)
            self.assertEqual(ss.time_step, time_step)
            self.assertEqual(ss.type, 'discrete')

    def test_discretize_state_space_returns_correct_euler_discretization(self):
        A = np.array([[1.,1.,1.],[0.,1.,1.],[0.,0.,1.]])
        B = np.array([[1.,1.],[0.,1.],[0.,0.]])
        C = np.array([[1.,0.,1.]])
        D = np.array([[0.,0.]])
        ss_c = gen_ctrl.stateSpace(A,B,C,D)
        time_step = 0.1
        ss = gen_ctrl.discretize_state_space(ss_c,time_step,c2d_method='euler')
        A_d_expected = np.eye(3) + (A * time_step)
        B_d_expected = B * time_step
        C_d_expected = C
        D_d_expected = D
        self.assertEqual(ss.a.shape, A.shape)
        self.assertEqual(ss.b.shape, B.shape)
        self.assertEqual(ss.c.shape, C.shape)
        self.assertEqual(ss.d.shape, D.shape)
        self.assertEqual(ss.time_step, time_step)
        self.assertEqual(ss.type, 'discrete')  
        self.assertEqual(ss.a.tolist(), A_d_expected.tolist())            
        self.assertEqual(ss.b.tolist(), B_d_expected.tolist())        
        self.assertEqual(ss.c.tolist(), C_d_expected.tolist())
        self.assertEqual(ss.d.tolist(), D_d_expected.tolist())

    def test_descritize_state_space_rejects_discrete_input_system(self):
        A = np.array([[1,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        time_step = 0.1   
        ss_d = gen_ctrl.stateSpace(A,B,C,D, time_step)
        with self.assertRaises(Exception) as assert_error_discrete:
            ss = gen_ctrl.discretize_state_space(ss_d,time_step)
        self.assertEqual(str(assert_error_discrete.exception), 'input state space is already discrete')
    
    def test_descritize_state_space_rejects_invalid_c2d_method(self):
        A = np.array([[1,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        time_step = 0.1   
        ss_d = gen_ctrl.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_c2d_method:
            ss = gen_ctrl.discretize_state_space(ss_d,time_step, c2d_method = 'lettuce')
        self.assertEqual(str(assert_error_c2d_method.exception), 'invalid discretization method')
     
    def test_descritize_state_space_rejects_singular_A_matrix_for_zoh(self):
        A = np.array([[0,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        time_step = 0.1   
        ss_d = gen_ctrl.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_singular_A_zoh:
            ss = gen_ctrl.discretize_state_space(ss_d,time_step, c2d_method = 'zoh')
        self.assertEqual(str(assert_error_singular_A_zoh.exception), 'determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid'
)

class initialize_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        len_seq = data_and_funcs.len_seq
        x_len   = data_and_funcs.x_len
        u_len   = data_and_funcs.u_len
        p_N     = np.ones([x_len, 1])
        P_N     = np.ones([x_len, x_len])
        P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq = ilqr.initialize_backwards_pass(P_N, p_N, x_len, u_len, len_seq)
        self.assertEqual(P_kp1.shape, (x_len, x_len))
        self.assertEqual(p_kp1.shape, (x_len, 1))
        self.assertEqual(K_seq.shape, ((len_seq-1), u_len, x_len))           
        self.assertEqual(d_seq.shape, ((len_seq-1), u_len, 1))        
        self.assertEqual(Del_V_vec_seq.shape, (len_seq-1, 2)) 

class initialize_forwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        seed_state_vec    = shared_data_funcs.seed_x_vec
        u_len             = shared_data_funcs.u_len
        x_len             = shared_data_funcs.x_len
        len_seq           = shared_data_funcs.len_seq
        state_seq_updated, control_seq_updated, cost_float_updated, cost_seq_updated, in_bounds_bool = ilqr.initialize_forwards_pass(x_len, u_len, seed_state_vec, len_seq)
        self.assertEqual(len(state_seq_updated), len_seq)
        self.assertEqual(len(control_seq_updated), len_seq-1)
        self.assertEqual(cost_float_updated, 0)
        self.assertEqual(in_bounds_bool, False)

class calculate_total_cost_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = np.ones([3,2,1])
        u_seq = jnp.ones([3,1,1])
        total_cost, cost_seq = gen_ctrl.calculate_total_cost(cost_func, x_seq, u_seq)
        total_cost_expect = (0.5*(2+1)) + (0.5*(2+1)) + (0.5*(2+0))
        self.assertEqual(total_cost, total_cost_expect)        
        # TODO add assert for matching expected value

    def test_accepts_valid_inputs_from_shared(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        len_seq = shared_data_funcs.len_seq
        Q     = shared_data_funcs.pend_unit_cost_func_params['Q']
        R     = shared_data_funcs.pend_unit_cost_func_params['R']
        Qf    = shared_data_funcs.pend_unit_cost_func_params['Qf']
        x_seq = shared_data_funcs.x_seq
        u_seq = shared_data_funcs.u_seq
        total_cost, cost_seq = gen_ctrl.calculate_total_cost(cost_func, x_seq, u_seq)
        # (len-1)*0.5*(x.T Q x + u.T R u) + 0.5*(x.T Qf x)
        total_cost_expect = 9*(0.5*(1+4) + (0.5*(1))) + (0.5*(4+1))       
        self.assertEqual(total_cost, total_cost_expect)      

class taylor_expand_pseudo_hamiltonian_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()        
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_k = np.array([[1.],[1.]])
        u_k = np.array([[1.]])
        x_len = len(x_k)
        u_len = len(u_k)
        A_lin_k = np.array([[0,1],[1,1]])     
        B_lin_k = np.array([[0],[1]])
        p_kp1   = np.ones([x_len, 1])
        P_kp1   = np.ones([x_len, x_len])
        k_step  = 1
        ro_reg  = 1
        q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = ilqr.taylor_expand_pseudo_hamiltonian(
                                                                            cost_func, 
                                                                            A_lin_k, 
                                                                            B_lin_k, 
                                                                            x_k, 
                                                                            u_k, 
                                                                            P_kp1, p_kp1,
                                                                            ro_reg, 
                                                                            k_step)
        self.assertEqual(np.shape(q_x),  (x_len, 1))
        self.assertEqual(np.shape(q_u),  (u_len, 1))
        self.assertEqual(np.shape(q_xx), (x_len,x_len))
        self.assertEqual(np.shape(q_uu), (u_len,u_len))
        self.assertEqual(np.shape(q_ux), (u_len,x_len)) 
        self.assertEqual(np.shape(q_uu_reg), (u_len,u_len))
        self.assertEqual(np.shape(q_ux_reg), (u_len,x_len)) 

    def test_accepts_valid_inputs_from_shared(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()        
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = shared_data_funcs.x_seq
        u_seq = shared_data_funcs.u_seq
        x_len = len(x_seq[0])
        u_len = len(u_seq[0])
        A_lin_k = np.array([[1,1],[0,1]])     
        B_lin_k = np.array([[0],[1]])
        p_kp1   = np.ones([x_len, 1])
        P_kp1   = np.ones([x_len, x_len])
        k_step  = 1
        ro_reg = 1.0
        q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = ilqr.taylor_expand_pseudo_hamiltonian(
                                                                            cost_func, 
                                                                            A_lin_k, 
                                                                            B_lin_k, 
                                                                            x_seq[k_step], 
                                                                            u_seq[k_step], 
                                                                            P_kp1, p_kp1,
                                                                            ro_reg, 
                                                                            k_step)
        self.assertEqual(np.shape(q_x),  (x_len, 1))
        self.assertEqual(np.shape(q_u),  (u_len, 1))
        self.assertEqual(np.shape(q_xx), (x_len,x_len))
        self.assertEqual(np.shape(q_uu), (u_len,u_len))
        self.assertEqual(np.shape(q_ux), (u_len,x_len)) 
        self.assertEqual(np.shape(q_uu_reg), (u_len,u_len))
        self.assertEqual(np.shape(q_ux_reg), (u_len,x_len)) 

class calculate_final_ctg_approx_tests(unittest.TestCase):
    def test_calculate_final_cost_to_go_approximation_accepts_valid_system(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_N = np.array(([1.0],[2.0]))
        u_N = np.array(([1.0]))
        P_N, p_N = ilqr.calculate_final_ctg_approx(cost_func,x_N,len(u_N),len_seq = 1)
        p_N_expected = shared_data_funcs.pend_unit_cost_func_params['Qf'] @ x_N
        P_N_expected = shared_data_funcs.pend_unit_cost_func_params['Qf']
        self.assertEqual(p_N.tolist(), p_N_expected.tolist())
        self.assertEqual(P_N.tolist(), P_N_expected.tolist())

class calculate_backstep_ctg_approx_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        x_len = data_and_funcs.x_len
        u_len = data_and_funcs.u_len
        q_x   = np.ones([x_len,1])
        q_u   = np.ones([u_len,1])
        q_xx  = np.ones([x_len, x_len])
        q_uu  = np.ones([u_len, u_len])
        q_ux  = np.ones([u_len, x_len])
        K_k   = np.ones([u_len, x_len])
        d_k   = np.ones([u_len, 1])
        P_k, p_k, Del_V_vec_k = ilqr.calculate_backstep_ctg_approx(q_x, q_u, 
                                                                   q_xx, q_uu, q_ux, 
                                                                   K_k, d_k)
        P_k_expect = np.ones([x_len,x_len]) * 4.
        p_k_expect = np.ones([x_len, 1]) * 4.
        Del_V_vec_k_expect = np.array([1.0, 0.5])
        self.assertEqual(np.shape(P_k), (x_len, x_len))
        self.assertEqual(np.shape(p_k), (x_len, 1))  
        self.assertEqual(np.shape(Del_V_vec_k), (2, ))      
        self.assertEqual(P_k.tolist(), P_k_expect.tolist())
        self.assertEqual(p_k.tolist(), p_k_expect.tolist()) 
        self.assertEqual(Del_V_vec_k.tolist(), Del_V_vec_k_expect.tolist())

class calculate_optimal_gains_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        q_uu = np.array([[1.,1.],[0,1.]])
        q_ux = np.array([[1.,1.],[0,1.]])
        q_u  = np.array([[1.],[1.]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        # print(jnp.linalg.inv(q_uu))
        K_k_expected = -np.linalg.inv(q_uu) @ q_ux
        d_k_expected = -np.linalg.inv(q_uu) @ q_u
        K_k_expected = -(np.array([[1.,-1.],[0,1.]])) @ (np.array([[1.,1.],[0,1.]]))
        d_k_expected = -(np.array([[1.,-1.],[0,1.]])) @ (np.array([[1.],[1.]]))
        self.assertEqual(K_k.tolist(), K_k_expected.tolist())
        self.assertEqual(d_k.tolist(), d_k_expected.tolist())

    def test_accepts_valid_inputs_from_shared(self):
        q_uu = np.array([[1.,1.],[0,1.]])
        q_ux = np.array([[0.1,0.2,0.3,0.4],[1., 2., 3., 4.]])
        q_u  = np.array([[1],[2]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        K_k_expected = -np.linalg.inv(q_uu) @ q_ux
        d_k_expected = -np.linalg.inv(q_uu) @ q_u
        self.assertEqual(K_k.tolist(), K_k_expected.tolist())
        self.assertEqual(d_k.tolist(), d_k_expected.tolist())    

class calculate_u_k_new_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = np.array([[1.0], [2.0]])
        x_nom_k = np.array([[1.0], [1.0]])
        x_new_k = x_nom_k + 1.0
        K_k     = np.ones([2, 2])
        d_k     = np.ones([2, 1]) * 2.0
        line_search_factor = 1
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        # u_k_expected = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
        u_k_expected = np.array([[1. + (1.* (2.-1.) + 1. * (2.-1.)) + 1.*2.],
                                 [2. + (1.* (2.-1.) + 1. * (2.-1.)) + 1.*2.]])
        self.assertEqual(u_k_updated_new.tolist(), u_k_expected.tolist())

    def test_reject_invalid_K_k_dimension(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = np.array([1.0, 2.0])
        x_nom_k = np.array([1.0, 1.0])
        x_new_k = x_nom_k + 1.0
        K_k     = np.ones([3, 2])
        d_k     = np.ones([2, 1]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))')

    def test_reject_invalid_d_k_dimension(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = np.array([1.0, 2.0])
        x_nom_k = np.array([1.0, 1.0])
        x_new_k = x_nom_k + 1.0
        K_k     = np.ones([2, 2])
        d_k     = np.ones([2, 2]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'd_k incorrect shape, must be (len(u_nom_k), 1)')

    def test_reject_invalid_x_new_k_dimension(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = np.array([1.0, 2.0])
        x_nom_k = np.array([1.0, 1.0])
        x_new_k = np.array([1.0, 2.0, 3.0])
        K_k     = np.ones([2, 2])
        d_k     = np.ones([2, 1]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'x_nom_k and x_new_k must be the same length')

    def test_reject_invalid_x_dimension(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = np.array([1.0, 2.0])
        x_nom_k = np.array([1.0, 1.0])
        x_new_k = np.array([1.0, 2.0])
        K_k     = np.ones([2, 2])
        d_k     = np.ones([2, 1]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'x_nom_k must be column vector (n,1)')


    def test_accepts_valid_inputs_with_shared_data(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = data_and_funcs.u_seq[k_step]
        x_nom_k = data_and_funcs.x_seq[k_step]
        x_new_k = x_nom_k + 0.1
        K_k     = data_and_funcs.K_seq[k_step]
        d_k     = data_and_funcs.d_seq[k_step]
        line_search_factor = 0.5
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)

        K_k_mult = K_k @ (x_new_k - x_nom_k)
        u_k_expected = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
        self.assertEqual(u_k_updated_new, u_k_expected)    

class calculate_cost_decrease_ratio_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        prev_cost_float = 100.0
        new_cost_float  = 70.0
        Del_V_vec_seq = np.ones([3,2])
        Del_V_vec_seq[:,1] *= 2.0
        line_search_factor = 2.0
        cost_decrease_ratio = ilqr.calculate_cost_decrease_ratio(prev_cost_float,
                                                                 new_cost_float,
                                                                 Del_V_vec_seq,
                                                                 line_search_factor)
        del_V_sum_exp = ((1+1+1) * line_search_factor) + ((2+2+2) * line_search_factor**2)
        expected_output = np.array((1 / -del_V_sum_exp) * (prev_cost_float - new_cost_float))
        self.assertEqual(cost_decrease_ratio, expected_output)

    def test_rejects_invalid_input_dimension(self):
        prev_cost_float = 100.0
        new_cost_float  = 70.0
        Del_V_vec_seq = np.ones([3,3])
        Del_V_vec_seq[:,1] *= 2.0
        line_search_factor = 2
        with self.assertRaises(AssertionError) as assert_seq_len_error:
            cost_decrease_ratio = ilqr.calculate_cost_decrease_ratio(prev_cost_float,
                                                                 new_cost_float,
                                                                 Del_V_vec_seq,
                                                                 line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'Del_V_vec_seq is not the right shape, must be (:,2)')

class calculate_expected_cost_decrease_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        # Del_V_vec_seq = np.array([[1,2],[1,2],[1,2]]) 
        # Del_V_vec_seq = Del_V_vec_seq[:, np.newaxis]
        Del_V_vec_seq = np.ones([3,2])
        Del_V_vec_seq[:,1] *= 2.0
        line_search_factor = 2.0
        Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
        expected_output = ((1+1+1) * line_search_factor) + ((2+2+2) * line_search_factor**2)
        self.assertEqual(Del_V_sum, expected_output)

    def test_reject_invalid_array_shape(self):
        Del_V_vec_seq = np.ones([3,3]) 
        line_search_factor = 2
        with self.assertRaises(AssertionError) as assert_seq_len_error:
            Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'Del_V_vec_seq is not the right shape, must be (:,2)')

class analyze_cost_decrease_tests(unittest.TestCase):
    def test_returns_true_for_in_bound_inputs(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config_lin_pend_unit_cost['cost_ratio_bounds']
        cost_decrease_ratio = bounds[0] + (bounds[0] + bounds[1]) * 0.5
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, True)
 
    def test_returns_false_for_input_too_high(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config_lin_pend_unit_cost['cost_ratio_bounds']
        cost_decrease_ratio = bounds[1] + 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)

    def test_returns_false_for_input_too_low(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config_lin_pend_unit_cost['cost_ratio_bounds']
        cost_decrease_ratio = bounds[0] - 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)


class calculate_avg_ff_gains_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        #create test conditions
        int_iter = 5
        d_seq = np.ones((4,2,1))
        u_seq = np.ones((4,2,1))
        #test function
        ff_avg_gains = ilqr.calculate_avg_ff_gains(int_iter, d_seq, u_seq)
        # create expected output
        ff_avg_gains_expect = 1 / (5-1) * (1 / (np.sqrt(2) + 1)) * 4
        #compare outputs
        self.assertEqual(ff_avg_gains, ff_avg_gains_expect)

if __name__ == '__main__':
    unittest.main()
