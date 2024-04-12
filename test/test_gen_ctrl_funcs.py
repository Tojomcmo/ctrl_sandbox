import unittest
from jax import numpy as jnp
import numpy as np
import scipy
import numpy.typing as npt
import numpy.testing
from typing import Callable, Tuple
import src.gen_ctrl_funcs as gen_ctrl
import src.ilqr_utils as util
 

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
    

    def pend_unit_dyn_func(self, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])-> npt.NDArray:
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
     
    def step_rk4(self,
            dyn_func_ad:Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
            h:float, 
            x_k:npt.NDArray[np.float64], 
            u_k:npt.NDArray[np.float64]) -> npt.NDArray:
        #rk4 integration with zero-order hold on u
        f1 = dyn_func_ad(x_k            , u_k)
        f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
        f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
        f4 = dyn_func_ad(x_k + h*f3     , u_k)
        return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)    
    
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
    
    def pend_unit_dyn_func_curried(self, state, control):
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
            cost_float = ((0.5) * (jnp.transpose(x_k_corr_col)  @ Qf @ x_k_corr_col)).item()
        else:
            u_k_corr     = util.calc_and_shape_array_diff(control_vec, control_des_seq[k_step])
            u_k_corr_col = util.array_to_col(u_k_corr)
            cost_float = ((0.5) * ( (jnp.transpose(x_k_corr_col) @ Q @ x_k_corr_col)
                                  +(jnp.transpose(u_k_corr_col) @ R @ u_k_corr_col))).item()
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

class simulate_forward_dynamics_tests(unittest.TestCase):
    def pend_unit_dyn_func(self, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])-> npt.NDArray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([
                    x_k_jax[1],
                    -(b/l) * x_k_jax[1] - (np.sin(x_k_jax[0]) * g/l) + u_k_jax[0]
                    ], dtype=float)
        return state_dot  

    def test_accepts_valid_input(self):
        len_seq         = 4
        u_seq     = np.ones([4,1,1])
        x_init      = np.array([[1.],[1.]])
        x_seq       = gen_ctrl.simulate_forward_dynamics_seq(self.pend_unit_dyn_func, x_init, u_seq)
        # calculate expected output
        x_seq_expected    = np.zeros([5,2,1])
        x_seq_expected[0] = x_init
        for idx in range(len_seq):
            x_delta    = self.pend_unit_dyn_func(x_seq[idx], u_seq[idx])
            x_next       = x_seq[idx] + x_delta
            x_seq_expected[idx+1] = x_next
        self.assertEqual(len(x_seq), len_seq +1)
        self.assertEqual(x_seq[-1].tolist(), x_seq_expected[-1].tolist())

class linearize_dynamics_tests(unittest.TestCase):

    def pend_unit_dyn_func(self, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])-> npt.NDArray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([x_k_jax[1],
                             -(b/l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g/l) + u_k_jax[0]], dtype=float)
        return state_dot  
    
    def step_rk4(self,
            dyn_func_ad:Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
            h:float, 
            x_k:npt.NDArray[np.float64], 
            u_k:npt.NDArray[np.float64]) -> npt.NDArray:
        #rk4 integration with zero-order hold on u
        f1 = dyn_func_ad(x_k            , u_k)
        f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
        f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
        f4 = dyn_func_ad(x_k + h*f3     , u_k)
        return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)
    
    def test_linearize_dynamics_accepts_valid_case(self):
        h = 0.1
        x = np.array([[1.],[1.]])
        u = np.array([[1.]])
        discrete_dyn_func = lambda x,u: self.step_rk4(self.pend_unit_dyn_func,h,x,u)
        a_lin, b_lin    = gen_ctrl.linearize_dynamics(discrete_dyn_func,x,u)
        g = 1.0
        l = 1.0
        b = 1.0
        state_pos = x[0][0]
        A_lin_expected = np.array([[0, 1],[-(g/l)*jnp.cos(state_pos), -(b/l)]])
        B_lin_expected = np.array([[0],[1]])
        A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
        B_lin_d_expected = B_lin_expected * h
        self.assertEqual(jnp.shape(a_lin), (len(x), len(x)))
        self.assertEqual(jnp.shape(b_lin), (len(x), len(u)))
        numpy.testing.assert_allclose(a_lin, A_lin_d_expected,rtol=1e-1, atol=1e-1)
        numpy.testing.assert_allclose(b_lin, B_lin_d_expected,rtol=1e-1, atol=1e-1)

class calculate_linearized_state_space_seq_tests(unittest.TestCase):

    def pend_unit_dyn_func(self, state:npt.NDArray[np.float64], control:npt.NDArray[np.float64])-> npt.NDArray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([x_k_jax[1],
                    -(b/l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g/l) + u_k_jax[0]], dtype=float)
        return state_dot  
    
    def step_rk4(self,
            dyn_func_ad:Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray], 
            h:float, 
            x_k:npt.NDArray[np.float64], 
            u_k:npt.NDArray[np.float64]) -> npt.NDArray:
        #rk4 integration with zero-order hold on u
        f1 = dyn_func_ad(x_k            , u_k)
        f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
        f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
        f4 = dyn_func_ad(x_k + h*f3     , u_k)
        return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)
    
    def test_calculate_linearized_state_space_seq_accepts_valid_system(self):
        len_seq     = 4
        h           = 0.1
        u_seq = np.ones((3,1,1), dtype=np.float64)
        x_seq   = np.ones((4,2,1), dtype=np.float64)
        discrete_dyn_func = lambda x,u: self.step_rk4(self.pend_unit_dyn_func,h,x,u)
        A_lin_d_seq, B_lin_d_seq = gen_ctrl.calculate_linearized_state_space_seq(discrete_dyn_func,
                                                                                x_seq, 
                                                                                u_seq)
        # calculate expected output
        x_init_pos = x_seq[0][0]
        A_lin_expected = np.array([[0, 1],[-(1./1.) * np.cos(x_init_pos[0]), -(1./1.)]])
        B_lin_expected = np.array([[0],[1]])
        A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
        B_lin_d_expected = B_lin_expected * h
        #assert validity
        self.assertEqual(len(A_lin_d_seq), len_seq-1)
        self.assertEqual(len(B_lin_d_seq), len_seq-1)
        numpy.testing.assert_allclose(A_lin_d_seq[0], A_lin_d_expected,rtol=1e-1, atol=1e-1)
        numpy.testing.assert_allclose(B_lin_d_seq[0], B_lin_d_expected,rtol=1e-1, atol=1e-1)
        
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
            ss = gen_ctrl.discretize_continuous_state_space(ss_c,time_step,c2d_method)
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
        ss = gen_ctrl.discretize_continuous_state_space(ss_c,time_step,c2d_method='euler')
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
            ss = gen_ctrl.discretize_continuous_state_space(ss_d,time_step)
        self.assertEqual(str(assert_error_discrete.exception), 'input state space is already discrete')
    
    def test_descritize_state_space_rejects_invalid_c2d_method(self):
        A = np.array([[1,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        time_step = 0.1   
        ss_d = gen_ctrl.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_c2d_method:
            ss = gen_ctrl.discretize_continuous_state_space(ss_d,time_step, c2d_method = 'lettuce')
        self.assertEqual(str(assert_error_c2d_method.exception), 'invalid discretization method')
     
    def test_descritize_state_space_rejects_singular_A_matrix_for_zoh(self):
        A = np.array([[0,1,1],[0,1,1],[0,0,1]])
        B = np.array([[1,1],[0,1],[0,0]])
        C = np.array([[1,0,1]])
        D = np.array([[0,0]])
        time_step = 0.1   
        ss_d = gen_ctrl.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_singular_A_zoh:
            ss = gen_ctrl.discretize_continuous_state_space(ss_d,time_step, c2d_method = 'zoh')
        self.assertEqual(str(assert_error_singular_A_zoh.exception), 'determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid'
)

class calculate_total_cost_tests(unittest.TestCase):

    def cost_func_quad_state_and_control(self, 
                                         cost_func_params:dict, 
                                         x_k:npt.NDArray[np.float64], 
                                         u_k:npt.NDArray[np.float64], 
                                         k_step:int, 
                                         x_des_seq:npt.NDArray[np.float64], 
                                         u_des_seq:npt.NDArray[np.float64], 
                                         is_final_bool:bool):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array(cost_func_params['Q'])
        R  = jnp.array(cost_func_params['R'])
        Qf = jnp.array(cost_func_params['Qf'])
        x_des_seq_jax = jnp.array(x_des_seq)
        u_des_seq_jax = jnp.array(u_des_seq)
        # check that dimensions match [TODO]
        if is_final_bool:
            x_k_corr = util.calc_and_shape_array_diff(x_k  , x_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k_step == 0:
            u_k_corr   = util.calc_and_shape_array_diff(u_k, u_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = util.calc_and_shape_array_diff(x_k, x_des_seq_jax[k_step], shape='col')   
            u_k_corr = util.calc_and_shape_array_diff(u_k, u_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr) @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((jnp.transpose(x_k_corr) @ Q @ x_k_corr)+(jnp.transpose(u_k_corr) @ R @ u_k_corr)))
        return total_cost, x_cost, u_cost                    

    def unit_cost_func_quad_state_and_control_pend_curried(self, x_k, u_k, k_step, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        cost_func         = self.cost_func_quad_state_and_control
        cost_func_params  = {'Q'  : np.array([[1.,0],[0,1.]]), 
                             'R'  : np.array([[1.]]),
                             'Qf' : np.array([[1.,0],[0,1.]])} 
        x_des_seq     = np.zeros([3,2,1])
        u_des_seq   = np.zeros([2,1,1])
        cost_func_curried = lambda x, u, k, is_final_bool=False: cost_func(cost_func_params, x, u, k, x_des_seq, u_des_seq, is_final_bool)
        cost_float = cost_func_curried(x_k, u_k, k_step, is_final_bool)
        return cost_float   

    def test_accepts_valid_inputs(self):
        cost_func = self.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = np.ones([3,2,1])
        u_seq = np.ones([2,1,1])
        cost_func_calc = gen_ctrl.prep_cost_func_for_calc(cost_func)
        total_cost, cost_seq, x_cost_seq, u_cost_seq = gen_ctrl.calculate_total_cost(cost_func_calc, x_seq, u_seq)
        total_cost_expect = (0.5*(0+1)) + (0.5*(2+1)) + (0.5*(2+0))
        cost_seq_expect   = np.array([0.5, 1.5, 1.0]) 
        x_cost_seq_expect = np.array([0.0, 1.0, 1.0])
        u_cost_seq_expect = np.array([0.5, 0.5, 0.0])
        self.assertEqual(total_cost, total_cost_expect) 
        self.assertEqual(cost_seq.tolist(), cost_seq_expect.tolist())
        self.assertEqual(x_cost_seq.tolist(), x_cost_seq_expect.tolist())
        self.assertEqual(u_cost_seq.tolist(), u_cost_seq_expect.tolist())                       
     

class taylor_expand_cost_tests(unittest.TestCase):  
    
    def cost_func_quad_state_and_control(self, 
                                         cost_func_params:dict, 
                                         x_k:npt.NDArray[np.float64], 
                                         u_k:npt.NDArray[np.float64], 
                                         k_step:int, 
                                         x_des_seq:npt.NDArray[np.float64], 
                                         u_des_seq:npt.NDArray[np.float64], 
                                         is_final_bool:bool):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array(cost_func_params['Q'])
        R  = jnp.array(cost_func_params['R'])
        Qf = jnp.array(cost_func_params['Qf'])
        x_des_seq_jax = jnp.array(x_des_seq)
        u_des_seq_jax = jnp.array(u_des_seq)
        # check that dimensions match [TODO]
        if is_final_bool:
            x_k_corr = util.calc_and_shape_array_diff(x_k  , x_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k_step == 0:
            u_k_corr   = util.calc_and_shape_array_diff(u_k, u_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = util.calc_and_shape_array_diff(x_k, x_des_seq_jax[k_step], shape='col')   
            u_k_corr = util.calc_and_shape_array_diff(u_k, u_des_seq_jax[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr) @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((jnp.transpose(x_k_corr) @ Q @ x_k_corr)+(jnp.transpose(u_k_corr) @ R @ u_k_corr)))
        return total_cost, x_cost, u_cost                  

    def unit_cost_func_quad_state_and_control_pend_curried(self, 
                                                           x_k:npt.NDArray[np.float64], 
                                                           u_k:npt.NDArray[np.float64], 
                                                           k_step:int, 
                                                           is_final_bool:bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        cost_func         = self.cost_func_quad_state_and_control
        cost_func_params  = {'Q'  : np.array([[1.,0],[0,1.]]), 
                             'R'  : np.array([[1.]]),
                             'Qf' : np.array([[1.,0],[0,1.]])} 
        x_des_seq   = np.zeros([3,2,1])
        u_des_seq   = np.zeros([2,1,1])
        cost_func_curried = lambda x, u, k, is_final_bool=False: cost_func(cost_func_params, x, u, k, x_des_seq, u_des_seq, is_final_bool)
        cost_tuple = cost_func_curried(x_k, u_k, k_step, is_final_bool)
        return cost_tuple   
    
    def test_taylor_expand_cost_accepts_list_of_arrays(self):
        cost_func = self.unit_cost_func_quad_state_and_control_pend_curried
        x_seq      = np.ones([3,2,1])
        u_seq      = np.ones([2,1,1])
        k_step     = 1
        l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost(cost_func,x_seq[k_step],u_seq[k_step],k_step)
        cost_func_params  = {'Q'  : np.array([[1.,0],[0,1.]]), 
                             'R'  : np.array([[1.]]),
                             'Qf' : np.array([[1.,0],[0,1.]])} 
        x_des_seq  = np.zeros([3,2,1])
        u_des_seq  = np.zeros([2,1,1])
        l_x_expected  = cost_func_params['Q'] @ (x_seq[k_step]-x_des_seq[k_step])
        l_u_expected  = cost_func_params['R'] @ (u_seq[k_step]-u_des_seq[k_step])
        l_xx_expected = cost_func_params['Q']
        l_uu_expected = cost_func_params['R']  
        self.assertEqual(l_x.tolist(),  l_x_expected.tolist())
        self.assertEqual(l_u.tolist(),  l_u_expected.tolist())
        self.assertEqual(l_xx.tolist(), l_xx_expected.tolist())
        self.assertEqual(l_uu.tolist(), l_uu_expected.tolist())
        self.assertEqual(l_ux.tolist(), np.zeros([len(u_seq[0]),len(x_seq[0])]).tolist())            


class prep_xu_vecs_for_diff_tests(unittest.TestCase):
    def test_accepts_valid_np_col_inputs(self):
        #create test conditions
        x_k = np.array([[1],[2],[3]])
        u_k = np.array([[4],[5]])
        #test function
        xu_k_jax, xu_k_len, x_k_len = gen_ctrl.prep_xu_vec_for_diff(x_k, u_k)
        # create expected output
        xu_k_jax_expected = jnp.array([[1],[2],[3],[4],[5]])
        xu_k_len_expected = 5
        x_k_len_expected  = 3
        #compare outputs
        self.assertEqual(jnp.shape(xu_k_jax), (5,1))
        self.assertEqual(xu_k_jax.tolist(), xu_k_jax_expected.tolist())
        self.assertEqual(xu_k_len, xu_k_len_expected)
        self.assertEqual(x_k_len, x_k_len_expected)

    def test_rejects_np_row_x_inputs(self):
        #create test conditions
        x_k = np.array([[1,2,3]])
        u_k = np.array([[4,5]])
        with self.assertRaises(AssertionError) as assert_seq_len_error:
            xu_k_jax, xu_k_len, x_k_len = gen_ctrl.prep_xu_vec_for_diff(x_k, u_k)
        self.assertEqual(str(assert_seq_len_error.exception), 'x_vector must be column vector (n,1)')

    def test_rejects_np_row_u_inputs(self):
        #create test conditions
        x_k = np.array([[1],[2],[3]])
        u_k = np.array([[4,5]])
        with self.assertRaises(AssertionError) as assert_seq_len_error:
            xu_k_jax, xu_k_len, x_k_len = gen_ctrl.prep_xu_vec_for_diff(x_k, u_k)
        self.assertEqual(str(assert_seq_len_error.exception), 'u_vector must be column vector (m,1)')


class prep_cost_func_for_diff_tests(unittest.TestCase):
    def cost_func(self, vec_1_and_2): 
        cost_val_1 = np.sum(vec_1_and_2)
        cost_val_2 = np.sum(vec_1_and_2) + 2
        combined_cost = cost_val_1 + cost_val_2
        return combined_cost, cost_val_1, cost_val_2        
    
    def test_accepts_valid_cost_func_inputs(self):
        #create test conditions
        x_k = np.array([[1],[2]])
        u_k = np.array([[3],[4]])
        xu_k = np.stack([x_k, u_k], axis=0)
        #test function
        cost_func_for_diff = gen_ctrl.prep_cost_func_for_diff(self.cost_func)
        combined_cost = cost_func_for_diff(xu_k)
        _, cost_val_1, cost_val_2 = self.cost_func(xu_k)
        # create expected output
        combined_cost_expected = 22
        cost_val_1_expected = 10
        cost_val_2_expected  = 12
        #compare outputs
        self.assertEqual(combined_cost, combined_cost_expected)
        self.assertEqual(cost_val_1, cost_val_1_expected)
        self.assertEqual(cost_val_2, cost_val_2_expected)

if __name__ == '__main__':
    unittest.main()
