import unittest
import numpy as np
import ilqr_funcs as ilqr
from jax import numpy as jnp
import ilqr_utils as util

# for each input
#   test one known case
#   if discrete change of behavior (abs value add, check pos and neg)
#   check edge cases (NaN, overflow, string, divide by zero)

class shared_unit_test_data_and_funcs:
    def __init__(self) -> None:
        self.pend_unit_cost_func_params = {
                        'Q'  : jnp.array([[1.,0],[0,1.]]),
                        'R'  : jnp.array([[1.]]),
                        'Qf' : jnp.array([[1.,0],[0,1.]])
                      } 
        self.pend_unit_state_trans_params = {
                        'b'  : 1.0,
                        'l'  : 1.0,
                        'g'  : 1.0
                      } 
        self.gen_cost_func_params = {
                        'Q'  : jnp.array([[1.,0,0],[0,10.,0],[0,0,1.]]),
                        'R'  : jnp.array([[1.,0],[0,10.]]),
                        'Qf' : jnp.array([[1.,0,0],[0,10.,0],[0,0,10.]])
                      } 
        self.gen_state_trans_params = {
                        'b'  : 1.0,
                        'l'  : 2.0,
                        'g'  : 3.0
                      } 
        self.ilqr_config = {
            'state_trans_func'          : self.pend_dyn_func_full,
            'state_trans_func_params'   : self.pend_unit_state_trans_params,
            'cost_func'                 : self.cost_func_quad_state_and_control_full,
            'cost_func_params'          : self.pend_unit_cost_func_params,
            'sim_method'                : 'euler',
            'c2d_method'                : 'euler',
            'max_iter'                  : 100,
            'time_step'                 : 0.1,
            'converge_crit'             : 1e-6,
            'cost_ratio_bounds'         : [1e-4, 10]
            }
        
        # self.len_seq              = 20
        # self.seed_state_vec       = (jnp.ones([1,2]) * 0.1)
        # self.seed_control_vec_seq = jnp.zeros([self.len_seq,1])

        # self.state_seq            = (jnp.ones([self.len_seq,2]) * 0.1)
        # self.control_seq          = jnp.ones([self.len_seq,1])
        # self.time_step            = self.ilqr_config['time_step']
        # self.time_seq             = jnp.arange(self.len_seq) * self.time_step
        # pos_des_traj         = jnp.ones(self.len_seq) * jnp.pi
        # vel_des_traj         = jnp.zeros(self.len_seq)
        # self.des_state_seq   = jnp.stack([pos_des_traj,vel_des_traj], axis=1)
        # self.des_control_seq = jnp.zeros([20,1])

        self.len_seq          = 10
        self.seed_state_vec   = (jnp.ones([2,1]) * 0.1)
        self.seed_control_seq = [jnp.ones([1,1])] * (self.len_seq-1)

        self.x_len            = jnp.shape(util.vec_1D_array_to_col(self.seed_state_vec))[0]
        self.u_len            = jnp.shape(util.vec_1D_array_to_col(self.seed_control_seq[0]))[0]

        self.state_seq        = [self.seed_state_vec] * self.len_seq
        self.control_seq      = [jnp.ones([1,1])] * (self.len_seq-1)

        self.des_state_seq    = [jnp.array([[0], [0]])] * self.len_seq
        self.des_control_seq  = [jnp.array([[0]])] * (self.len_seq-1)

        self.K_seq            = [jnp.ones([self.u_len, self.x_len])] * (self.len_seq-1)
        self.d_seq            = [jnp.ones([self.u_len, 1         ])] * (self.len_seq-1)
        self.Del_V_vec_seq    = [jnp.ones([1         , 2         ])] * (self.len_seq-1)

        self.time_step        = self.ilqr_config['time_step']
        time_id               = list(range(self.len_seq))
        self.time_seq         = [i * self.time_step for i in time_id]

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

    def cost_func_quad_state_and_control_full(self,cost_func_params:dict, state_vec, control_vec, k_step, state_des_seq, control_des_seq, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = cost_func_params['Q']
        R  = cost_func_params['R']
        Qf = cost_func_params['Qf']
        # check that dimensions match [TODO]
        x_k_corr     = util.calculate_current_minus_des_array(state_vec, state_des_seq[k_step])
        x_k_corr_col = util.vec_1D_array_to_col(x_k_corr)
        if is_final_bool:
            cost_float = (0.5) * (jnp.transpose(x_k_corr_col)  @ Qf @ x_k_corr_col)
        else:
            u_k_corr     = util.calculate_current_minus_des_array(control_vec, control_des_seq[k_step])
            u_k_corr_col = util.vec_1D_array_to_col(u_k_corr)
            cost_float = (0.5) * ( (jnp.transpose(x_k_corr_col) @ Q @ x_k_corr_col)
                                  +(jnp.transpose(u_k_corr_col) @ R @ u_k_corr_col))
        return cost_float      

    def unit_cost_func_quad_state_and_control_pend_curried(self, state_vec, control_vec, k_step, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = self.pend_unit_cost_func_params['Q']
        R  = self.pend_unit_cost_func_params['R']
        Qf = self.pend_unit_cost_func_params['Qf']
        
        # check that dimensions match [TODO]
        x_k_corr     = util.calculate_current_minus_des_array(state_vec  , self.des_state_seq[k_step]  )
        x_k_corr_col = util.vec_1D_array_to_col(x_k_corr)
        if is_final_bool:
            cost_float   = (0.5) * (jnp.transpose(x_k_corr_col)  @ Qf @ x_k_corr_col)
        else:
            u_k_corr     = util.calculate_current_minus_des_array(control_vec, self.des_control_seq[k_step])
            u_k_corr_col = util.vec_1D_array_to_col(u_k_corr)       
            cost_float   = (0.5) * ( (jnp.transpose(x_k_corr_col) @ Q @ x_k_corr_col)
                                    +(jnp.transpose(u_k_corr_col) @ R @ u_k_corr_col))
        return cost_float   

class stateSpace_tests(unittest.TestCase):
    def test_state_space_class_accepts_valid_continuous_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss = ilqr.stateSpace(A,B,C,D)
        self.assertEqual(ss.a.all(), A.all())
        self.assertEqual(ss.b.all(), B.all())
        self.assertEqual(ss.c.all(), C.all())
        self.assertEqual(ss.d.all(), D.all())
        self.assertEqual(ss.type, 'continuous')
        self.assertEqual(ss.time_step, None)

    def test_state_space_class_accepts_valid_discrete_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss = ilqr.stateSpace(A,B,C,D, time_step = 0.1)
        self.assertEqual(ss.a.all(), A.all())
        self.assertEqual(ss.b.all(), B.all())
        self.assertEqual(ss.c.all(), C.all())
        self.assertEqual(ss.d.all(), D.all())
        self.assertEqual(ss.type, 'discrete')
        self.assertEqual(ss.time_step, 0.1)

    def test_state_space_rejects_nonsquare_A_matrix(self):
        A = jnp.array([[1,1,1],[0,1,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = ilqr.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A matrix must be square')

    def test_state_space_rejects_A_and_B_matrix_n_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = ilqr.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A and B matrices must have same n(state) dimension')        

    def test_state_space_rejects_A_and_C_matrix_m_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0]])
        D = jnp.array([[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = ilqr.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'A and C matrices must have same m(state) dimension')        

    def test_state_space_rejects_B_and_D_matrix_m_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = ilqr.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'B and D matrices must have the same m(control) dimension')  
 
    def test_state_space_rejects_C_and_D_matrix_n_dim_mismatch(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,],[0,0]])
        with self.assertRaises(Exception) as assert_error:
            ss = ilqr.stateSpace(A,B,C,D) 
        self.assertEqual(str(assert_error.exception), 'C and D matrices must have the same n(measurement) dimension')  

    def test_state_space_rejects_invalid_timestep_input(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0,]])
        with self.assertRaises(Exception) as assert_error_negative:
            ss = ilqr.stateSpace(A,B,C,D, time_step = -0.1) 
        self.assertEqual(str(assert_error_negative.exception), 'invalid time step definition -0.1. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_string:
            ss = ilqr.stateSpace(A,B,C,D, time_step = 'lettuce') 
        self.assertEqual(str(assert_error_string.exception), 'invalid time step definition lettuce. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_zero:
            ss = ilqr.stateSpace(A,B,C,D, time_step = 0) 
        self.assertEqual(str(assert_error_zero.exception), 'invalid time step definition 0. time_step must be a positive float or None')
        with self.assertRaises(Exception) as assert_error_NaN:
            ss = ilqr.stateSpace(A,B,C,D, time_step = float("nan")) 
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
        ilqr_config = data_and_funcs.ilqr_config
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.ilqr_config,
                                              data_and_funcs.seed_state_vec,
                                              data_and_funcs.control_seq,
                                              data_and_funcs.des_state_seq,
                                              data_and_funcs.des_control_seq)
        config_funcs, state_seq, cost_float, prev_cost_float = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        
        self.assertEqual(len(state_seq),data_and_funcs.len_seq) 
        self.assertEqual((jnp.abs(cost_float-prev_cost_float) > ilqr_config['converge_crit']), True)   

class run_ilqr_controller_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        self.assertEqual(True,True)

class calculate_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.seed_state_vec,
                                              data_and_funcs.control_seq,
                                              data_and_funcs.time_step,
                                              data_and_funcs.des_state_seq,
                                              data_and_funcs.des_control_seq)
        ctrl_state.state_seq = data_and_funcs.state_seq
        config_funcs = ilqr.ilqrConfiguredFuncs(data_and_funcs.ilqr_config,
                                                ctrl_state)
        
        k_seq, d_seq, Del_V_vec_seq = ilqr.calculate_backwards_pass(config_funcs, ctrl_state)
        self.assertEqual(len(k_seq),data_and_funcs.len_seq-1)
        self.assertEqual(len(d_seq),data_and_funcs.len_seq-1)
        self.assertEqual(len(Del_V_vec_seq),data_and_funcs.len_seq-1) 
        self.assertEqual(jnp.shape(k_seq[0]),(data_and_funcs.u_len, data_and_funcs.x_len))                  
        self.assertEqual(jnp.shape(d_seq[0]),(data_and_funcs.u_len, 1))                       
        self.assertEqual(jnp.shape(Del_V_vec_seq[0]),(1, 2)) 

class calculate_forwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = data_and_funcs.ilqr_config
        ctrl_state = ilqr.ilqrControllerState(data_and_funcs.seed_state_vec,
                                              data_and_funcs.control_seq,
                                              data_and_funcs.time_step,
                                              data_and_funcs.des_state_seq,
                                              data_and_funcs.des_control_seq)
        config_funcs, state_seq, cost_float, prev_cost_float = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state)
        ctrl_state.state_seq = state_seq
        ctrl_state.cost_float = cost_float
        ctrl_state.prev_cost_float = prev_cost_float
        k_seq, d_seq, Del_V_vec_seq = ilqr.calculate_backwards_pass(config_funcs, ctrl_state)
        ctrl_state.K_seq         = k_seq
        ctrl_state.d_seq         = d_seq
        ctrl_state.Del_V_vec_seq = Del_V_vec_seq                
        # ctrl_state.K_seq         = data_and_funcs.K_seq
        # ctrl_state.d_seq         = data_and_funcs.d_seq
        # ctrl_state.Del_V_vec_seq = data_and_funcs.Del_V_vec_seq
        state_seq_new, control_seq_new, cost_float_new = ilqr.calculate_forwards_pass(config_funcs, ctrl_state)
        
        self.assertEqual(True,True)

class simulate_forward_dynamics_tests(unittest.TestCase):
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

    def test_accepts_valid_inputs_euler(self):
        len_seq         = 4
        time_step       = 0.1
        control_seq     = [jnp.array([[1.]])] * (len_seq-1)
        state_seq       = [jnp.array([[1.],[1.]])] * (len_seq)
        time_step       = 0.1
        state_init      = jnp.array([[1.],[1.]])
        state_seq       = ilqr.simulate_forward_dynamics(self.pend_unit_dyn_func, 
                                                         state_init, control_seq, time_step, 
                                                         sim_method = 'euler')
        # calculate expected output
        time_seq = [0,0.1,0.2,0.3]
        state_seq_expected = [None] * (len_seq)
        state_seq_expected[0] = state_init
        for idx in range(len_seq-1):
            state_dot_row    = self.pend_unit_dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next       = state_seq[idx] + state_dot_row * time_step
            state_seq_expected[idx+1] = state_next
        self.assertEqual(len(state_seq), len_seq)
        self.assertEqual(state_seq[-1].all(), state_seq_expected[-1].all())

    def test_accepts_valid_inputs_euler_from_shared(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        dyn_func        = data_and_funcs.pend_unit_dyn_func_curried
        time_step       = data_and_funcs.time_step
        control_seq     = data_and_funcs.control_seq
        state_init      = data_and_funcs.seed_state_vec
        x_len           = len(data_and_funcs.state_seq[0])
        u_len           = len(data_and_funcs.control_seq[0])
        len_seq         = data_and_funcs.len_seq
        state_seq       = ilqr.simulate_forward_dynamics(dyn_func, state_init, 
                                                         control_seq, time_step, 
                                                         sim_method = 'euler')
        self.assertEqual(len(state_seq), len_seq)

    def test_accepts_valid_inputs_solve_ivp_zoh(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        dyn_func        = data_and_funcs.pend_unit_dyn_func_curried
        time_step       = data_and_funcs.time_step
        control_seq     = data_and_funcs.control_seq
        state_init      = data_and_funcs.seed_state_vec
        x_len           = len(data_and_funcs.state_seq[0])
        u_len           = len(data_and_funcs.control_seq[0])
        len_seq         = data_and_funcs.len_seq
        state_seq       = ilqr.simulate_forward_dynamics(dyn_func, state_init, 
                                                         control_seq, time_step, 
                                                         sim_method = 'solve_ivp_zoh')
        self.assertEqual(len(state_seq), len_seq)
        self.assertEqual(jnp.shape(state_seq[0]), (x_len,1))        
        self.assertEqual(True,True)        

class linearize_dynamics_tests(unittest.TestCase):

    def test_linearize_dynamics_accepts_valid_case(self):
        data_and_funcs  = shared_unit_test_data_and_funcs()
        time_seq        = data_and_funcs.time_seq
        control_seq     = data_and_funcs.control_seq
        state_seq       = data_and_funcs.state_seq
        x_len           = data_and_funcs.x_len
        u_len           = data_and_funcs.u_len
        k_step          = 1
        a_lin, b_lin    = ilqr.linearize_dynamics(data_and_funcs.pend_unit_dyn_func_curried,
                                                  time_seq[k_step],
                                                  state_seq[k_step],
                                                  control_seq[k_step])
        
        g = data_and_funcs.pend_unit_state_trans_params['g']
        l = data_and_funcs.pend_unit_state_trans_params['l']
        b = data_and_funcs.pend_unit_state_trans_params['b']
        state_pos = state_seq[k_step][0][0]
        a_lin_expected = jnp.array([[0, 1], 
                                    [-(g/l)*jnp.cos(state_pos), -(b/l)]])
        b_lin_expected = jnp.array([[0],[1]])
        print('a_lin: ', a_lin)
        print('a_lin_expected: ', a_lin_expected)
        print('b_lin: ', b_lin)
        print('b_lin_expected: ', b_lin_expected)
        self.assertEqual(jnp.shape(a_lin), (x_len, x_len))
        self.assertEqual(jnp.shape(b_lin), (x_len, u_len))
        self.assertEqual(a_lin.all(), a_lin_expected.all())
        self.assertEqual(b_lin.all(), b_lin_expected.all())

class calculate_linearized_state_space_seq_tests(unittest.TestCase):

    def discretize_ss_simple(self, input_state_space:ilqr.stateSpace):
        time_step = 0.1
        n  = input_state_space.a.shape[0] 
        a_d = jnp.eye(n) + (input_state_space.a * time_step)
        b_d = input_state_space.b * time_step
        c_d = input_state_space.c
        d_d = input_state_space.d
        d_state_space = ilqr.stateSpace(a_d, b_d, c_d, d_d, time_step)
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
        A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.pend_unit_dyn_func,
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
        self.assertEqual((A_lin_array[0]).all(), A_lin_d_expected.all())
        self.assertEqual((B_lin_array[0]).all(), B_lin_d_expected.all())        

    def test_calculate_linearized_state_space_seq_accepts_valid_system_from_shared(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        time_seq      = data_and_funcs.time_seq
        len_seq        = data_and_funcs.len_seq
        x_len          = len(data_and_funcs.state_seq[0])
        u_len          = len(data_and_funcs.control_seq[0])
        control_seq    = data_and_funcs.control_seq
        state_seq      = data_and_funcs.state_seq
        A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(data_and_funcs.pend_unit_dyn_func_curried,
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
    #         A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_dim_error.exception), 'state or control sequence is incorrect array dimension. sequences must be 2d arrays')

    # def test_calculate_linearized_state_space_seq_rejects_invalid_sequence_lengths(self):
    #     time_step    = 0.1
    #     num_steps    = 20
    #     num_states   = 3
    #     num_controls = 2
    #     control_seq  = jnp.ones((num_steps, num_controls))
    #     state_seq    = jnp.ones((num_steps+2, num_states)) * 0.1
    #     with self.assertRaises(Exception) as assert_seq_len_error:
    #         A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_len_error.exception), 'state and control sequences are incompatible lengths. state seq must be control seq length +1')

    # def test_calculate_linearized_state_space_seq_rejects_invalid_time_step(self):
    #     time_step    = -0.1
    #     num_steps    = 20
    #     num_states   = 3
    #     num_controls = 2
    #     control_seq  = jnp.ones((num_steps, num_controls))
    #     state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
    #     with self.assertRaises(Exception) as assert_seq_len_error:
    #         A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
    #     self.assertEqual(str(assert_seq_len_error.exception), 'time_step is invalid. Must be positive float')

class discretize_state_space_tests(unittest.TestCase):

    def test_discretize_state_space_accepts_valid_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss_c = ilqr.stateSpace(A,B,C,D)
        time_step = 0.1
        c2d_methods = ['euler', 'zoh', 'zohCombined']
        for c2d_method  in c2d_methods:
            ss = ilqr.discretize_state_space(ss_c,time_step,c2d_method)
            self.assertEqual(ss.a.shape, A.shape)
            self.assertEqual(ss.b.shape, B.shape)
            self.assertEqual(ss.c.shape, C.shape)
            self.assertEqual(ss.d.shape, D.shape)
            self.assertEqual(ss.time_step, time_step)
            self.assertEqual(ss.type, 'discrete')

    def test_discretize_state_space_returns_correct_euler_discretization(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss_c = ilqr.stateSpace(A,B,C,D)
        time_step = 0.1
        ss = ilqr.discretize_state_space(ss_c,time_step,c2d_method='euler')
        A_d_expected = jnp.eye(3) + (A * time_step)
        B_d_expected = B * time_step
        C_d_expected = C
        D_d_expected = D
        self.assertEqual(ss.a.shape, A.shape)
        self.assertEqual(ss.b.shape, B.shape)
        self.assertEqual(ss.c.shape, C.shape)
        self.assertEqual(ss.d.shape, D.shape)
        self.assertEqual(ss.time_step, time_step)
        self.assertEqual(ss.type, 'discrete')     
        self.assertEqual(B.all(), B_d_expected.all())       
        self.assertEqual(A.all(), A_d_expected.all())  
        self.assertEqual(C.all(), C_d_expected.all())
        self.assertEqual(D.all(), D_d_expected.all())

    def test_descritize_state_space_rejects_discrete_input_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        time_step = 0.1   
        ss_d = ilqr.stateSpace(A,B,C,D, time_step)
        with self.assertRaises(Exception) as assert_error_discrete:
            ss = ilqr.discretize_state_space(ss_d,time_step)
        self.assertEqual(str(assert_error_discrete.exception), 'input state space is already discrete')
    
    def test_descritize_state_space_rejects_invalid_c2d_method(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        time_step = 0.1   
        ss_d = ilqr.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_c2d_method:
            ss = ilqr.discretize_state_space(ss_d,time_step, c2d_method = 'lettuce')
        self.assertEqual(str(assert_error_c2d_method.exception), 'invalid discretization method')
     
    def test_descritize_state_space_rejects_singular_A_matrix_for_zoh(self):
        A = jnp.array([[0,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        time_step = 0.1   
        ss_d = ilqr.stateSpace(A,B,C,D)
        with self.assertRaises(Exception) as assert_error_singular_A_zoh:
            ss = ilqr.discretize_state_space(ss_d,time_step, c2d_method = 'zoh')
        self.assertEqual(str(assert_error_singular_A_zoh.exception), 'determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid'
)

class initialize_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        len_seq = data_and_funcs.len_seq
        x_len   = data_and_funcs.x_len
        u_len   = data_and_funcs.u_len
        p_N     = jnp.ones([x_len, 1])
        P_N     = jnp.ones([x_len, x_len])
        P_kp1, p_kp1, K_seq, d_seq, Del_V_vec_seq = ilqr.initialize_backwards_pass(P_N, p_N, len_seq)
        self.assertEqual(jnp.shape(P_kp1), (x_len, x_len))
        self.assertEqual(jnp.shape(p_kp1), (x_len, 1))
        self.assertEqual(len(K_seq), len_seq-1)
        self.assertEqual(len(K_seq), len_seq-1)
        self.assertEqual(len(Del_V_vec_seq), len_seq-1) 

class initialize_forwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        seed_state_vec    = shared_data_funcs.seed_state_vec
        len_seq           = shared_data_funcs.len_seq
        state_seq_updated, control_seq_updated, cost_float_updated, in_bounds_bool = ilqr.initialize_forwards_pass(seed_state_vec, len_seq)
        self.assertEqual(len(state_seq_updated), len_seq)
        self.assertEqual(len(control_seq_updated), len_seq-1)
        self.assertEqual(cost_float_updated, 0)
        self.assertEqual(in_bounds_bool, False)

class calculate_total_cost_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = [jnp.array([[1.],[1.]])] * 3
        u_seq = [jnp.array([[1.]])] * 3
        total_cost = ilqr.calculate_total_cost(cost_func, x_seq, u_seq)
        total_cost_expect = jnp.array([(0.5*(2+1)) + (0.5*(2+1)) + (0.5*(2+0))])
        self.assertEqual(jnp.shape(total_cost), (1,1))
        self.assertEqual(total_cost, total_cost_expect)        
        # TODO add assert for matching expected value

    def test_accepts_valid_inputs_from_shared(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = shared_data_funcs.state_seq
        u_seq = shared_data_funcs.control_seq
        total_cost = ilqr.calculate_total_cost(cost_func, x_seq, u_seq)
        self.assertEqual(jnp.shape(total_cost), (1,1))
        # TODO add assert for matching expected value        

class taylor_expand_cost_tests(unittest.TestCase):      
    def test_taylor_expand_cost_accepts_list_of_arrays(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        cost_func_params = shared_data_funcs.pend_unit_cost_func_params
        x_seq = shared_data_funcs.state_seq
        u_seq = shared_data_funcs.control_seq
        k_step = 1
        l_x, l_u, l_xx, l_uu, l_ux = ilqr.taylor_expand_cost(cost_func,
                                                            x_seq[k_step],
                                                            u_seq[k_step],
                                                            k_step)
        l_x_expected  = cost_func_params['Q'] @ (x_seq[k_step]-shared_data_funcs.des_state_seq[k_step])
        l_u_expected  = cost_func_params['R'] @ (u_seq[k_step]-shared_data_funcs.des_control_seq[k_step])
        l_xx_expected = cost_func_params['Q']
        l_uu_expected = cost_func_params['R']  
        print('l_x: ', l_x)  
        print('l_x_expected: ',l_x_expected)    
        print('state_length: ', len(x_seq[0]))
        self.assertEqual(l_x.all(),  l_x_expected.all())
        self.assertEqual(l_u.all(),  l_u_expected.all())
        self.assertEqual(l_xx.all(), l_xx_expected.all())
        self.assertEqual(l_uu.all(), l_uu_expected.all())
        self.assertEqual(l_ux.all(), jnp.zeros([len(u_seq[0]),len(x_seq[0])]).all())            

class taylor_expand_pseudo_hamiltonian_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()        
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_k = jnp.array([[1.],[1.]])
        u_k = jnp.array([[1.]])
        x_len = jnp.shape(x_k)[0]
        u_len = jnp.shape(u_k)[0]
        A_lin_k = jnp.array([[0,1],[1,1]])     
        B_lin_k = jnp.array([[0],[1]])
        p_kp1   = jnp.ones([x_len, 1])
        P_kp1   = jnp.ones([x_len, x_len])
        k_step  = 1
        q_x, q_u, q_xx, q_ux, q_uu = ilqr.taylor_expand_pseudo_hamiltonian(cost_func, 
                                                                           A_lin_k, 
                                                                           B_lin_k, 
                                                                           x_k, 
                                                                           u_k, 
                                                                           P_kp1, p_kp1, 
                                                                           k_step)
        self.assertEqual(jnp.shape(q_x),  (x_len, 1))
        self.assertEqual(jnp.shape(q_u),  (u_len, 1))
        self.assertEqual(jnp.shape(q_xx), (x_len,x_len))
        self.assertEqual(jnp.shape(q_uu), (u_len,u_len))
        self.assertEqual(jnp.shape(q_ux), (u_len,x_len)) 

    def test_accepts_valid_inputs_from_shared(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()        
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_seq = shared_data_funcs.state_seq
        u_seq = shared_data_funcs.control_seq
        x_len = len(x_seq[0])
        u_len = len(u_seq[0])
        A_lin_k = jnp.array([[1,1],[0,1]])     
        B_lin_k = jnp.array([[0],[1]])
        p_kp1   = jnp.ones([x_len, 1])
        P_kp1   = jnp.ones([x_len, x_len])
        k_step  = 1
        q_x, q_u, q_xx, q_ux, q_uu = ilqr.taylor_expand_pseudo_hamiltonian(cost_func, 
                                                                           A_lin_k, 
                                                                           B_lin_k, 
                                                                           x_seq[k_step], 
                                                                           u_seq[k_step], 
                                                                           P_kp1, p_kp1, 
                                                                           k_step)
        self.assertEqual(jnp.shape(q_x),  (x_len, 1))
        self.assertEqual(jnp.shape(q_u),  (u_len, 1))
        self.assertEqual(jnp.shape(q_xx), (x_len,x_len))
        self.assertEqual(jnp.shape(q_uu), (u_len,u_len))
        self.assertEqual(jnp.shape(q_ux), (u_len,x_len))  

class calculate_final_cost_to_go_approximation_tests(unittest.TestCase):
    def test_calculate_final_cost_to_go_approximation_accepts_valid_system(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        cost_func = shared_data_funcs.unit_cost_func_quad_state_and_control_pend_curried
        x_k = jnp.array(([1.0],[2.0]))
        u_k = jnp.array(([1.0]))
        P_N, p_N = ilqr.calculate_final_cost_to_go_approximation(cost_func,
                                                                x_k,
                                                                len(u_k))
        p_N_expected = (shared_data_funcs.pend_unit_cost_func_params['Qf'] @ (x_k))
        P_N_expected = shared_data_funcs.pend_unit_cost_func_params['Qf']
        # print('P_N_exp: ', P_N_expected)
        # print('p_N_exp: ', p_N_expected)
        self.assertEqual(p_N.all(), p_N_expected.all())
        self.assertEqual(P_N.all(), P_N_expected.all())

class calculate_backstep_ctg_approx_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        x_len = data_and_funcs.x_len
        u_len = data_and_funcs.u_len
        q_x   = jnp.ones([x_len, 1])
        q_u   = jnp.ones([u_len, 1])
        q_xx  = jnp.ones([x_len, x_len])
        q_uu  = jnp.ones([u_len, u_len])
        q_ux  = jnp.ones([u_len, x_len])
        K_k   = jnp.ones([u_len, x_len])
        d_k   = jnp.ones([u_len, 1])
        P_k, p_k, Del_V_vec_k = ilqr.calculate_backstep_ctg_approx(q_x, q_u, 
                                                                   q_xx, q_uu, q_ux, 
                                                                   K_k, d_k)
        P_k_expect = jnp.ones([x_len,x_len]) * 4.
        d_k_expect = jnp.ones([x_len,1]) * 4.
        Del_V_vec_k_expect = jnp.array([[1.0, 0.5]])
        self.assertEqual(jnp.shape(P_k), (x_len, x_len))
        self.assertEqual(jnp.shape(p_k), (x_len, 1))  
        self.assertEqual(jnp.shape(Del_V_vec_k), (1,2))      
        self.assertEqual(P_k.all(), P_k_expect.all())
        self.assertEqual(d_k.all(), d_k_expect.all()) 
        self.assertEqual(Del_V_vec_k.all(), Del_V_vec_k_expect.all())

class calculate_optimal_gains_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        q_uu = jnp.array([[1.,1.],[0,1.]])
        q_ux = jnp.array([[1.,1.],[0,1.]])
        q_u  = jnp.array([[1.],[1.]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        print(jnp.linalg.inv(q_uu))
        K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
        d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
        K_k_expected = -(jnp.array([[1.,-1.],[0,1.]])) @ (jnp.array([[1.,1.],[0,1.]]))
        d_k_expected = -(jnp.array([[1.,-1.],[0,1.]])) @ (jnp.array([[1.],[1.]]))
        self.assertEqual(K_k.all(), K_k_expected.all())
        self.assertEqual(d_k.all(), d_k_expected.all())

    def test_accepts_valid_inputs_from_shared(self):
        q_uu = jnp.array([[1.,1.],[0,1.]])
        q_ux = jnp.array([[0.1,0.2,0.3,0.4],[1., 2., 3., 4.]])
        q_u  = jnp.array([[1],[2]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
        d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
        self.assertEqual(K_k.all(), K_k_expected.all())
        self.assertEqual(d_k.all(), d_k_expected.all())    

class calculate_u_k_new_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = jnp.array([[1.0]])
        x_nom_k = jnp.array([[1.0],[1.0]])
        x_new_k = x_nom_k + 1.0
        K_k     = jnp.ones([data_and_funcs.u_len, data_and_funcs.x_len])
        d_k     = jnp.ones([data_and_funcs.u_len, 1])
        line_search_factor = 1
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)

        # u_k_expected = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
        u_k_expected = jnp.array([[1. + (1.* (2.-1.) + 1. * (2.-1.)) + 1.*1.]])
        self.assertEqual(u_k_updated_new, u_k_expected)

    def test_accepts_valid_inputs_with_shared_data(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = data_and_funcs.control_seq[k_step]
        x_nom_k = data_and_funcs.state_seq[k_step]
        x_new_k = x_nom_k + 0.1
        K_k     = data_and_funcs.K_seq[k_step]
        d_k     = data_and_funcs.d_seq[k_step]
        line_search_factor = 0.5
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)

        u_k_expected = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
        self.assertEqual(u_k_updated_new, u_k_expected)    

class calculate_cost_decrease_ratio_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        prev_cost_float = 100.0
        new_cost_float  = 70.0
        Del_V_seq = [jnp.array([[1,2]])] * 3
        line_search_factor = 2
        cost_decrease_ratio = ilqr.calculate_cost_decrease_ratio(prev_cost_float,
                                                                 new_cost_float,
                                                                 Del_V_seq,
                                                                 line_search_factor)
        del_V_sum_exp = ((1+1+1) * line_search_factor) + ((2+2+2) * line_search_factor**2)
        expected_output = jnp.array((1 / del_V_sum_exp) * (prev_cost_float - new_cost_float))
        self.assertEqual(cost_decrease_ratio, expected_output)

class calculate_expected_cost_decrease_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        Del_V_vec_seq = [jnp.array([[1,2]])] * 3
        line_search_factor = 2
        Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
        expected_output = ((1+1+1) * line_search_factor) + ((2+2+2) * line_search_factor**2)
        self.assertEqual(Del_V_sum, expected_output)

class analyze_cost_decrease_tests(unittest.TestCase):
    def test_returns_true_for_in_bound_inputs(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config['cost_ratio_bounds']
        cost_decrease_ratio = bounds[0] + (bounds[0] + bounds[1]) * 0.5
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, True)
 
    def test_returns_false_for_input_too_high(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config['cost_ratio_bounds']
        cost_decrease_ratio = bounds[1] + 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)

    def test_returns_false_for_input_too_low(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config['cost_ratio_bounds']
        cost_decrease_ratio = bounds[0] - 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)

if __name__ == '__main__':
    unittest.main()
