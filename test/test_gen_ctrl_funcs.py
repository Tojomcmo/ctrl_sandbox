import unittest
from jax import numpy as jnp
from jax import lax
import numpy as np
import scipy
import numpy.typing as npt
import numpy.testing
from typing import Callable, Tuple
import gen_ctrl_funcs as gen_ctrl
import ilqr_utils as util
 

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
    def pend_unit_dyn_func(self, state:jnp.ndarray, control:jnp.ndarray)-> jnp.ndarray:
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
        u_seq     = np.ones((4,1))
        x_init      = np.array([1.,1.])
        x_seq       = gen_ctrl.simulate_forward_dynamics_seq(self.pend_unit_dyn_func, x_init, u_seq)
        # calculate expected output
        x_seq_expected    = np.zeros([5,2])
        x_seq_expected[0] = x_init
        for idx in range(len_seq):
            x_delta    = self.pend_unit_dyn_func(x_seq[idx], u_seq[idx])
            x_next       = x_delta
            x_seq_expected[idx+1] = x_next
        self.assertEqual(len(x_seq), len_seq +1)
        self.assertEqual(x_seq[-1].tolist(), x_seq_expected[-1].tolist())

class linearize_dynamics_tests(unittest.TestCase):

    def pend_unit_dyn_func(self, state:jnp.ndarray, control:jnp.ndarray)-> jnp.ndarray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([x_k_jax[1],
                             -(b/l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g/l) + u_k_jax[0]], dtype=float)
        return state_dot  
       
    def step_rk4(self,
            dyn_func_ad:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
            h:float, 
            x_k:jnp.ndarray, 
            u_k:jnp.ndarray) -> jnp.ndarray:
        #rk4 integration with zero-order hold on u
        f1 = dyn_func_ad(x_k            , u_k)
        f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
        f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
        f4 = dyn_func_ad(x_k + h*f3     , u_k)
        return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)
    
    def test_linearize_dynamics_accepts_valid_case(self):
        h = 0.1
        x = jnp.array([1., 1.])
        u = jnp.array([1.])
        discrete_dyn_func = lambda x,u: self.step_rk4(self.pend_unit_dyn_func,h,x,u)
        xu = jnp.concatenate((x, u))
        a_lin, b_lin    = gen_ctrl.linearize_dynamics(discrete_dyn_func,xu,len(x))
        g = 1.0
        l = 1.0
        b = 1.0
        state_pos = x[0]
        A_lin_expected = np.array([[0, 1],[-(g/l)*jnp.cos(state_pos), -(b/l)]])
        B_lin_expected = np.array([[0],[1]])
        A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
        B_lin_d_expected = B_lin_expected * h
        self.assertEqual(jnp.shape(a_lin), (len(x), len(x)))
        self.assertEqual(jnp.shape(b_lin), (len(x), len(u)))
        numpy.testing.assert_allclose(a_lin, A_lin_d_expected,rtol=1e-1, atol=1e-1)
        numpy.testing.assert_allclose(b_lin, B_lin_d_expected,rtol=1e-1, atol=1e-1)

class calculate_linearized_state_space_seq_tests(unittest.TestCase):

    def pend_unit_dyn_func(self, state:jnp.ndarray, control:jnp.ndarray)-> jnp.ndarray:
        x_k_jax = jnp.array(state)
        u_k_jax = jnp.array(control)
        g = 1.0
        l = 1.0
        b = 1.0
        state_dot = jnp.array([x_k_jax[1],
                    -(b/l) * x_k_jax[1] - (jnp.sin(x_k_jax[0]) * g/l) + u_k_jax[0]], dtype=float)
        return state_dot  
    
    def step_rk4(self,
            dyn_func_ad:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
            h:float, 
            x_k:jnp.ndarray, 
            u_k:jnp.ndarray) -> jnp.ndarray:
        #rk4 integration with zero-order hold on u
        f1 = dyn_func_ad(x_k            , u_k)
        f2 = dyn_func_ad(x_k + 0.5*h*f1 , u_k)
        f3 = dyn_func_ad(x_k + 0.5*h*f2 , u_k)
        f4 = dyn_func_ad(x_k + h*f3     , u_k)
        return x_k + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)
    
    def test_calculate_linearized_state_space_seq_accepts_valid_system(self):
        len_seq     = 4
        h           = 0.005
        u_seq = jnp.ones((3,1), dtype=np.float64)
        x_seq = jnp.ones((4,2), dtype=np.float64)
        discrete_dyn_func = lambda x,u: self.step_rk4(self.pend_unit_dyn_func,h,x,u)
        A_lin_d_seq, B_lin_d_seq = gen_ctrl.calculate_linearized_state_space_seq(discrete_dyn_func,
                                                                                x_seq, 
                                                                                u_seq)
        # calculate expected output
        x_init_pos = x_seq[0]
        A_lin_expected = np.array([[0, 1],[-(1./1.) * np.cos(x_init_pos[0]), -(1./1.)]])
        B_lin_expected = np.array([[0],[1]])
        A_lin_d_expected = np.eye(2) + (A_lin_expected * h)
        B_lin_d_expected = B_lin_expected * h
        #assert validity
        self.assertEqual(len(A_lin_d_seq), len_seq-1)
        self.assertEqual(len(B_lin_d_seq), len_seq-1)
        numpy.testing.assert_allclose(A_lin_d_seq[0], A_lin_d_expected,rtol=1e-4, atol=1e-4)
        numpy.testing.assert_allclose(B_lin_d_seq[0], B_lin_d_expected,rtol=1e-4, atol=1e-4)
        
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
                                         x_k:jnp.ndarray, 
                                         u_k:jnp.ndarray, 
                                         k:int) -> Tuple[float,float,float]:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array([[1.,0],[0,1.]])
        R  = jnp.array([[1.]])
        Qf = jnp.array([[1.,0],[0,1.]])
        x_des_seq   = jnp.ones([3,2,1]) * 0.1
        u_des_seq   = jnp.ones([2,1,1]) * 0.1
        len_seq     = len(x_des_seq)
        # check that dimensions match [TODO]
        if k == len_seq-1:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k == 0:
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
        return total_cost.item(), x_cost.item(), u_cost.item()                      

    def test_accepts_valid_inputs(self):
        cost_func_for_calc = self.cost_func_quad_state_and_control
        x_seq = jnp.ones([3,2])
        u_seq = jnp.ones([2,1])
        total_cost, cost_seq, x_cost_seq, u_cost_seq = gen_ctrl.calculate_total_cost(cost_func_for_calc, x_seq, u_seq)
        total_cost_expect = (0.5*(0.9*0.9)) + (0.5*(2*(0.9*0.9)+(0.9*0.9))) + (0.5*(2*(0.9*0.9)))
        cost_seq_expect   = np.array([0.5*(0.9*0.9), (0.5*(2*(0.9*0.9)+(0.9*0.9))), (0.5*(2*(0.9*0.9)))]) 
        x_cost_seq_expect = np.array([0.0, (0.9*0.9), 0.9*0.9])
        u_cost_seq_expect = np.array([0.5*(0.9*0.9), 0.5*(0.9*0.9), 0.0])
        self.assertAlmostEqual(total_cost, total_cost_expect,places=6) 
        numpy.testing.assert_allclose(cost_seq, cost_seq_expect,rtol=1e-6, atol=1e-6)        
        numpy.testing.assert_allclose(x_cost_seq, x_cost_seq_expect,rtol=1e-6, atol=1e-6) 
        numpy.testing.assert_allclose(u_cost_seq, u_cost_seq_expect,rtol=1e-6, atol=1e-6) 
                    

class taylor_expand_cost_tests(unittest.TestCase):  
    
    def cost_func_quad_state_and_control(self, 
                                         x_k:jnp.ndarray, 
                                         u_k:jnp.ndarray, 
                                         k:int) -> jnp.ndarray:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array([[1.,0],[0,1.]])
        R  = jnp.array([[1.]])
        Qf = jnp.array([[1.,0],[0,1.]])
        x_des_seq   = jnp.ones([3,2]) * 0.1
        u_des_seq   = jnp.ones([2,1]) * 0.1
        len_seq     = len(x_des_seq)
        # check that dimensions match [TODO]
        if k == len_seq -1:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k == 0:
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
        return total_cost             
 
    def test_taylor_expand_cost_accepts_list_of_arrays(self):
        cost_func_for_diff = self.cost_func_quad_state_and_control
        x_seq      = jnp.ones([3,2])
        u_seq      = jnp.ones([2,1])
        k     = 1
        xu_seq  = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
        l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost(cost_func_for_diff,xu_seq[k],len(x_seq[0]),k) #type:ignore (Jax NDArray)
        cost_func_params  = {'Q'  : np.array([[1.,0],[0,1.]]), 
                             'R'  : np.array([[1.]]),
                             'Qf' : np.array([[1.,0],[0,1.]])} 
        x_des_seq  = np.ones([3,2]) * 0.1
        u_des_seq  = np.ones([2,1]) * 0.1
        l_x_expected  = cost_func_params['Q'] @ (x_seq[k]-x_des_seq[k]).reshape(-1,1)
        l_u_expected  = cost_func_params['R'] @ (u_seq[k]-u_des_seq[k]).reshape(-1,1)
        l_xx_expected = cost_func_params['Q']
        l_uu_expected = cost_func_params['R']  
        l_ux_expected = np.zeros([len(u_seq[0]),len(x_seq[0])])

        numpy.testing.assert_allclose(l_x, l_x_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_u, l_u_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_xx, l_xx_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_uu, l_uu_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_ux, l_ux_expected,rtol=1e-7, atol=1e-7)            


class taylor_expand_cost_seq_tests(unittest.TestCase):  
    
    def cost_func_quad_state_and_control_old(self, 
                                         x_k:jnp.ndarray, 
                                         u_k:jnp.ndarray, 
                                         k:int) -> jnp.ndarray:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array([[1.,0],[0,1.]])
        R  = jnp.array([[1.]])
        Qf = jnp.array([[1.,0],[0,1.]])
        x_des_seq   = jnp.ones([3,2]) * 0.1
        u_des_seq   = jnp.ones([2,1]) * 0.1
        # check that dimensions match [TODO]
        if k == 0:
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        elif k == len(x_des_seq) -1:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        else:
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
        return total_cost             
 
    def cost_func_quad_state_and_control(self, 
                                         x_k:jnp.ndarray, 
                                         u_k:jnp.ndarray, 
                                         k:int) -> jnp.ndarray:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array([[1.,0],[0,1.]])
        R  = jnp.array([[1.]])
        Qf = jnp.array([[1.,0],[0,1.]])
        x_des_seq   = jnp.ones([3,2]) * 0.1
        u_des_seq   = jnp.ones([2,1]) * 0.1
        # check that dimensions match [TODO]
        def first_index_case():
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
            return total_cost
        def last_index_case():
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
            return total_cost
        def default_case():
            x_k_corr = x_k.reshape(-1,1) - (x_des_seq[k]).reshape(-1,1)   
            u_k_corr = u_k.reshape(-1,1) - (u_des_seq[k]).reshape(-1,1)   
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
            return total_cost
        is_first_case = (k == 0)
        is_last_case  = (k == len(x_des_seq)-1)
        return lax.cond(is_first_case, first_index_case, lambda : lax.cond(is_last_case, last_index_case, default_case))   

    def test_taylor_expand_cost_accepts_list_of_arrays(self):
        cost_func_for_diff = self.cost_func_quad_state_and_control
        x_seq      = jnp.ones([3,2])
        u_seq      = jnp.ones([2,1])
        l_x, l_u, l_xx, l_uu, l_ux = gen_ctrl.taylor_expand_cost_seq(cost_func_for_diff,x_seq, u_seq)
        cost_func_params  = {'Q'  : np.array([[1.,0],[0,1.]]), 
                             'R'  : np.array([[1.]]),
                             'Qf' : np.array([[1.,0],[0,1.]])} 
        x_des_seq  = np.ones([3,2]) * 0.1
        u_des_seq  = np.ones([2,1]) * 0.1
        l_x_expected  = cost_func_params['Q'] @ (x_seq[1]-x_des_seq[1]).reshape(-1,1)
        l_u_expected  = cost_func_params['R'] @ (u_seq[1]-u_des_seq[1]).reshape(-1,1)
        l_xx_expected = cost_func_params['Q']
        l_uu_expected = cost_func_params['R']  
        l_ux_expected = np.zeros([len(u_seq[0]),len(x_seq[0])])

        numpy.testing.assert_allclose(l_x[1], l_x_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_u[1], l_u_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_xx[1], l_xx_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_uu[1], l_uu_expected,rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(l_ux[1], l_ux_expected,rtol=1e-7, atol=1e-7)            


class prep_xu_vecs_for_diff_tests(unittest.TestCase):
    def test_accepts_valid_np_col_inputs(self):
        #create test conditions
        x_k = np.array([1,2,3])
        u_k = np.array([4,5])
        #test function
        xu_k_jax, xu_k_len, x_k_len = gen_ctrl.prep_xu_vec_for_diff(x_k, u_k)
        # create expected output
        xu_k_jax_expected = jnp.array([1,2,3,4,5])
        xu_k_len_expected = 5
        x_k_len_expected  = 3
        #compare outputs
        self.assertEqual(jnp.shape(xu_k_jax), (5,))
        self.assertEqual(xu_k_jax.tolist(), xu_k_jax_expected.tolist())
        self.assertEqual(xu_k_len, xu_k_len_expected)
        self.assertEqual(x_k_len, x_k_len_expected)


class prep_cost_func_for_diff_tests(unittest.TestCase):
    def cost_func(self, vec_1:jnp.ndarray, vec_2:jnp.ndarray, input_int:int): 
        cost_val_1 = jnp.sum(vec_1)
        cost_val_2 = jnp.sum(vec_2)
        combined_cost = cost_val_1 + cost_val_2
        return combined_cost, cost_val_1, cost_val_2        
    
    def test_accepts_valid_cost_func_inputs(self):
        #create test conditions
        x_k = jnp.array([1,2])
        u_k = jnp.array([3,4])
        input_int = 1
        #test function
        cost_func_for_diff = gen_ctrl.prep_cost_func_for_diff(self.cost_func) # type: ignore (NDArray to Jax NDArray)
        combined_cost = cost_func_for_diff(x_k, u_k, input_int)
        # create expected output
        combined_cost_expected = jnp.array([10])
        #compare outputs
        self.assertEqual(combined_cost, combined_cost_expected)

if __name__ == '__main__':
    unittest.main()
