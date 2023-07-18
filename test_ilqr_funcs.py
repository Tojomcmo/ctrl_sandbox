import unittest
import numpy as np
import ilqr_funcs as ilqr
from jax import numpy as jnp

# for each input
#   test one known case
#   if discrete change of behavior (abs value add, check pos and neg)
#   check edge cases (NaN, overflow, string, divide by zero)

class state_space_tests(unittest.TestCase):
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

class discretize_state_space_tests(unittest.TestCase):

    def test_discretize_state_space_accepts_valid_system(self):
        A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
        B = jnp.array([[1,1],[0,1],[0,0]])
        C = jnp.array([[1,0,1]])
        D = jnp.array([[0,0]])
        ss_c = ilqr.stateSpace(A,B,C,D)
        time_step = 0.1
        c2d_methods = ['Euler', 'zoh', 'zohCombined']
        for c2d_method  in c2d_methods:
            ss = ilqr.discretize_state_space(ss_c,time_step,c2d_method)
            self.assertEqual(ss.a.shape, A.shape)
            self.assertEqual(ss.b.shape, B.shape)
            self.assertEqual(ss.c.shape, C.shape)
            self.assertEqual(ss.d.shape, D.shape)
            self.assertEqual(ss.time_step, time_step)
            self.assertEqual(ss.type, 'discrete')

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

class linearize_dynamics_tests(unittest.TestCase):

    def test_linearize_dynamics_accepts_valid_case(self):
        def dyn_func(t, x, u):
            y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
            return y
        x = jnp.array([0.1, 0.2, 0.3])
        u = jnp.array([0.1, 0.2])
        A_lin_expected = jnp.array([
                [4*(x[0]**3), 2*(x[1]**1),  1*jnp.cos(x[2])],
                [3*(x[0]**2), 3*(x[1]**2), -2*jnp.sin(x[2])],
                [2*(x[0]**1), 4*(x[1]**3),  3*jnp.cos(x[2])]
                ])
        B_lin_expected = jnp.array([
                [2*(u[0]**1), 4*(u[1]**3)],
                [3*(u[0]**2), 3*(u[1]**2)],
                [4*(u[0]**3), 2*(u[1]**1)]
                ])       
        time_dyn = None
        A_lin, B_lin = ilqr.linearize_dynamics(dyn_func, x, u)
        print('A_lin :', A_lin)
        print(A_lin_expected)
        print(B_lin)
        print(B_lin_expected)
        self.assertEqual(A_lin.tolist(), A_lin_expected.tolist())
        self.assertEqual(B_lin.tolist(), B_lin_expected.tolist())

class calculate_linearized_state_space_seq_tests(unittest.TestCase):

    def dyn_func(self, t, x, u):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y

    def test_calculate_linearized_state_space_seq_accepts_valid_system(self):
        time_step    = 0.1
        num_steps    = 20
        num_states   = 3
        num_controls = 2
        control_seq  = jnp.ones((num_steps, num_controls))
        state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
        A_lin_array_expected_shape = (num_steps, num_states, num_states)
        B_lin_array_expected_shape = (num_steps, num_states, num_controls)
        A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
        self.assertEqual(A_lin_array.shape, A_lin_array_expected_shape)
        self.assertEqual(B_lin_array.shape, B_lin_array_expected_shape)

    def test_calculate_linearized_state_space_seq_rejects_invalid_sequence_dimensions(self):
        time_step    = 0.1
        num_steps    = 20
        num_states   = 3
        num_controls = 2
        control_seq  = jnp.ones((num_steps, num_controls, 2))
        state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
        with self.assertRaises(Exception) as assert_seq_dim_error:
            A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
        self.assertEqual(str(assert_seq_dim_error.exception), 'state or control sequence is incorrect array dimension. sequences must be 2d arrays')

    def test_calculate_linearized_state_space_seq_rejects_invalid_sequence_lengths(self):
        time_step    = 0.1
        num_steps    = 20
        num_states   = 3
        num_controls = 2
        control_seq  = jnp.ones((num_steps, num_controls))
        state_seq    = jnp.ones((num_steps+2, num_states)) * 0.1
        with self.assertRaises(Exception) as assert_seq_len_error:
            A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
        self.assertEqual(str(assert_seq_len_error.exception), 'state and control sequences are incompatible lengths. state seq must be control seq length +1')

    def test_calculate_linearized_state_space_seq_rejects_invalid_time_step(self):
        time_step    = -0.1
        num_steps    = 20
        num_states   = 3
        num_controls = 2
        control_seq  = jnp.ones((num_steps, num_controls))
        state_seq    = jnp.ones((num_steps+1, num_states)) * 0.1
        with self.assertRaises(Exception) as assert_seq_len_error:
            A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(self.dyn_func, state_seq, control_seq, time_step)
        self.assertEqual(str(assert_seq_len_error.exception), 'time_step is invalid. Must be positive float')

class simulate_forward_dynamics(unittest.TestCase):

    def dyn_func(self, t, x, u):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y

    def test_simulate_forward_dynamics_accepts_valid_system(self):
        self.assertEqual(True,True)

class calculate_forward_rollout(unittest.TestCase):

    def dyn_func(self, t, x, u):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y

    def test_calculate_forward_rollout_accepts_valid_system(self):
        num_steps   = 20
        time_step   = 0.1
        control_seq = jnp.ones((num_steps, 1))
        state_init  = jnp.array([[0.1, 0.1]])
        kwparams    = {'sim_method':'solve_ivp_zoh', 'c2d_method':'zohCombined'}
 #       Ad_seq, Bd_seq, time_seq = ilqr.calculate_forward_rollout(self.dyn_func, self.cost_func state_init, control_seq, time_step, **kwparams)
        self.assertEqual(True, True)

class taylor_expand_cost(unittest.TestCase):

    def cost_func_quad_state_and_control(self, cost_func_params, state_vec, control_vec, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q = cost_func_params['Q']
        R = cost_func_params['R']
        Qf = cost_func_params['Qf']
        # check that dimensions match [TODO]
        if is_final_bool:
            cost = (0.5) * (  state_vec   @ Qf @ jnp.transpose(state_vec  ))
        else:
            cost = (0.5) * ( (state_vec   @ Q  @ jnp.transpose(state_vec  ))
                            +(control_vec @ R  @ jnp.transpose(control_vec)))
        return cost

    def test_taylor_expand_cost_accepts_valid_system(self):
        func_params = {
                        'Q'  : jnp.array([[1.,0,0],[0,10.,0],[0,0,1.]]),
                        'R'  : jnp.array([[1.,0],[0,10.]]),
                        'Qf' : jnp.array([[1.,0,0],[0,10.,0],[0,0,10.]])
                      } 
        x_k = np.array(([1.0,2.0,3.0]))
        u_k = np.array(([4.0,5.0]))
        l_x, l_u, l_xx, l_uu, l_ux = ilqr.taylor_expand_cost(self.cost_func_quad_state_and_control,
                                                            func_params,
                                                            x_k,
                                                            u_k)
        self.assertEqual(l_x.all(), (func_params['Q'] @ x_k).all())
        self.assertEqual(l_u.all(), (func_params['R'] @ u_k).all())
        self.assertEqual(l_xx.all(), func_params['Q'].all())
        self.assertEqual(l_uu.all(), func_params['R'].all())
        self.assertEqual(l_ux.all(), jnp.zeros([len(u_k),len(x_k)]).all())            

class calculate_final_cost_to_go_approximation(unittest.TestCase):
    
    def cost_func_quad_state_and_control(self, cost_func_params, state_vec, control_vec, is_final_bool=False):
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # S[in]     - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q = cost_func_params['Q']
        R = cost_func_params['R']
        Qf = cost_func_params['Qf']
        # check that dimensions match [TODO]
        if is_final_bool:
            cost = (0.5) * (  state_vec   @ Qf @ jnp.transpose(state_vec  ))
        else:
            cost = (0.5) * ( (state_vec   @ Q  @ jnp.transpose(state_vec  ))
                            +(control_vec @ R  @ jnp.transpose(control_vec)))
        return cost
    
    def test_calculate_final_cost_to_go_approximation_accepts_valid_system(self):
        func_params = {
                        'Q'  : jnp.array([[1.,0,0],[0,10.,0],[0,0,1.]]),
                        'R'  : jnp.array([[1.,0],[0,10.]]),
                        'Qf' : jnp.array([[1.,0,0],[0,10.,0],[0,0,10.]])
                      } 
        x_k = np.array(([1.0,2.0,3.0]))
        u_k = np.array(([4.0,5.0]))
        P_N, p_N = ilqr.calculate_final_cost_to_go_approximation(self.cost_func_quad_state_and_control,
                                                                func_params,
                                                                x_k,
                                                                len(u_k))
        self.assertEqual(p_N.all(), (func_params['Qf'] @ x_k).all())
        self.assertEqual(P_N.all(), func_params['Qf'].all())

# class taylor_expand_pseudo_hamiltonian_tests(unittest.TestCase):

# class calculate_backwards_pass(unittest.TestCase):

if __name__ == '__main__':
    unittest.main()