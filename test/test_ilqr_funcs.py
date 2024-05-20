import unittest
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple
from jax import numpy as jnp

import ilqr_funcs as ilqr
import ilqr_utils as util
import cost_functions as cost
import dyn_functions as dyn
import gen_ctrl_funcs as gen_ctrl

# for each input
#   test one known case
#   if discrete change of behavior (abs value add, check pos and neg)
#   check edge cases (NaN, overflow, string, divide by zero)

class shared_unit_test_data_and_funcs:
    def __init__(self) -> None:
        self.cost_func_quad_unit_params = cost.costFuncQuadStateAndControlParams(
                                                                Q  = np.array([[1.,0],[0,1.]]),
                                                                R  = np.array([[1.]]),
                                                                Qf = np.array([[1.,0],[0,1.]]),
                                                                x_des_seq=np.zeros((5,2)),
                                                                u_des_seq=np.zeros((4,1)))

        self.pend_dyn_obj = dyn.single_pend_dyn(m = 1.0, moi = 0.0, g=1.0, b=1.0, l=1.0)
        self.cost_func_obj = cost.cost_quad_x_and_u(self.cost_func_quad_unit_params.Q,
                                                    self.cost_func_quad_unit_params.R,
                                                    self.cost_func_quad_unit_params.Qf,
                                                    self.cost_func_quad_unit_params.x_des_seq,
                                                    self.cost_func_quad_unit_params.u_des_seq,)
        self.ilqr_config = ilqr.ilqrConfigStruct(num_states=2, num_controls=1, len_seq=5, time_step=0.1)
        self.ilqr_config.config_cost_func(self.cost_func_obj.cost_func_quad_state_and_control_for_diff,
                                          self.cost_func_obj.cost_func_quad_state_and_control_for_diff)
        self.ilqr_config.config_for_dyn_func(self.pend_dyn_obj.cont_dyn_func, gen_ctrl.step_rk4)
        self.ilqr_config.create_curried_funcs()

        self.seed_x_vec       = np.ones([self.ilqr_config.num_states]) 
        self.seed_u_seq       = np.ones([self.ilqr_config.len_seq-1,self.ilqr_config.num_controls])

        self.x_seq_example            = np.ones([self.ilqr_config.len_seq,2])
        self.x_seq_example[:,1]      *= 2.0
        self.u_seq_example            = np.ones([self.ilqr_config.len_seq-1,1])

        self.K_seq_example            = np.ones([self.ilqr_config.len_seq, self.ilqr_config.num_controls, self.ilqr_config.num_states])
        self.d_seq_example            = np.ones([self.ilqr_config.len_seq, self.ilqr_config.num_controls, 1         ])
        self.Del_V_vec_seq_example    = np.ones([self.ilqr_config.len_seq, 1                            , 2         ])

        self.ilqr_ctrl_state = ilqr.ilqrControllerState(self.ilqr_config, self.seed_x_vec, self.seed_u_seq)

    def gen_dyn_func_full(self, params, x, u):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y
    
    def gen_dyn_func_curried(self, x, u):
        y = jnp.array([
                    x[0]**4 + x[1]**2 + 1*jnp.sin(x[2]) + u[0]**2 + u[1]**4,
                    x[0]**3 + x[1]**3 + 2*jnp.cos(x[2]) + u[0]**3 + u[1]**3,
                    x[0]**2 + x[1]**4 + 3*jnp.sin(x[2]) + u[0]**4 + u[1]**2,
                    ])
        return y
    
    def pend_dyn_nl(self, params, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        state.reshape(-1)
        state_dot = jnp.array([
                    state[1],
                    -(params.b/params.l) * state[1] - (jnp.sin(state[0]) * params.g/params.l) + control[0]
                    ])
        return state_dot
    
    def pend_dyn_nl_func_unit_param_curried(self, state, control):
    # continuous time dynamic equation for simple pendulum 
    # time[in]       - time component, necessary prarmeter for ode integration
    # state[in]      - vector of state variables, 2 values, [0]: theta, [1]: theta dot, pend down is zero
    # control[in]    - vector of control variables, 1 value, [0]: torque, positive torque causes counter clockwise rotation (Right hand rule out of page)
    # params[in]     - g: gravity[m/s^2] (positive down), l: pend length[m], b: damping[Ns/m]
    # state_dot[out] - vector of state derivatives, corresponding to the time derivatives the state vector, [0]: theta dot, [1]: theta ddot
        state.reshape(-1)
        g = 1.0
        l = 1.0
        b = 1.0
        pos = state[0]
        vel = state[1]
        tq  = control[0]
        state_dot = jnp.array([
                    vel,
                    -(b/l) * vel - (jnp.sin(pos) * g/l) + tq
                    ])
        return state_dot  

    def cost_func_quad_state_and_control(self, 
                                         cost_func_params:cost.costFuncQuadStateAndControlParams, 
                                         x_k:npt.NDArray[np.float64], 
                                         u_k:npt.NDArray[np.float64], 
                                         k_step:int, 
                                         is_final_bool:bool)-> Tuple[npt.NDArray,npt.NDArray,npt.NDArray]:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array(cost_func_params.Q)
        R  = jnp.array(cost_func_params.R)
        Qf = jnp.array(cost_func_params.Qf)
        # check that dimensions match [TODO]
        if is_final_bool:
            x_k_corr = util.calc_and_shape_array_diff(x_k  , cost_func_params.x_des_seq[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k_step == 0:
            u_k_corr   = util.calc_and_shape_array_diff(u_k, cost_func_params.u_des_seq[k_step], shape='col')
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr)  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = util.calc_and_shape_array_diff(x_k, cost_func_params.x_des_seq[k_step], shape='col')   
            u_k_corr = util.calc_and_shape_array_diff(u_k, cost_func_params.u_des_seq[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (jnp.transpose(x_k_corr) @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (jnp.transpose(u_k_corr) @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((jnp.transpose(x_k_corr) @ Q @ x_k_corr)+(jnp.transpose(u_k_corr) @ R @ u_k_corr)))
        return total_cost, x_cost, u_cost   # type:ignore (Jax NDArrays)               

    def cost_func_quad_state_and_control_unit_param_curried(self, 
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
        total_cost, x_cost, u_cost = cost_func_curried(x_k, u_k, k_step, is_final_bool)
        return total_cost, x_cost, u_cost   
    
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
        ctrl_state = data_and_funcs.ilqr_ctrl_state
        x_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ctrl_state) 
        self.assertEqual((np.abs(cost_float-prev_cost_float) > ilqr_config.converge_crit), True)   

class run_ilqr_controller_tests(unittest.TestCase):
    def test_accepts_valid_linear_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config     = data_and_funcs.ilqr_config
        ilqr_ctrl_state = data_and_funcs.ilqr_ctrl_state
        state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, ilqr_ctrl_state)
        ilqr_ctrl_state.x_seq = state_seq
        ilqr_ctrl_state.cost_float = cost_float
        ilqr_ctrl_state.prev_cost_float = prev_cost_float
        ilqr_ctrl_state.cost_seq = cost_seq
        ilqr_ctrl_state = ilqr.run_ilqr_controller(ilqr_config, ilqr_ctrl_state)
        self.assertEqual(True,True)

class calculate_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_linear_dyn_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        
        k_seq, d_seq, Del_V_vec_seq, ro_reg = ilqr.calculate_backwards_pass(data_and_funcs.ilqr_config,
                                                                            data_and_funcs.ilqr_ctrl_state)
        self.assertEqual(len(k_seq),data_and_funcs.ilqr_config.len_seq-1)
        self.assertEqual(len(d_seq),data_and_funcs.ilqr_config.len_seq-1)
        self.assertEqual(len(Del_V_vec_seq),data_and_funcs.ilqr_config.len_seq-1) 
        self.assertEqual(jnp.shape(k_seq[0]),(data_and_funcs.ilqr_config.num_controls, data_and_funcs.ilqr_config.num_states))                  
        self.assertEqual(jnp.shape(d_seq[0]),(data_and_funcs.ilqr_config.num_controls, 1))                       
        self.assertEqual(jnp.shape(Del_V_vec_seq[0]),(2,)) 

class calculate_forwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_linear_dyn_inputs(self):
        shared_data_and_funcs = shared_unit_test_data_and_funcs()
        ilqr_config = shared_data_and_funcs.ilqr_config
        ilqr_ctrl_state  = shared_data_and_funcs.ilqr_ctrl_state

        state_seq, cost_float, prev_cost_float, cost_seq = ilqr.initialize_ilqr_controller(ilqr_config, 
                                                                                           ilqr_ctrl_state)
        ilqr_ctrl_state.x_seq = state_seq
        ilqr_ctrl_state.cost_float = cost_float
        ilqr_ctrl_state.prev_cost_float = prev_cost_float
        k_seq, d_seq, Del_V_vec_seq, ro_reg = ilqr.calculate_backwards_pass(ilqr_config, ilqr_ctrl_state)
        ilqr_ctrl_state.K_seq         = k_seq
        ilqr_ctrl_state.d_seq         = d_seq
        ilqr_ctrl_state.Del_V_vec_seq = Del_V_vec_seq                
        # ctrl_state.K_seq         = data_and_funcs.K_seq
        # ctrl_state.d_seq         = data_and_funcs.d_seq
        # ctrl_state.Del_V_vec_seq = data_and_funcs.Del_V_vec_seq
        x_seq_new, u_seq_new, cost_float_new,cost_seq_new,x_cost_seq_new, u_cost_seq_new,line_search_factor, ro_reg_change_bool = ilqr.calculate_forwards_pass(ilqr_config, ilqr_ctrl_state)
        self.assertEqual(True,True)

class simulate_forwards_pass_tests(unittest.TestCase):
    def example_cost_func(self,x_k:jnp.ndarray, u_k:jnp.ndarray, k:int)->jnp.ndarray:
        x = x_k.reshape(-1,1)
        u = u_k.reshape(-1,1)
        cost_float = (x.T@x + u.T@u )
        return cost_float
    
    def example_dyn_func(self, x:jnp.ndarray, u:jnp.ndarray)->jnp.ndarray:
        A = jnp.eye(len(x))
        B = jnp.ones((len(x),len(u)))    
        x_new = A@x.reshape(-1,1) + B@u.reshape(-1,1)
        return x_new.reshape(-1)

    
    def test_accepts_valid_system(self):
        x_seq = jnp.ones((4,2))
        u_seq = jnp.ones([3,2])
        K_seq = jnp.ones([3,2,2])
        d_seq = jnp.ones([3,2,1])
        lsf = 0.5
        carry_in = (0, jnp.zeros((1), dtype=float), x_seq)
        seqs_in = (x_seq, u_seq, K_seq, d_seq)
        fp_scan_func_curried = lambda lsf, carry_in, seqs_in : ilqr.scan_func_forward_pass(self.example_dyn_func,self.example_cost_func, lsf,carry_in, seqs_in)
        x_seq_new, u_seq_new, cost_float = ilqr.simulate_forward_pass(fp_scan_func_curried, self.example_cost_func, x_seq, u_seq, K_seq, d_seq, lsf)
        self.assertEqual(True,True)


class scan_func_forward_pass_tests(unittest.TestCase):
    def example_cost_func(self,x_k:jnp.ndarray, u_k:jnp.ndarray, k)->float:
        x = x_k.reshape(-1,1)
        u = u_k.reshape(-1,1)
        cost_float = (x.T@x + u.T@u )
        return cost_float
    
    def example_dyn_func(self, x:jnp.ndarray, u:jnp.ndarray)->jnp.ndarray:
        A = jnp.eye(len(x))
        B = jnp.ones((len(x),len(u)))    
        x_new = A@x.reshape(-1,1) + B@u.reshape(-1,1)
        return x_new.reshape(-1)

    def test_accepts_valid_system(self):
        x_k = jnp.array([1.0,1.0])
        u_k = jnp.array([1.0,1.0])
        K_k = jnp.array([[1.0,1.0],[1.0,1.0]])
        d_k = jnp.array([[1.0],[1.0]])
        k = 0
        lsf = 0.5
        carry_in = (0, 0.0, x_k)
        seqs_in = (x_k, u_k, K_k, d_k)
        carry_out, seqs_out = ilqr.scan_func_forward_pass(self.example_dyn_func,self.example_cost_func, lsf,carry_in, seqs_in)
        int_expected = 1
        u_k_new_expected = (u_k.reshape(-1,1) + K_k @ (x_k.reshape(-1,1) - x_k.reshape(-1,1)) + lsf * d_k).reshape(-1)
        cost_float_expected = self.example_cost_func(x_k, u_k_new_expected, k)
        x_kp1_new_expected = self.example_dyn_func(x_k, u_k_new_expected)
        self.assertEqual(carry_out[0], int_expected)
        self.assertEqual(carry_out[1], cost_float_expected)
        self.assertEqual((carry_out[2]).all(), (seqs_out[0]).all())
        self.assertEqual((seqs_out[0]).all(), (x_kp1_new_expected).all())
        self.assertEqual((seqs_out[1]).all(), (u_k_new_expected).all())

class initialize_backwards_pass_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        len_seq = data_and_funcs.ilqr_config.len_seq
        x_len   = data_and_funcs.ilqr_config.num_states
        u_len   = data_and_funcs.ilqr_config.num_controls
        p_N     = jnp.ones([x_len, 1])
        P_N     = jnp.ones([x_len, x_len])
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
        u_len             = shared_data_funcs.ilqr_config.num_controls
        x_len             = shared_data_funcs.ilqr_config.num_states
        len_seq           = shared_data_funcs.ilqr_config.len_seq
        state_seq_updated, control_seq_updated, cost_float_updated, cost_seq_updated, x_cost_seq_updated, u_cost_seq_updated, in_bounds_bool = ilqr.initialize_forwards_pass(x_len, u_len, seed_state_vec, len_seq)
        self.assertEqual(len(state_seq_updated), len_seq)
        self.assertEqual(len(control_seq_updated), len_seq-1)
        self.assertEqual(cost_float_updated, 0)
        self.assertEqual(in_bounds_bool, False)

# class taylor_expand_pseudo_hamiltonian_tests(unittest.TestCase):
#     def test_accepts_valid_inputs(self):
#         shared_data_funcs = shared_unit_test_data_and_funcs()        
#         x_k = jnp.array([[1.],[1.]])
#         u_k = jnp.array([[1.]])
#         x_len = len(x_k)
#         u_len = len(u_k)
#         A_lin_k = jnp.array([[0,1],[1,1]])     
#         B_lin_k = jnp.array([[0],[1]])
#         p_kp1   = jnp.ones([x_len, 1])
#         P_kp1   = jnp.ones([x_len, x_len])
#         k_step  = 1
#         ro_reg  = 1
#         q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = ilqr.taylor_expand_pseudo_hamiltonian(
#                                                                             shared_data_funcs.ilqr_config.cost_func_for_diff, 
#                                                                             A_lin_k, 
#                                                                             B_lin_k, 
#                                                                             x_k, 
#                                                                             u_k, 
#                                                                             P_kp1, p_kp1,
#                                                                             ro_reg, 
#                                                                             k_step)
#         self.assertEqual(np.shape(q_x),  (x_len, 1))
#         self.assertEqual(np.shape(q_u),  (u_len, 1))
#         self.assertEqual(np.shape(q_xx), (x_len,x_len))
#         self.assertEqual(np.shape(q_uu), (u_len,u_len))
#         self.assertEqual(np.shape(q_ux), (u_len,x_len)) 
#         self.assertEqual(np.shape(q_uu_reg), (u_len,u_len))
#         self.assertEqual(np.shape(q_ux_reg), (u_len,x_len)) 

#     def test_accepts_valid_inputs_from_shared(self):
#         shared_data_funcs = shared_unit_test_data_and_funcs()        
#         x_seq = shared_data_funcs.x_seq_example
#         u_seq = shared_data_funcs.u_seq_example
#         x_len = shared_data_funcs.ilqr_config.num_states
#         u_len = shared_data_funcs.ilqr_config.num_controls
#         A_lin_k = np.array([[1,1],[0,1]])     
#         B_lin_k = np.array([[0],[1]])
#         p_kp1   = np.ones([x_len, 1])
#         P_kp1   = np.ones([x_len, x_len])
#         k_step  = 1
#         ro_reg  = 1.0
#         q_x, q_u, q_xx, q_ux, q_uu, q_ux_reg, q_uu_reg = ilqr.taylor_expand_pseudo_hamiltonian(
#                                                                             shared_data_funcs.ilqr_config.cost_func_for_diff, 
#                                                                             A_lin_k, 
#                                                                             B_lin_k, 
#                                                                             x_seq[k_step], 
#                                                                             u_seq[k_step], 
#                                                                             P_kp1, p_kp1,
#                                                                             ro_reg, 
#                                                                             k_step)
#         self.assertEqual(np.shape(q_x),  (x_len, 1))
#         self.assertEqual(np.shape(q_u),  (u_len, 1))
#         self.assertEqual(np.shape(q_xx), (x_len,x_len))
#         self.assertEqual(np.shape(q_uu), (u_len,u_len))
#         self.assertEqual(np.shape(q_ux), (u_len,x_len)) 
#         self.assertEqual(np.shape(q_uu_reg), (u_len,u_len))
#         self.assertEqual(np.shape(q_ux_reg), (u_len,x_len)) 

class calculate_final_ctg_approx_tests(unittest.TestCase):
    def test_calculate_final_cost_to_go_approximation_accepts_valid_system(self):
        shared_data_funcs = shared_unit_test_data_and_funcs()
        x_N = shared_data_funcs.x_seq_example[-1]
        u_N = shared_data_funcs.u_seq_example[-1]
        P_N, p_N = ilqr.calculate_final_ctg_approx(shared_data_funcs.ilqr_config.cost_func_for_diff,x_N,len(u_N),len_seq = 1)
        p_N_expected = shared_data_funcs.cost_func_quad_unit_params.Qf @ x_N.reshape(-1,1)
        P_N_expected = shared_data_funcs.cost_func_quad_unit_params.Qf
        self.assertEqual(p_N.tolist(), p_N_expected.tolist())
        self.assertEqual(P_N.tolist(), P_N_expected.tolist())

class calculate_backstep_ctg_approx_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        x_len = data_and_funcs.ilqr_config.num_states
        u_len = data_and_funcs.ilqr_config.num_controls
        q_x   = jnp.ones([x_len,1])
        q_u   = jnp.ones([u_len,1])
        q_xx  = jnp.ones([x_len, x_len])
        q_uu  = jnp.ones([u_len, u_len])
        q_ux  = jnp.ones([u_len, x_len])
        K_k   = jnp.ones([u_len, x_len])
        d_k   = jnp.ones([u_len, 1])
        P_k, p_k, Del_V_vec_k = ilqr.calculate_backstep_ctg_approx(q_x, q_u, 
                                                                   q_xx, q_uu, q_ux, 
                                                                   K_k, d_k)
        P_k_expect = jnp.ones([x_len,x_len]) * 4.
        p_k_expect = jnp.ones([x_len, 1]) * 4.
        Del_V_vec_k_expect = jnp.array([1.0, 0.5])
        self.assertEqual(jnp.shape(P_k), (x_len, x_len))
        self.assertEqual(jnp.shape(p_k), (x_len, 1))  
        self.assertEqual(jnp.shape(Del_V_vec_k), (2, ))      
        self.assertEqual(P_k.tolist(), P_k_expect.tolist())
        self.assertEqual(p_k.tolist(), p_k_expect.tolist()) 
        self.assertEqual(Del_V_vec_k.tolist(), Del_V_vec_k_expect.tolist())

class calculate_optimal_gains_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        q_uu = jnp.array([[1.,1.],[0,1.]])
        q_ux = jnp.array([[1.,1.],[0,1.]])
        q_u  = jnp.array([[1.],[1.]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
        d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
        K_k_expected = -(jnp.array([[1.,-1.],[0,1.]])) @ (jnp.array([[1.,1.],[0,1.]]))
        d_k_expected = -(jnp.array([[1.,-1.],[0,1.]])) @ (jnp.array([[1.],[1.]]))
        self.assertEqual(K_k.tolist(), K_k_expected.tolist())
        self.assertEqual(d_k.tolist(), d_k_expected.tolist())

    def test_accepts_valid_inputs_from_shared(self):
        q_uu = jnp.array([[1.,1.],[0,1.]])
        q_ux = jnp.array([[0.1,0.2,0.3,0.4],[1., 2., 3., 4.]])
        q_u  = jnp.array([[1],[2]])
        K_k, d_k = ilqr.calculate_optimal_gains(q_uu, q_ux, q_u)
        K_k_expected = -jnp.linalg.inv(q_uu) @ q_ux
        d_k_expected = -jnp.linalg.inv(q_uu) @ q_u
        self.assertEqual(K_k.tolist(), K_k_expected.tolist())
        self.assertEqual(d_k.tolist(), d_k_expected.tolist())    

class calculate_u_k_new_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        u_nom_k = jnp.array([1.0, 2.0])
        x_nom_k = jnp.array([1.0, 1.0])
        x_new_k = x_nom_k + 1.0
        K_k     = jnp.ones([2, 2])
        d_k     = jnp.ones([2, 1]) * 2.0
        line_search_factor = 1
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        # u_k_expected = u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k
        u_k_expected = np.array([1. + (1.* (2.-1.) + 1. * (2.-1.)) + 1.*2.,
                                 2. + (1.* (2.-1.) + 1. * (2.-1.)) + 1.*2.])
        self.assertEqual(u_k_updated_new.tolist(), u_k_expected.tolist())

    def test_reject_invalid_K_k_dimension(self):
        u_nom_k = jnp.array([1.0, 2.0])
        x_nom_k = jnp.array([1.0, 1.0])
        x_new_k = x_nom_k + 1.0
        K_k     = jnp.ones([3, 2])
        d_k     = jnp.ones([2, 1]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'K_k incorrect shape, must be (len(u_nom_k), len(x_nom_k))')

    def test_reject_invalid_d_k_dimension(self):
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
        u_nom_k = np.array([1.0, 2.0])
        x_nom_k = np.array([[1.0], [1.0]])
        x_new_k = np.array([[1.0], [2.0]])
        K_k     = np.ones([2, 2])
        d_k     = np.ones([2, 1]) * 2.0
        line_search_factor = 1
        with self.assertRaises(AssertionError) as assert_seq_len_error:
           u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'x_nom_k must be column vector (n,)')


    def test_accepts_valid_inputs_with_shared_data(self):
        data_and_funcs = shared_unit_test_data_and_funcs()
        k_step = 1
        u_nom_k = data_and_funcs.u_seq_example[k_step]
        x_nom_k = data_and_funcs.x_seq_example[k_step]
        x_new_k = x_nom_k + 0.1
        K_k     = data_and_funcs.K_seq_example[k_step]
        d_k     = data_and_funcs.d_seq_example[k_step]
        line_search_factor = 0.5
        u_k_updated_new = ilqr.calculate_u_k_new(u_nom_k, x_nom_k, x_new_k,
                                                 K_k, d_k, line_search_factor)
        u_k_expected = (u_nom_k + K_k @ (x_new_k - x_nom_k) + line_search_factor * d_k).reshape(-1)
        self.assertEqual(u_k_updated_new, u_k_expected)    

class calculate_cost_decrease_ratio_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        prev_cost_float = 100.0
        new_cost_float  = 70.0
        Del_V_vec_seq = jnp.ones([3,2])
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
        Del_V_vec_seq = jnp.ones([3,3])
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
        Del_V_vec_seq = jnp.ones([3,2])
        Del_V_vec_seq[:,1] *= 2.0
        line_search_factor = 2.0
        Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
        expected_output = ((1+1+1) * line_search_factor) + ((2+2+2) * line_search_factor**2)
        self.assertEqual(Del_V_sum, expected_output)

    def test_reject_invalid_array_shape(self):
        Del_V_vec_seq = jnp.ones([3,3]) 
        line_search_factor = 2
        with self.assertRaises(AssertionError) as assert_seq_len_error:
            Del_V_sum = ilqr.calculate_expected_cost_decrease(Del_V_vec_seq, line_search_factor)
        self.assertEqual(str(assert_seq_len_error.exception), 'Del_V_vec_seq is not the right shape, must be (:,2)')

class analyze_cost_decrease_tests(unittest.TestCase):
    def test_returns_true_for_in_bound_inputs(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config.cost_ratio_bounds
        cost_decrease_ratio = bounds[0] + (bounds[0] + bounds[1]) * 0.5
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, True)
 
    def test_returns_false_for_input_too_high(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config.cost_ratio_bounds
        cost_decrease_ratio = bounds[1] + 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)

    def test_returns_false_for_input_too_low(self):
        data_and_funcs      = shared_unit_test_data_and_funcs()
        bounds              = data_and_funcs.ilqr_config.cost_ratio_bounds
        cost_decrease_ratio = bounds[0] - 1
        in_bounds_bool      = ilqr.analyze_cost_decrease(cost_decrease_ratio, bounds)
        self.assertEqual(in_bounds_bool, False)

class calculate_avg_ff_gains_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        #create test conditions
        int_iter = 5
        d_seq = jnp.ones((4,2,1))
        u_seq = jnp.ones((4,2,1))
        #test function
        ff_avg_gains = ilqr.calculate_avg_ff_gains(d_seq, u_seq)
        # create expected output
        ff_avg_gains_expect = 1 / (5-1) * (1 / (np.sqrt(2) + 1)) * 4
        #compare outputs
        self.assertEqual(ff_avg_gains, ff_avg_gains_expect)

if __name__ == '__main__':
    unittest.main()
