import unittest
import numpy as np
import jax
import jax.numpy as jnp

import dyn_functions as dyn
import gen_ctrl_funcs as gen_ctrl

class nl_pend_dyn_tests(unittest.TestCase):
    def test_nl_pend_dyn_accepts_valid_inputs(self):
        pend_sys = dyn.single_pm_pend_dyn(g=9.81,b=0.1,l=1.0)
        x_vec  = np.array([1.0,1.0])
        u_vec  = np.array([1.0])
        x_dot  = pend_sys.cont_dyn_func(x_vec, u_vec)                 
        # analyze.plot_2d_state_scatter_sequences(x_seq)
        self.assertEqual(True, True)        
 

class double_pm_pend_dyn_tests(unittest.TestCase):
    def test_double_pend_accepts_valid_inputs(self):    
        dpend_sys = dyn.double_pm_pend_dyn(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1, 
                                        shoulder_act=True, elbow_act=True)
        x_vec = np.array([0.1,0.1,0.1,0.1])
        u_vec = np.array([5.0,1.0]) 
        x_dot = dpend_sys.cont_dyn_func(x_vec,u_vec)
        self.assertEqual(True, True)

    def test_double_pend_is_jax_compatible(self):
        dpend_sys = dyn.double_pm_pend_dyn(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1, 
                                        shoulder_act=True, elbow_act=True)
        x_vec = np.array([0.1,0.1,0.1,0.1])
        u_vec = np.array([1.0,1.0])        
        A, B = gen_ctrl.linearize_dynamics(dpend_sys.cont_dyn_func, x_vec, u_vec)
        self.assertEqual(True, True)

    def test_double_pend_is_jax_compatible_thru_rk(self):
        dpend_sys = dyn.double_pm_pend_dyn(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1, 
                                        shoulder_act=True, elbow_act=True)
        x_vec = np.array([0.1,0.1,0.1,0.1])
        u_vec = np.array([1.0,1.0]) 
        time_step = 0.01
        dyn_func_discrete = lambda x, u: gen_ctrl.step_rk4(dpend_sys.cont_dyn_func, time_step, x, u)        
        A, B = gen_ctrl.linearize_dynamics(dyn_func_discrete, x_vec, u_vec)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
