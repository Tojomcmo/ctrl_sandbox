import unittest
import numpy as np
import jax
import jax.numpy as jnp

import src.dyn_functions as dyn
import src.gen_ctrl_funcs as gen_ctrl

class nl_pend_dyn_tests(unittest.TestCase):
    def test_nl_pend_dyn_accepts_valid_inputs(self):
        params = dyn.nlPendParams(g=9.81,b=0.1,l=1.0)
        x_vec  = np.array([1.0,1.0])
        u_vec  = np.array([1.0])
        x_dot  = dyn.pend_dyn_nl(params, x_vec, u_vec)                 
        # analyze.plot_2d_state_scatter_sequences(x_seq)
        self.assertEqual(True, True)        
 

class double_pend_dyn_tests(unittest.TestCase):
    def test_double_pend_accepts_valid_inputs(self):    
        params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1)
        x_vec = np.array([0.1,0.1,0.1,0.1])
        u_vec = np.array([5.0,1.0]) 
        x_dot = dyn.double_pend_no_damp_full_act_dyn(params,x_vec,u_vec)
        self.assertEqual(True, True)

    def test_double_pend_is_jax_compatible(self):
        params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1)
        x_vec = np.array([0.1,0.1,0.1,0.1])
        u_vec = np.array([5.0,1.0]) 
        dyn_func = lambda x, u : dyn.double_pend_no_damp_full_act_dyn(params,x, u)           
        A, B = gen_ctrl.linearize_dynamics(dyn_func, x_vec, u_vec)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
