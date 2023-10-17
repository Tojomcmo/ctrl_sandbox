import unittest
from jax import numpy as jnp

import src.cost_functions as cfuncs


class cost_func_quad_state_and_control_tests(unittest.TestCase):
    def test_accepts_valid_system(self):
        Q              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]])
        R              = jnp.array([[1, 0], [0, 10]])
        Qf              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]]) * 2       
        cost_func_params = {
                            'Q'  : Q,
                            'R'  : R,
                            'Qf' : Qf
                            }
        state          = jnp.array([[0.1, 1, 10]])
        control        = jnp.array([[1, 0.1]])
        state_des      = jnp.array([[0.1, 1, 10]]) * 2
        control_des    = jnp.array([[1, 0.1]]) *2       
        k_step = 1
        j,j_state,j_control = cfuncs.cost_func_quad_state_and_control(cost_func_params,
                                                                 state, control,
                                                                 k_step,
                                                                 state_des, control_des)
        state_corr   = state - state_des
        control_corr = control - control_des
        j_expect       = (0.5) * ( (state_corr   @ Q @ jnp.transpose(state_corr))
                                  +(control_corr @ R @ jnp.transpose(control_corr)))                                     
        self.assertEqual(j, j_expect)

    def test_accepts_valid_final_system(self):
        Q              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]])
        R              = jnp.array([[1, 0], [0, 10]])
        Qf              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]]) * 2       
        cost_func_params = {
                            'Q'  : Q,
                            'R'  : R,
                            'Qf' : Qf
                            }
        state          = jnp.array([[0.1, 1, 10]])
        control        = jnp.array([[0, 0]])
        state_des      = jnp.array([[0.1, 1, 10]]) * 2
        control_des    = jnp.array([[0, 0]])      
        k_step = 1
        j,j_state,j_control = cfuncs.cost_func_quad_state_and_control(cost_func_params,
                                                                 state, control,
                                                                 k_step,
                                                                 state_des, control_des,is_final_bool=True)
        state_corr   = state - state_des
        control_corr = control - control_des
        j_expect       = (0.5) * (state_corr @ Qf @ jnp.transpose(state_corr))
                                  
        self.assertEqual(j, j_expect)

if __name__ == '__main__':
    unittest.main()