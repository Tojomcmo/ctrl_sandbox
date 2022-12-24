import unittest
import cost_functions as cfuncs
from jax import numpy as jnp

class cost_func_quad_state_and_control_tests(unittest.TestCase):
    def cost_func_quad_state_and_control_accepts_valid_system(self):
        Q              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]])
        R              = jnp.array([[1, 0], [0, 10]])
        S              = jnp.array([[1, 0, 0], [0, 10, 0], [0, 0, 100]])
        state          = jnp.array([[0.1, 1, 10]])
        control        = jnp.array([[1, 0.1]])
        j_expect       = (0.5) * ((state   * Q * jnp.transpose(state)) 
                                +(control * R * jnp.transpose(control)))
        j_final_expect = (0.5) * (state   * S * jnp.transpose(state))  

        j              = cfuncs.cost_func_quad_state_and_control(Q,R,S,state,control)
        j_final        = cfuncs.cost_func_quad_state_and_control(Q,R,S,state)
        
        self.assertEqual(j, j_expect)
        self.assertEqual(j_final, j_final_expect)



if __name__ == '__main__':
    unittest.main()