import unittest
import cost_functions as cfuncs
from jax import numpy as jnp



class cost_func_quad_state_and_control_tests(unittest.TestCase):
    def test_cost_func_quad_state_and_control_accepts_valid_system(self):
        Q              = jnp.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]])
        R              = jnp.array([[1, 0], [0, 10]])
        state          = jnp.array([[0.1, 1, 10]])
        control        = jnp.array([[1, 0.1]])
        j_expect       = (0.5) * ((state   @ Q @ jnp.transpose(state))
                                 +(control @ R @ jnp.transpose(control)))
        j              = cfuncs.cost_func_quad_state_and_control(Q,R,state,control)
        self.assertEqual(j,       j_expect)



if __name__ == '__main__':
    unittest.main()