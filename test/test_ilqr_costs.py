import unittest
from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
import src.cost_functions as cost


class cost_func_quad_state_and_control_tests(unittest.TestCase):
    def test_accepts_valid_system(self):
        Q              = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
        R              = np.array([[1, 0], [0, 10]], dtype=float)
        Qf             = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0       

        state          = np.array([[0.1, 1, 10]], dtype=float)
        control        = np.array([[1, 0.1]], dtype=float)
        state_des      = np.array([[0.1, 1, 10]], dtype=float) * 2.0
        control_des    = np.array([[1, 0.1]], dtype=float) *2.0       

        cost_func_params = cost.costFuncQuadStateAndControlParams(Q,R,Qf,state_des,control_des)        
        k_step = 0
        j,j_state,j_control = cost.cost_func_quad_state_and_control(cost_func_params,
                                                                 state, control,
                                                                 k_step, is_final_bool=False)
        state_corr   = state - state_des
        control_corr = control - control_des
        j_expect       = ((0.5) * ((state_corr@ Q @ jnp.transpose(state_corr))+(control_corr@ R @ jnp.transpose(control_corr)))).item()                                     
        self.assertEqual(j.item(), j_expect)

    def test_accepts_valid_final_system(self):
        Q              = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
        R              = np.array([[1, 0], [0, 10]], dtype=float)
        Qf             = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0       

        state          = np.array([[0.1, 1, 10]], dtype=float)
        control        = np.array([[1, 0.1]], dtype=float)
        state_des      = np.array([[0.1, 1, 10]], dtype=float) * 2.0
        control_des    = np.array([[1, 0.1]], dtype=float) *2.0       

        cost_func_params = cost.costFuncQuadStateAndControlParams(Q,R,Qf,state_des,control_des)        
        k_step = 1
        j,j_state,j_control = cost.cost_func_quad_state_and_control(cost_func_params,
                                                                 state, control,
                                                                 k_step, is_final_bool=False)
        state_corr   = state - state_des
        control_corr = control - control_des
        j_expect       = (0.5) * (state_corr @ Qf @ jnp.transpose(state_corr))
                                  
        self.assertEqual(j, j_expect)

if __name__ == '__main__':
    unittest.main()