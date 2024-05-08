import unittest
from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
import cost_functions as cost


class cost_func_quad_state_and_control_tests(unittest.TestCase):
    def test_accepts_valid_system(self):
        Q              = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
        R              = np.array([[1, 0], [0, 10]], dtype=float)
        Qf             = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0       

        state          = np.array([0.1, 1, 10], dtype=float)
        control        = np.array([1, 0.1], dtype=float)
        state_des      = np.ones((3,3), dtype=float) * 2.0
        control_des    = np.ones((3,2), dtype=float) *2.0       
        k_step         = 1

        cost_func_obj = cost.cost_quad_x_and_u(Q,R,Qf,state_des,control_des)
        j,j_state,j_control = cost_func_obj.cost_func_quad_state_and_control(state,control,k_step, is_final_bool=False)

        state_corr   = (state - state_des[k_step]).reshape(-1,1)
        control_corr = (control - control_des[k_step]).reshape(-1,1)
        j_expect       = jnp.array((0.5) * ((state_corr.T @ Q @ state_corr)+(control_corr.T @ R @ control_corr)))                                    
        self.assertEqual(j, j_expect)

    def test_accepts_valid_final_system(self):
        Q              = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
        R              = np.array([[1, 0], [0, 10]], dtype=float)
        Qf             = np.array([[100, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float) * 2.0       

        state          = np.array([0.1, 1, 10], dtype=float)
        control        = np.array([1, 0.1], dtype=float)
        state_des      = np.ones((3,3), dtype=float) * 2.0
        control_des    = np.ones((3,2), dtype=float) *2.0       
        k_step         = 1

        cost_func_obj = cost.cost_quad_x_and_u(Q,R,Qf,state_des,control_des)
        j,j_state,j_control = cost_func_obj.cost_func_quad_state_and_control(state,control,k_step, is_final_bool=True)

        state_corr   = (state - state_des[k_step]).reshape(-1,1)
        control_corr = (control - control_des[k_step]).reshape(-1,1)
        j_expect       = jnp.array((0.5) * (state_corr.T @ Qf @ state_corr))
                                  
        self.assertEqual(j, j_expect)

if __name__ == '__main__':
    unittest.main()