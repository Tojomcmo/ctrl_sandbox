import matplotlib.pyplot as plt
import unittest
from jax import numpy as jnp
import numpy as np

import src.mujoco_funcs as mj_funcs
import src.mjcf_models as mjcf_models
import src.visualize_mj_funcs as mj_vis

class shared_analysis_funcs():
    def __init__(self) -> None:
        pass



class test_create_acrobot_model(unittest.TestCase):
    def test_creat_acrobot_model_loads(self):
        model = mjcf_models.create_acrobot()
        time_step = 0.01
        x_vec = np.array([[0.1],[0.1],[0.1],[0.1]], dtype=float)
        u_vec = np.array([[0.1]], dtype=float)
        mjModel, mjRender, mjData = mj_funcs.create_mujoco_model(model, time_step)
        mj_funcs.set_mj_state_vec(mjData, x_vec)
        mj_funcs.set_mj_ctrl_vec(mjData, u_vec)
        A, B = mj_funcs.linearize_mujoco_state_and_control(mjModel, mjData)
        self.assertEqual(A.shape, (len(x_vec),len(x_vec)))



if __name__ == '__main__':
    unittest.main()
