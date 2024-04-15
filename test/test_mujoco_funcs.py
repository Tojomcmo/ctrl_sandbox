import matplotlib.pyplot as plt
import unittest
from jax import numpy as jnp
import numpy as np

import src.mujoco_funcs as mj_funcs

class shared_mjcf_models:
    def __init__(self) -> None:
        pass

    def create_MJCF_single_pend(self):
        mjcf_pend_model = """
                      <mujoco>
                      <option gravity="0 0 -9.81">
                          <flag sensornoise="enable"/>
                      </option>

                      <worldbody>
                          <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                          <camera name="fixed" pos="0 -5 1" xyaxes="1 0 0 0 0 1"/>
                          <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                          <body name="pend1" pos="0 0 2" euler="0 0 0">
                              <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
                              <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
                          </body>
                      </worldbody>

                      <actuator>
                          <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
                      </actuator>

                      </mujoco>
                                    """
        return mjcf_pend_model

    def create_MJCF_single_pend_m_d_mod(self, mass, damping, length):
        mjcf_pend_model = """
                      <mujoco>
                      <option gravity="0 0 -9.81">
                          <flag sensornoise="enable"/>
                      </option>

                      <worldbody>
                          <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                          <camera name="fixed" pos="0 -5 1" xyaxes="1 0 0 0 0 1"/>
                          <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                          <body name="pend1" pos="0 0 2" euler="0 0 0">
                              <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
                              <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
                          </body>
                      </worldbody>

                      <actuator>
                          <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
                      </actuator>

                      </mujoco>
                                    """
        return mjcf_pend_model

class fwd_sim_mj_w_ctrl_tests(unittest.TestCase):
    def test_function_recieves_valid_system(self):
        models = shared_mjcf_models()
        model, _, data = mj_funcs.create_mujoco_model(models.create_MJCF_single_pend())
        model.opt.timestep = 0.1
        x_init = np.array([[0.0],[0.0]])
        u_seq  = np.ones((3,1,1))
        x_seq = mj_funcs.fwd_sim_mj_w_ctrl(model, data, x_init, u_seq)
        self.assertEqual(True, True)        


if __name__ == '__main__':
    unittest.main()
