import pytest
from jax import numpy as jnp
import numpy as np

import ctrl_sandbox.mujoco_funcs as mj_funcs
import ctrl_sandbox.mjcf_models as mjcf


# def test_create_mujoco_model_recieves_valid_system():
#     model_obj = mjcf.mjcf_dpend(shoulder_act=True, elbow_act=True)
#     model, renderer, data = mj_funcs.create_mujoco_model(
#         model_obj.get_mjcf_model(), time_step=0.1
#     )
#     model.opt.timestep = 0.1
#     x_init = np.array([0.0, 0.0, 0.0, 0.0])
#     u_seq = np.ones((3, 2))
#     x_seq = mj_funcs.fwd_sim_mj_w_ctrl(model, data, x_init, u_seq)
#     assert True
