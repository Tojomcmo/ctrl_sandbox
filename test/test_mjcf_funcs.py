import pytest
from jax import numpy as jnp

import ctrl_sandbox.mujoco_funcs as mj_funcs
import ctrl_sandbox.mjcf_models as mjcf_models


# def test_create_acrobot_model_loads():
#     model_obj = mjcf_models.mjcf_dpend(shoulder_act=False, elbow_act=True)
#     model = model_obj.get_mjcf_model()
#     time_step = 0.01
#     x_vec = jnp.array([0.1, 0.1, 0.1, 0.1], dtype=float)
#     u_vec = jnp.array([0.1], dtype=float)
#     mjModel, mjRender, mjData = mj_funcs.create_mujoco_model(model, time_step)
#     mj_funcs.set_mj_state_vec(mjData, x_vec)
#     mj_funcs.set_mj_ctrl_vec(mjData, u_vec)
#     A, B = mj_funcs.linearize_mujoco_state_and_control(mjModel, mjData)
#     assert A.shape == (len(x_vec), len(x_vec))
