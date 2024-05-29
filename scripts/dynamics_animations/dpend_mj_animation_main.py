import numpy as np
import mujoco as mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animate

import mujoco_funcs as mj_funcs
import mjcf_models as mj_models
import visualize_mj_funcs as mj_vis
import visualize_dyn_funcs as vis_dyn

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # mjcf_model  = mj_models.create_MJCF_double_pend_m_d_mod(1,1,1)
    mjcf_model = mj_models.create_MJCF_double_pend_fa_dev()
    model = mujoco.MjModel.from_xml_string(mjcf_model)
    data = mujoco.MjData(model)

    # renderer and visualization
    show_plot = False
    show_custom_ani = True

    # finite difference prep
    nu = model.nu
    nx = model.nv
    eps = 1e-6
    A = np.zeros((2 * nx, 2 * nx))
    B = np.zeros((2 * nx, nu))
    flg_centered = True

    # update model and data parameters
    time_step = 0.005
    model.opt.timestep = time_step
    steps = 1000
    data.ctrl = np.array([6.0, 4.0])
    x_init = np.array([0.0, 0.0, 0.0, 0.0])
    mj_funcs.set_mj_state_vec(data, x_init)
    mujoco.mj_forward(model, data)

    # initial linearization and predictiuon
    state_vec = np.hstack((data.qpos.copy(), data.qvel.copy())).reshape(-1, 1)
    control_vec = np.array(data.ctrl).reshape(-1, 1)
    A, B = mj_funcs.linearize_mujoco_state_and_control(model, data, eps, flg_centered)
    state_next_pred = (A @ state_vec) + (B @ control_vec)

    x_vec_seq = np.zeros((steps, 4))
    x_vec_seq[0] = state_vec.reshape(-1)
    if show_plot is True:
        renderer = mujoco.Renderer(model, 480, 480)
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        renderer.update_scene(data, "fixed", scene_option=scene_option)
        frames = []
        img_set = []
        fig1, ax1 = plt.subplots()

    for idx in range(steps):
        mujoco.mj_step(model, data)
        state_vec = np.hstack((data.qpos.copy(), data.qvel.copy())).reshape(-1, 1)
        x_vec_seq[idx] = state_vec.reshape(-1)
        A, B = mj_funcs.linearize_mujoco_state_and_control(
            model, data, eps, flg_centered=False
        )
        control_vec = np.array(data.ctrl).reshape(-1, 1)

        # update frame
        if show_plot is True:
            mj_vis.update_plt_frame(renderer, data, frames, img_set)
        idx += 1

    if show_plot is True:
        ani = animate.ArtistAnimation(fig1, img_set, interval=time_step * 1000)
        plt.show()

    if show_custom_ani is True:
        fig = plt.figure(figsize=[10, 8])
        pend_animation = vis_dyn.double_pm_pend_animation(
            1.0, 1.0, x_vec_seq, time_step, fig, th2="rel"
        )
        pend_animation.create_double_pend_animation()
        pend_animation.show_plot()

###################################

# for idx in range(steps):
#     print('state_predict: ', state_next_pred)
#     mujoco.mj_step(model, data)
#     state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)
#     print('state_actual: ', state_vec)
#     A, B = mj_funcs.linearize_mujoco_state_and_control(model, data, eps, flg_centered)
#     # mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
#     print('A: ', A)
#     print('B: ', B)
#     state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)
#     control_vec = np.array(data.ctrl).reshape(-1,1)
#     state_next_pred = A @ state_vec + B @   control_vec
