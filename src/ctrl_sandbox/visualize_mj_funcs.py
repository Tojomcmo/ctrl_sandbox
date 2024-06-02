import mujoco as mujoco
import numpy as np
import matplotlib.pyplot as plt

import ctrl_sandbox.mujoco_funcs as mj_funcs
import ctrl_sandbox.ilqr_funcs as ilqr
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl


def update_plt_frame(
    renderer: mujoco.Renderer, data: mujoco._structs.MjData, frames: list, img_set: list
):
    renderer.update_scene(data, "fixed")
    pixels = renderer.render()
    frames.append(pixels)
    img_set.append([plt.imshow(pixels)])


def create_mj_sim_video(model, data, framerate, sim_dyn_step):
    renderer = mujoco.Renderer(model, 480, 480)
    renderer.update_scene(data, "fixed")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    frames = []
    img_set = []
    fig1, ax1 = plt.subplots()


def create_mj_video_ilqr_w_ctrl(
    model,
    data,
    renderer,
    scene_option,
    framerate: int,
    x_init,
    ctrl_out: ilqr.ilqrControllerState,
    ilqr_config: ilqr.ilqrConfigStruct,
):
    """
    function simulates provided model and data forward in time using u sequence
    and returns state trajectory x_seq and linearized state and control matricies A
    and B at each pair x_seq, u_seq
    """
    ts_sim = ilqr_config.time_step
    frames: list = []
    img_set: list = []
    nx = model.nv
    nu = model.nu
    x_sim_seq = np.zeros((ctrl_out.len_seq, nx * 2))
    u_sim_seq = np.zeros((ctrl_out.len_seq - 1, nu))
    mujoco.mj_resetData(model, data)
    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    mj_funcs.set_mj_state_vec(data, x_init)
    mujoco.mj_forward(model, data)
    x_sim_seq[0] = mj_funcs.get_state_vec(data)
    for idx in range(len(ctrl_out.u_seq - 1)):
        u_sim_seq[idx] = gen_ctrl.calculate_ff_fb_u(
            ctrl_out.K_seq[idx],
            ctrl_out.u_seq[idx],
            ctrl_out.x_seq[idx],
            x_sim_seq[idx],
        )
        data.ctrl = (u_sim_seq[idx]).reshape(-1)
        mujoco.mj_step(model, data)
        x_sim_seq[idx + 1] = mj_funcs.get_state_vec(data)
        if len(frames) < data.time * framerate:
            update_plt_frame(renderer, data, frames, img_set)
        idx += 1
    model.opt.timestep = prev_timestep
    return img_set, frames


def create_mj_video_w_ctrl_with_different_time_steps(
    model,
    data,
    renderer,
    scene_option,
    framerate,
    x_init,
    ctrl_out: ilqr.ilqrControllerState,
    ilqr_config: ilqr.ilqrConfigStruct,
):
    """
    function simulates provided model and data forward in time using u sequence
    and returns state trajectory x_seq and linearized state and control matricies A
    and B at each pair x_seq, u_seq
    """
    ts_sim = ilqr_config.time_step
    ts_ctrl = ilqr_config.time_step
    if int(ts_ctrl % ts_sim) != 0:
        raise ValueError(
            "the control timestep must be an integer multiple of the simulation timestep"
        )
    elif ts_ctrl < ts_sim:
        raise ValueError(
            "the control timestep must be larger than or equal to the simulation timestep"
        )
    frames = []
    img_set = []
    nx = model.nv
    nu = model.nu
    t_final = (ctrl_out.len_seq - 1) * ts_ctrl
    sim_steps = int(t_final / ts_sim)
    steps_zoh = int(ts_ctrl / ts_sim)
    x_sim_seq = np.zeros((ctrl_out.len_seq, nx * 2, 1))
    u_sim_seq = np.zeros((ctrl_out.len_seq - 1, nu, 1))
    mujoco.mj_resetData(model, data)
    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    mj_funcs.set_mj_state_vec(data, x_init)
    mujoco.mj_forward(model, data)
    x_sim_seq[0] = mj_funcs.get_state_vec(data)
    assert (
        x_sim_seq[0]
    ).tolist() == x_init.tolist(), "x_init vector improperly initialized or queried"
    ctrl_idx = 0
    u_sim_seq[0] = gen_ctrl.calculate_ff_fb_u(
        ctrl_out.K_seq[0],
        ctrl_out.u_seq[0],
        ctrl_out.x_seq[0],
        x_sim_seq[0],
    )
    data.ctrl = (u_sim_seq[0]).reshape(-1)
    ctrl_count = 0
    for sim_idx in range(sim_steps + 1):
        if ctrl_count == steps_zoh:
            ctrl_count = 0
            ctrl_idx += 1
            x_sim_seq[ctrl_idx] = mj_funcs.get_state_vec(data)
            if sim_idx < sim_steps:
                u_sim_seq[ctrl_idx] = gen_ctrl.calculate_ff_fb_u(
                    ctrl_out.K_seq[ctrl_idx],
                    ctrl_out.u_seq[ctrl_idx],
                    ctrl_out.x_seq[ctrl_idx],
                    x_sim_seq[ctrl_idx],
                )
                data.ctrl = (u_sim_seq[ctrl_idx]).reshape(-1)
            else:
                break
        if len(frames) < data.time * framerate:
            update_plt_frame(renderer, data, frames, img_set)
        # data.ctrl = 0.0
        mujoco.mj_step(model, data)
        ctrl_count += 1
    model.opt.timestep = prev_timestep
    return img_set, frames
