import mujoco as mujoco
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def create_mujoco_model(mjcf_model: str, time_step):
    model = mujoco.MjModel.from_xml_string(mjcf_model)  # type:ignore
    model.opt.timestep = time_step
    renderer = mujoco.Renderer(model)
    data = mujoco.MjData(model)  # type:ignore
    return model, renderer, data


def fwd_sim_mj_w_ctrl_step(model, data, x, u):
    set_mj_state_vec(data, x)
    data.ctrl = u.reshape(-1)
    mujoco.mj_step(model, data)  # type:ignore
    return get_state_vec(data)


def fwd_sim_mj_w_ctrl(model, data, x_init, u_seq):
    """
    function simulates provided model and data forward in time using u sequence
    and returns state trajectory x_seq and linearized state and control matricies A
    and B at each pair x_seq, u_seq
    """
    mujoco.mj_resetData(model, data)  # type:ignore
    set_mj_state_vec(data, x_init)
    mujoco.mj_forward(model, data)  # type:ignore
    x_seq = jnp.zeros((len(u_seq) + 1, model.nv * 2))
    x_seq = x_seq.at[0].set(x_init)
    for idx in range(len(u_seq)):
        data.ctrl = (u_seq[idx]).reshape(-1)
        mujoco.mj_step(model, data)  # type:ignore
        x_seq = x_seq.at[idx + 1].set(get_state_vec(data))
        idx += 1
    return x_seq


def linearize_mj_seq(model, data, x_seq, u_seq):
    """
    function simulates provided model and data forward in time using u sequence
    and returns state trajectory x_seq and linearized state and control matricies A
    and B at each pair x_seq, u_seq
    """
    A_seq = jnp.zeros((len(u_seq), model.nv * 2, model.nv * 2))
    B_seq = jnp.zeros((len(u_seq), model.nv * 2, model.nu))
    for idx in range(len(u_seq)):
        mujoco.mj_resetData(model, data)  # type:ignore
        data.ctrl = (u_seq[idx]).reshape(-1)
        set_mj_state_vec(data, x_seq[idx])
        mujoco.mj_forward(model, data)  # type:ignore
        A_idx, B_idx = linearize_mujoco_state_and_control(model, data)
        A_seq = A_seq.at[idx].set(A_idx)
        B_seq = B_seq.at[idx].set(B_idx)
    return A_seq, B_seq


def linearize_mujoco_state_and_control(model, data, eps=1e-4, flg_centered=True):
    nu = model.nu
    nx = model.nv
    A = np.zeros((2 * nx, 2 * nx))
    B = np.zeros((2 * nx, nu))
    mujoco.mjd_transitionFD(
        model, data, eps, flg_centered, A, B, None, None
    )  # type:ignore
    return jnp.array(A), jnp.array(B)


def get_state_vec(data):
    return jnp.hstack((data.qpos, data.qvel)).reshape(-1)


def set_mj_state_vec(data, x_vec) -> None:
    x_vec_split = int(len(x_vec) / 2)
    data.qpos = x_vec[:x_vec_split]
    data.qvel = x_vec[x_vec_split:]


def set_mj_ctrl_vec(data, u_vec) -> None:
    data.ctrl = (u_vec).reshape(-1)


def fwd_sim_mj_w_ctrl_different_time_steps(model, data, x_init, u_seq, ts_sim, ts_ctrl):
    """
    function simulates provided model and data forward in time using u sequence
    and returns state trajectory x_seq and linearized state and control matricies A
    and B at each pair x_seq, u_seq
    """
    if int(ts_ctrl % ts_sim) != 0:
        raise ValueError(
            "control timestep must be an integer multiple of the sim timestep"
        )
    elif ts_ctrl < ts_sim:
        raise ValueError(
            "control timestep must be larger than or equal to the sim timestep"
        )
    nx = model.nv
    t_final = len(u_seq) * ts_ctrl
    sim_steps = int(t_final / ts_sim)
    steps_zoh = int(ts_ctrl / ts_sim)
    x_seq = np.zeros((len(u_seq) + 1, nx * 2, 1))
    time_seq = np.zeros(len(u_seq) + 1)

    mujoco.mj_resetData(model, data)  # type:ignore
    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    set_mj_state_vec(data, x_init)
    mujoco.mj_forward(model, data)  # type:ignore
    x_seq[0] = get_state_vec(data)
    assert (
        x_seq[0]
    ).tolist() == x_init.tolist(), "x_init vector improperly initialized or queried"
    data.ctrl = (u_seq[0]).reshape(-1)
    ctrl_idx = 0
    ctrl_count = 0

    for sim_idx in range(sim_steps + 1):
        if ctrl_count == steps_zoh:
            ctrl_count = 0
            ctrl_idx += 1
            x_seq[ctrl_idx] = get_state_vec(data)
            time_seq[ctrl_idx] = data.time
            if sim_idx < sim_steps:
                data.ctrl = (u_seq[ctrl_idx]).reshape(-1)
            else:
                break
        mujoco.mj_step(model, data)  # type:ignore
        ctrl_count += 1
    model.opt.timestep = prev_timestep
    return x_seq
