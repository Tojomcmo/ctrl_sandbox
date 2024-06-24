from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import mujoco as mujoco

import ctrl_sandbox.ilqr_funcs as ilqr
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.analyze_ilqr_output_funcs as analyze
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.mjcf_models as mj_models
import ctrl_sandbox.mujoco_funcs as mj_funcs
import ctrl_sandbox.visualize_mj_funcs as mj_vis
import ctrl_sandbox.integrate_funcs as integrate

if __name__ == "__main__":
    # ----- define timestep and sequence length -----#
    time_step = 0.1
    len_seq = 40
    num_states = 2
    num_controls = 1
    mj_ctrl = True
    Q_cost = np.array([[10.0, 0], [0.0, 1.0]]) * 1.0
    R_cost = np.array([[1.0]])
    Qf_cost = np.array([[10.0, 0], [0.0, 1.0]]) * 10.0

    # ---------- initialize ilqr configuration object
    ilqr_config = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
    ilqr_config.max_iter = 40
    ilqr_config.log_ctrl_history = True

    # ---------- create desired trajectory ----------#
    dyn_func_traj_gen_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=9.81, b=5.0, l=1.0)
    x_tg_init_vec = jnp.array([0.0, 0.0])
    u_tg_seq = jnp.ones([len_seq - 1, 1])
    traj_gen_disc_dyn_func = lambda x, u: integrate.step_rk4(
        dyn_func_traj_gen_obj.cont_dyn_func, ilqr_config.time_step, x, u
    )
    x_des_seq_traj_gen = gen_ctrl.simulate_forward_dynamics_seq(
        traj_gen_disc_dyn_func, x_tg_init_vec, u_tg_seq
    )

    # ---------- create simulation system ----------#
    dyn_func_sim_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=9.81, b=1.0, l=1.0)
    x_sim_init_vec = jnp.array([-1.0, -1.0])

    # ---------- set system init ----------#
    x_init_vec = x_tg_init_vec

    ctrl_target_condition = 2

    if ctrl_target_condition == 1:
        u_init_seq = jnp.ones([len_seq - 1, 1]) * (-1)
        u_des_seq = u_tg_seq
        x_des_seq = x_des_seq_traj_gen

    elif ctrl_target_condition == 2:
        x_des_seq = jnp.zeros([len_seq, 2])
        x_des_seq = x_des_seq.at[:, 0].set(jnp.pi)
        u_init_seq = jnp.ones([len_seq - 1, 1]) * (-1)
        u_des_seq = jnp.zeros([len_seq - 1, 1])
    else:
        raise ValueError("invalid ctrl_target_condition")

    # ------- complete ilqr configuration --------#
    cost_func = cost.cost_quad_x_and_u(
        Q_cost, R_cost, Qf_cost, x_des_seq=x_des_seq, u_des_seq=u_des_seq
    )
    if mj_ctrl is True:
        ilqr_config.config_for_mujoco(
            mj_models.create_MJCF_single_pend_m_d_mod(1, 1, 1)
        )
    else:
        pass
    ilqr_config.config_cost_func(
        cost_func.cost_func_quad_state_and_control_scan_compatible
    )
    ilqr_config.create_curried_funcs()

    # ----- Run iLQR algorithm -----#

    # run ilqr controller
    controller_output = ilqr.run_ilqr_controller(ilqr_config, x_init_vec, u_init_seq)
    # ------- Simulate controller output --------#
    if ilqr_config.mj_ctrl is True:
        mjsim_model, _, mjsim_data = mj_funcs.create_mujoco_model(
            ilqr_config.mjcf_model, ilqr_config.time_step
        )
        sim_dyn_disc_func = lambda x, u: (
            mj_funcs.fwd_sim_mj_w_ctrl_step(mjsim_model, mjsim_data, x, u)
        )
    else:
        sim_dyn_disc_func = lambda x, u: integrate.step_rk4(
            dyn_func_sim_obj.cont_dyn_func, ilqr_config.time_step, x, u
        )
    x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(
        sim_dyn_disc_func, controller_output, x_sim_init_vec
    )

    # ------ Create mujoco video --------#
    if ilqr_config.mj_ctrl is True:
        framerate = 30
        fig1, ax1 = plt.subplots()
        mjvid_model, mjvid_renderer, mjvid_data = mj_funcs.create_mujoco_model(
            ilqr_config.mjcf_model, ilqr_config.time_step
        )
        mjvid_renderer.update_scene(mjvid_data, "fixed")
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        img_set, frames = mj_vis.create_mj_video_ilqr_w_ctrl(
            mjvid_model,
            mjvid_data,
            mjvid_renderer,
            scene_option,
            framerate,
            x_sim_init_vec,
            controller_output,
            ilqr_config,
        )
        ani = animate.ArtistAnimation(fig1, img_set, interval=int(1 / framerate * 1000))
    # ------- plot simulation and controller outputs ------#

    analyze.plot_ilqr_iter_sim_ctrl_cost(
        controller_output, x_des_seq, x_sim_seq, u_sim_seq
    )
    plt.show()
