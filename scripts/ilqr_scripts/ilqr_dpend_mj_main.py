import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animate
import mujoco as mujoco

import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost
import mjcf_models as mj_models
import mujoco_funcs as mj_funcs
import visualize_mj_funcs as mj_vis
import visualize_dyn_funcs as vis_dyn

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    mujoco_vis = True
    # ------- Define controller configuration -------#
    time_step = 0.01
    len_seq = 400
    num_states = 4
    num_controls = 1
    shoulder_act = False
    elbow_act = True
    mj_ctrl = True
    dpend_model_obj = mj_models.mjcf_dpend(shoulder_act, elbow_act)
    Q_cost = (
        np.array(
            [[10.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 5.0]],
            dtype=float,
        )
        * 0.1
    )
    R_cost = np.array([[1.0]], dtype=float) * 5.0
    # R_cost  = np.array([[1.0, 0],[0, 1.0]],dtype=float)*0.5
    Qf_cost = (
        np.array(
            [[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
            dtype=float,
        )
        * 10000.0
    )

    # ---------- initialize ilqr configuration object
    ctrl_config = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
    ctrl_config.converge_crit = 1e-6
    ctrl_config.max_iter = 500
    ctrl_config.fp_max_iter = 10
    # ---------- create desired trajectory ----------#
    # traj_gen_dyn_func_params = dyn.nlPendParams(g=9.81,b=5.0,l=1.0)
    # x_tg_init_vec = np.array([[0.0],[0.0],[0.0],[0.0]])
    # u_tg_seq      = np.ones([len_seq-1,1,1])
    # traj_gen_cont_dyn_func = lambda x,u: dyn.pend_dyn_nl(traj_gen_dyn_func_params,x,u)
    # traj_gen_disc_dyn_func = lambda x,u: gen_ctrl.step_rk4(traj_gen_cont_dyn_func, ilqr_config.time_step, x, u)
    # x_des_seq_traj_gen         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_disc_dyn_func,x_tg_init_vec, u_tg_seq)

    # ---------- set system init ----------#
    x_init_vec = jnp.array([0.0, 0.0, 0.0, 0.0])
    x_sim_init_vec = jnp.array([0.0, 0.0, 0.0, 0.0])

    ctrl_target_condition = 2

    if ctrl_target_condition == 1:
        # u_init_seq = np.ones([len_seq-1, num_controls,1]) * (-1.0)
        # u_des_seq  = u_tg_seq
        # x_des_seq  = x_des_seq_traj_gen
        pass
    elif ctrl_target_condition == 2:
        x_des_seq = jnp.zeros([len_seq, num_states])
        x_des_seq = x_des_seq.at[:, 0].set(jnp.pi)
        x_des_seq = x_des_seq.at[:, 1].set(0.0)
        u_init_seq = jnp.ones([len_seq - 1, num_controls]) * (0.01)
        u_des_seq = jnp.zeros([len_seq - 1, num_controls])

    else:
        raise ValueError("invalid ctrl_target_condition")

    # ------- complete ilqr configuration --------#
    cost_func_obj = cost.cost_quad_x_and_u(
        Q_cost, R_cost, Qf_cost, x_des_seq=x_des_seq, u_des_seq=u_des_seq
    )

    ctrl_config.config_for_mujoco(dpend_model_obj.get_mjcf_model())
    ctrl_config.config_cost_func(
        cost_func_obj.cost_func_quad_state_and_control_scan_compatible
    )
    ctrl_config.create_curried_funcs()

    # ----- Run iLQR algorithm -----#

    ctrl_out = ilqr.run_ilqr_controller(ctrl_config, x_init_vec, u_init_seq)

    # ------- Simulate controller output --------#
    mjcf_obj = mj_models.mjcf_dpend(shoulder_act, elbow_act)
    mj_model, _, mj_data = mj_funcs.create_mujoco_model(
        mjcf_obj.get_mjcf_model(), time_step=time_step
    )
    sim_disc_dyn_func = lambda x_k, u_k: mj_funcs.fwd_sim_mj_w_ctrl_step(
        mj_model, mj_data, np.array(x_k), np.array(u_k)
    )
    print("simulating with mujoco...")
    x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(
        sim_disc_dyn_func, ctrl_out, x_sim_init_vec
    )

    # ------ Create mujoco video --------#
    if mujoco_vis is True:
        framerate = 30
        fig1, ax1 = plt.subplots()
        mjvid_model, mjvid_renderer, mjvid_data = mj_funcs.create_mujoco_model(
            ctrl_config.mjcf_model, ctrl_config.time_step
        )
        mjvid_renderer.update_scene(mjvid_data, "fixed")
        scene_option = mujoco.MjvOption()  # type:ignore
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False  # type:ignore
        img_set, frames = mj_vis.create_mj_video_ilqr_w_ctrl(
            mjvid_model,
            mjvid_data,
            mjvid_renderer,
            scene_option,
            framerate,
            x_sim_init_vec,
            ctrl_out,
            ctrl_config,
        )
        ani = animate.ArtistAnimation(fig1, img_set, interval=int(1 / framerate * 1000))

    # ------- plot simulation and controller outputs ------#
    print("animating simulation output...")
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])  # row 0, col 0
    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    ax3 = fig.add_subplot(gs[1, 1])  # row 1, span all columns

    vis_dyn.plot_ilqr_dpend_act_and_cost_axes(
        shoulder_act, elbow_act, ctrl_out, u_sim_seq, ax2, ax3
    )

    h_bar = 1.0
    r_bar = 0.05
    m_bar = 1.0
    d_bar = h_bar / 2
    moi = (1 / 12) * h_bar * (m_bar**2 + 3 * r_bar**2)
    dyn_func_sys_vis = dyn.double_pend_rel_dyn(
        g=9.81,
        m1=m_bar,
        moi1=moi,
        d1=d_bar,
        l1=h_bar,
        m2=m_bar,
        moi2=moi,
        d2=d_bar,
        l2=h_bar,
        b1=0.0,
        b2=0.0,
        shoulder_act=shoulder_act,
        elbow_act=elbow_act,
    )

    pend_animation = vis_dyn.double_pend_animation(
        dyn_func_sys_vis.get_animate_value_dict(),
        np.array(x_sim_seq),
        time_step,
        fig,
        ax1,
        th2="rel",
    )
    pend_animation.create_double_pend_animation()
    plt.tight_layout()
    plt.show()
