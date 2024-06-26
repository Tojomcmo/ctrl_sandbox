import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import ctrl_sandbox.ilqr_funcs as ilqr
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.integrate_funcs as integrate
import ctrl_sandbox.visualize_dyn_funcs as vis_dyn
import ctrl_sandbox.mujoco_funcs as mj_funcs
import ctrl_sandbox.mjcf_models as mjcf
import ctrl_sandbox.simulate_funcs as sim


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    # ------- Define controller configuration -------#
    save_ani_bool = True
    sim_with_mj = False
    ani_save_location = "media_output/"
    ani_save_name = "acrobot_ilqr"
    fig_title = "iLQR: acrobot"
    os.makedirs(ani_save_location, exist_ok=True)
    time_step = 0.01
    len_seq = 500
    num_states = 4
    num_controls = 1
    shoulder_act = False
    elbow_act = True
    Q_cost = (
        np.array(
            [[10.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 5.0]],
            dtype=float,
        )
        * 0.05
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

    h_bar = 1.0
    r_bar = 0.05
    m_bar = 1.0
    d_bar = h_bar / 2
    moi = (1 / 12) * h_bar * (m_bar**2 + 3 * r_bar**2)
    dyn_func_sys_ctrl = dyn.double_pend_rel_dyn(
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

    # ---------- initialize ilqr configuration object --------------#
    ilqr_config = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
    ilqr_config.converge_crit = 1e-6
    ilqr_config.max_iter = 500
    ilqr_config.fp_max_iter = 10

    # ---------- create simulation system for post algorithm test ----------#
    dyn_func_sys_sim = dyn_func_sys_ctrl
    # ---------- set system state init and desired trajectories ----------#
    x_init_vec = jnp.array([0.0, 0.0, 0.0, 0.0])
    u_init_seq = jnp.ones([len_seq - 1, num_controls], dtype=float) * 0.01

    x_sim_init_vec = jnp.array([0.0, 0.0, 0.0, 0.0])

    x_des_seq = jnp.zeros([len_seq, num_states], dtype=float)
    x_des_seq = x_des_seq.at[:, 0].set(jnp.pi)
    x_des_seq = x_des_seq.at[:, 1].set(0.0)
    u_des_seq = jnp.zeros([len_seq - 1, num_controls], dtype=float)

    # ------- complete ilqr configuration --------#
    cost_func_obj = cost.cost_quad_x_and_u(
        Q_cost, R_cost, Qf_cost, x_des_seq=x_des_seq, u_des_seq=u_des_seq
    )

    ilqr_config.config_for_dyn_func(dyn_func_sys_ctrl.cont_dyn_func, integrate.step_rk4)
    ilqr_config.config_cost_func(
        cost_func_obj.cost_func_quad_state_and_control_scan_compatible
    )
    ilqr_config.create_curried_funcs()

    # ----- Run iLQR algorithm -----#
    ctrl_out = ilqr.run_ilqr_controller(ilqr_config, x_init_vec, u_init_seq)
    # ------- Simulate controller output --------#

    if sim_with_mj is True:
        mjcf_obj = mjcf.mjcf_dpend(shoulder_act, elbow_act)
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
    else:
        sim_disc_dyn_func = lambda k, x_k, u_k: integrate.step_rk4(
            dyn_func_sys_sim.cont_dyn_func, ilqr_config.time_step, x_k, u_k
        )
        control_func = sim.prep_ff_fb_u_for_sim(
            ctrl_out.u_seq, ctrl_out.x_seq, ctrl_out.K_seq
        )
        measure_func = sim.direct_pass
        disturb_func = sim.direct_pass
        noise_func = sim.direct_pass
        print("simulating with dpend func...")
        x_sim_seq, x_est_sim_seq, u_sim_seq = sim.sim_dyn(
            sim_disc_dyn_func,
            control_func,
            measure_func,
            disturb_func,
            noise_func,
            x_sim_init_vec,
            ctrl_out.len_seq,
        )

    # ------- plot simulation and controller outputs ------#
    print("animating simulation output...")
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(fig_title)
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])  # row 0, col 0
    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    ax3 = fig.add_subplot(gs[1, 1])  # row 1, span all columns

    vis_dyn.plot_ilqr_dpend_act_and_cost_axes(
        shoulder_act, elbow_act, ctrl_out, u_sim_seq, ax2, ax3
    )

    pend_animation = vis_dyn.double_pend_animation(
        dyn_func_sys_sim.get_animate_value_dict(),
        np.array(x_sim_seq),
        time_step,
        fig,
        ax1,
        th2="rel",
    )
    pend_animation.create_double_pend_animation()
    plt.tight_layout()
    plt.show()

    if save_ani_bool:
        print("saving animation...")
        filename: str | os.PathLike = ani_save_location + ani_save_name
        pend_animation.save_animation_gif(filename)
        print("animation saved!")
