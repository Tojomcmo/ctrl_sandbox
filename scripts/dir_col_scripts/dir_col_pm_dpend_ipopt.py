import jax.numpy as jnp
import jax
import numpy as np
import scipy.optimize as sciopt
from cyipopt import minimize_ipopt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


import ctrl_sandbox.dir_col_funcs as dir_col
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.visualize_dyn_funcs as vis_dyn


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    save_ani_bool = True
    ani_save_location = "media_output/"
    ani_save_name = "acrobot_dir_col"
    os.makedirs(ani_save_location, exist_ok=True)

    x_len = 4
    time_step = 0.03
    len_seq = 200
    shoulder_act = False
    elbow_act = True
    ctrl_upper_bound = 20
    ctrl_lower_bound = -20
    m = 1.0
    g = 9.81
    b = 0.1
    l = 1.0

    dyn_func_obj = dyn.double_pm_pend_dyn(
        g=g,
        m1=m,
        m2=m,
        l1=l,
        l2=l,
        b1=b,
        b2=b,
        shoulder_act=shoulder_act,
        elbow_act=elbow_act,
    )

    Q = jnp.diag(jnp.array([1, 1, 1, 1], dtype=float)) * 0.0001
    Qf = Q

    if shoulder_act is True and elbow_act is True:
        u_len = 2
        R = jnp.diag(jnp.array([1, 1], dtype=float))
        u_des_seq = np.zeros((len_seq - 1, 2), dtype=float)
        u_seed_seq = np.ones((len_seq - 1, 2), dtype=float) * 0.001
    else:
        u_len = 1
        R = jnp.array([1.0], dtype=float)
        u_des_seq = np.zeros((len_seq - 1,), dtype=float)
        u_seed_seq = np.ones((len_seq - 1,), dtype=float) * -0.001

    x_des_seq = np.zeros((len_seq, x_len), dtype=float)
    x_des_seq[:, 0] = np.pi
    x_des_seq[:, 1] = np.pi
    x_init = np.zeros((x_len,), dtype=float)
    cost_func_obj = cost.cost_quad_x_and_u(Q, R, Qf, x_des_seq, u_des_seq)
    dir_col_config = dir_col.dirColConfig(
        cost_func_obj.cost_func_quad_state_and_control_scan_compatible,
        dyn_func_obj.cont_dyn_func,
        time_step,
        x_len,
        u_len,
        len_seq,
        x_init,
        x_des_seq[-1],
    )

    x_seed_seq = np.zeros((len_seq, x_len), dtype=float)
    x_seed_seq[:, 0] = np.linspace(0, x_des_seq[-1, 0], len_seq)
    x_seed_seq[:, 1] = np.linspace(0, x_des_seq[-1, 1], len_seq)
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seed_seq, u_seed_seq)
    constraints = {
        "type": "eq",
        "fun": dir_col_config.dyn_colloc_calc,
        "jac": dir_col_config.dyn_colloc_diff,
    }

    x_bnds = [(-np.inf, np.inf)] * (len_seq * x_len)
    u_bnds = [(ctrl_lower_bound, ctrl_upper_bound)] * ((len_seq - 1) * u_len)
    bnds = x_bnds + u_bnds

    opt_out = minimize_ipopt(
        dir_col_config.dir_col_cost_calc,
        opt_vec,
        jac=dir_col_config.dir_col_cost_diff,
        bounds=bnds,
        constraints=constraints,
        options={"disp": 5},
    )

    x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_out.x)

    time_vec = np.linspace(0, (len_seq - 1) * time_step, len_seq)

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])  # row 0, col 0
    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    ax3 = fig.add_subplot(gs[1, 1])  # row 1, span all columns

    vis_dyn.plot_dpend_pos_to_ax(ax2, x_seq, time_vec)
    vis_dyn.plot_dpend_control_to_ax(
        ax3, x_seq, u_seq, time_vec, shoulder_act, elbow_act
    )

    pend_vis_obj = vis_dyn.double_pm_pend_animation(
        l,
        l,
        np.array(x_seq),
        time_step,
        fig,
        ax1,
    )
    pend_vis_obj.create_animation()
    plt.tight_layout()
    # Show the plot
    plt.show()

    if save_ani_bool:
        print("saving animation...")
        filename: str | os.PathLike = ani_save_location + ani_save_name
        pend_vis_obj.save_animation_gif(filename)
        print("animation saved!")
