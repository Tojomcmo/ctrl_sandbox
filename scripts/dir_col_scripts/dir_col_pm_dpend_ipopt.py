import jax.numpy as jnp
import jax
import numpy as np
import scipy.optimize as sciopt
from cyipopt import minimize_ipopt
import matplotlib.pyplot as plt


import ctrl_sandbox.dir_col_funcs as dir_col
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.visualize_dyn_funcs as vis_dyn


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    x_len = 4
    time_step = 0.05
    len_seq = 190
    shoulder_act = False
    elbow_act = True
    ctrl_bound = 4
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

    Q = (
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        * 0
    )
    Qf = Q

    if shoulder_act is True and elbow_act is True:
        u_len = 2
        R = jnp.eye(2, dtype=float)
        u_des_seq = np.zeros((len_seq - 1, 2), dtype=float)
        u_seed_seq = np.ones((len_seq - 1, 2), dtype=float) * 0.001
    else:
        u_len = 1
        R = jnp.array([1.0], dtype=float)
        u_des_seq = np.zeros((len_seq - 1,), dtype=float)
        u_seed_seq = np.ones((len_seq - 1,), dtype=float) * 0.001

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
    u_bnds = [(-ctrl_bound, ctrl_bound)] * ((len_seq - 1) * u_len)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot vectors 1 and 2 on the first subplot
    ax1.plot(time_vec, x_seq[:, 0], label="th1", marker="o")
    ax1.plot(time_vec, x_seq[:, 1], label="th2", marker="s")
    ax1.set_title("states")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Values")
    ax1.grid(True)
    ax1.legend()

    # Plot vector 3 on the second subplot
    if dyn_func_obj.shoulder_act is True and dyn_func_obj.elbow_act is True:
        ax2.plot(
            time_vec[:-1], u_seq[:, 0], label="shoulder ctrl", color="r", marker="o"
        )
        ax2.plot(
            time_vec[:-1], u_seq[:, 1], label="elbow control", color="b", marker="s"
        )
    else:
        if dyn_func_obj.shoulder_act is True and dyn_func_obj.elbow_act is False:
            ax2.plot(
                time_vec[:-1], u_seq, label="shoulder control", color="r", marker="o"
            )
        else:
            ax2.plot(time_vec[:-1], u_seq, label="elbow control", color="r", marker="o")
    ax2.set_title("control")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Values")
    ax2.grid(True)
    ax2.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 8))

    pend_vis_obj = vis_dyn.double_pm_pend_animation(
        l,
        l,
        np.array(x_seq),
        time_step,
        fig2,
    )
    pend_vis_obj.create_animation()
    # Show the plot
    plt.show()
