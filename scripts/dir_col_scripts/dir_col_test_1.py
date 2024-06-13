import jax.numpy as jnp
import jax
import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt


import ctrl_sandbox.dir_col_funcs as dir_col
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.visualize_dyn_funcs as vis_dyn
import ctrl_sandbox.analyze_ilqr_output_funcs as analyze


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    m = 1.0
    moi = 0.0
    g = 9.81
    b = 0.1
    l = 1.0
    dyn_func_obj = dyn.single_pend_dyn(m=m, moi=moi, g=g, b=b, l=l)
    Q = jnp.array([[1, 0], [0, 1]], dtype=float) * 0.001
    R = jnp.array([1.0], dtype=float)
    Qf = Q

    len_seq = 200
    x_len = 2
    u_len = 1
    time_step = 0.05
    x_des_seq = np.zeros((len_seq, x_len), dtype=float)
    x_des_seq[:, 0] = np.pi
    u_des_seq = np.zeros((len_seq - 1,), dtype=float)
    x_init = np.zeros((2,), dtype=float)
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

    x_seq = np.zeros((len_seq, x_len), dtype=float)
    u_seq = np.ones((len_seq - 1,), dtype=float) * 0.001
    opt_vec = dir_col.create_opt_vec_from_x_u_seqs(x_seq, u_seq)
    constraints = {
        "type": "eq",
        "fun": dir_col_config.dyn_colloc_calc,
        "jac": dir_col_config.dyn_colloc_diff,
    }

    opt_out = sciopt.minimize(
        dir_col_config.dir_col_cost_calc,
        opt_vec,
        jac=dir_col_config.dir_col_cost_diff,
        constraints=constraints,
        options={"disp": True, "iprint": 3, "maxiter": 200},
    )

    x_seq_out, u_seq_out = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_out.x)

    time_vec = np.linspace(0, (len_seq - 1) * time_step, len_seq)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot vectors 1 and 2 on the first subplot
    ax1.plot(time_vec, x_seq_out[:, 0], label="th", marker="o")
    ax1.plot(time_vec, x_seq_out[:, 1], label="thdot", marker="o")
    ax1.set_title("states")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Values")
    ax1.grid(True)
    ax1.legend()

    # Plot vector 3 on the second subplot
    ax2.plot(time_vec[:-1], u_seq_out, label="control", color="r", marker="o")
    ax2.set_title("control")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Values")
    ax2.grid(True)
    ax2.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 8))

    pend_vis_obj = vis_dyn.pm_pend_animation(l, np.array(x_seq_out), time_step, fig2)
    pend_vis_obj.create_animation()
    # Show the plot
    plt.show()
