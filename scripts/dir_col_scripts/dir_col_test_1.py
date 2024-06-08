import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt


import ctrl_sandbox.dir_col_funcs as dir_col
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost


if __name__ == "__main__":
    dyn_func_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=9.81, b=0.1, l=1.0)
    Q = jnp.array([[1, 0], [0, 1]], dtype=float)
    R = jnp.array([0.1], dtype=float)
    Qf = Q

    len_seq = 20
    x_len = 2
    u_len = 1
    time_step = 0.1
    x_des_seq = np.zeros((len_seq, x_len), dtype=float)
    x_des_seq[:, 0] = np.pi
    u_des_seq = np.zeros((len_seq,), dtype=float)
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
    u_seq = np.ones((len_seq,), dtype=float) * 0.001
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
    )

    x_seq_out, u_seq_out = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_out.x)

    print(x_seq_out)
    print(u_seq_out)
