from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import mujoco as mujoco

import ctrl_sandbox.ilqr_funcs as ilqr
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.cost_functions as cost
import ctrl_sandbox.analyze_ilqr_output_funcs as analyze
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.mujoco_funcs as mj_funcs

if __name__ == "__main__":
    # ----- define timestep and sequence length -----#
    time_step = 0.01
    len_seq = 30
    num_states = 2
    num_controls = 1
    Q_cost = jnp.array([[10.0, 0], [0.0, 1.0]]) * 0.1
    R_cost = jnp.array([[0.1]]) * 0.1
    Qf_cost = jnp.array([[10.0, 0], [0.0, 1.0]]) * 1000.0
    dyn_func_ctrl_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=9.81, b=1.0, l=1.0)

    # ---------- initialize ilqr configuration object
    ilqr_config = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
    ilqr_config.max_iter = 40
    ilqr_config.log_ctrl_history = True

    # ---------- create desired trajectory ----------#
    dyn_func_traj_gen_obj = dyn.single_pend_dyn(m=1.0, moi=0.0, g=9.81, b=5.0, l=1.0)
    x_tg_init_vec = jnp.array([0.0, 0.0])
    u_tg_seq = jnp.ones([len_seq - 1, 1])
    traj_gen_disc_dyn_func = lambda x, u: gen_ctrl.step_rk4(
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

    ilqr_config.config_for_dyn_func(dyn_func_ctrl_obj.cont_dyn_func, gen_ctrl.step_rk4)
    ilqr_config.config_cost_func(
        cost_func.cost_func_quad_state_and_control_scan_compatible
    )
    ilqr_config.create_curried_funcs()

    # ----- Run iLQR algorithm -----#

    controller_output = ilqr.run_ilqr_controller(ilqr_config, x_init_vec, u_init_seq)

    # ------- Simulate controller output --------#
    if ilqr_config.mj_ctrl is True:
        mjsim_model, _, mjsim_data = mj_funcs.create_mujoco_model(
            ilqr_config.mj_model, time_step
        )
        sim_dyn_func_step = lambda x, u: (
            mj_funcs.fwd_sim_mj_w_ctrl(mjsim_model, mjsim_data, x, u)
        )[-1]
    else:
        sim_dyn_disc_func = lambda x, u: gen_ctrl.step_rk4(
            dyn_func_sim_obj.cont_dyn_func, ilqr_config.time_step, x, u
        )
    x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(
        sim_dyn_disc_func, controller_output, x_sim_init_vec
    )
    # ------- plot simulation and controller outputs ------#

    analyze.plot_ilqr_iter_sim_ctrl_cost(
        ilqr_config, controller_output, x_des_seq, x_sim_seq, u_sim_seq
    )
    plt.show()
