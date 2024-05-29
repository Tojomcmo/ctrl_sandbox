import numpy as np
import matplotlib.pyplot as plt
import os

import dyn_functions as dyn
import visualize_dyn_funcs as vis_dyn
import gen_ctrl_funcs as gen_ctrl
import analyze_ilqr_output_funcs as analyze


if __name__ == "__main__":
    save_ani_bool = False
    lock_arm_num = 0
    dt = 0.005
    len_seq = 1000
    time_vec = np.arange(0, len_seq * dt, dt)
    dpend_sys = dyn.double_pm_pend_dyn(
        g=9.81,
        m1=1.0,
        l1=1.0,
        m2=1.0,
        l2=1.0,
        b1=0.0,
        b2=0.0,
        shoulder_act=True,
        elbow_act=True,
    )
    x_init = np.array([3.14, 3.14, 0.0, 0.0])
    u_vec = np.array([0.0, 0.0])

    x_seq = np.zeros((len_seq, 4))
    pot_energy_seq = np.zeros((len_seq, 1))
    kin_energy_seq = np.zeros((len_seq, 1))
    tot_energy_seq = np.zeros((len_seq, 1))
    x_seq[0] = x_init
    pot_energy_seq[0] = dpend_sys.calculate_potential_energy(x_init)
    kin_energy_seq[0] = dpend_sys.calculate_kinetic_energy(x_init)
    tot_energy_seq[0] = dpend_sys.calculate_total_energy(x_init)
    for k in range(len_seq - 1):
        if lock_arm_num == 1:
            x_seq[k, 0] = 0.0
            x_seq[k, 2] = 0.0
        elif lock_arm_num == 2:
            x_seq[k, 1] = x_seq[k, 0]
            x_seq[k, 3] = x_seq[k, 2]
        x_seq[k + 1] = gen_ctrl.step_rk4(dpend_sys.cont_dyn_func, dt, x_seq[k], u_vec)
        pot_energy_seq[k + 1] = dpend_sys.calculate_potential_energy(x_seq[k + 1])
        kin_energy_seq[k + 1] = dpend_sys.calculate_kinetic_energy(x_seq[k + 1])
        tot_energy_seq[k + 1] = dpend_sys.calculate_total_energy(x_seq[k + 1])

    fig1 = plt.figure(figsize=[6, 4])
    plt.plot(np.arange(0, len_seq * dt, dt), pot_energy_seq, label="potential")
    plt.plot(np.arange(0, len_seq * dt, dt), kin_energy_seq, label="kinetic")
    plt.plot(np.arange(0, len_seq * dt, dt), tot_energy_seq, label="total")
    plt.title("Passive undamped pendulum energy conservation")
    plt.xlabel("time [s]")
    plt.ylabel("energy [J]")
    plt.legend()

    fig = plt.figure(figsize=[10, 8])
    pend_animation = vis_dyn.double_pm_pend_animation(
        dpend_sys.l1, dpend_sys.l2, x_seq, dt, fig
    )
    pend_animation.create_double_pend_animation()
    pend_animation.show_plot()

    if save_ani_bool == True:
        print("saving animation...")
        filename: str | os.PathLike = "/Users/thomasmoriarty/Desktop/dpend_passive"
        pend_animation.save_animation_gif(filename)
        print("animation saved!")
