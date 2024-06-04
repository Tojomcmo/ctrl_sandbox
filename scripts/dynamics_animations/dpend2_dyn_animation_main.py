import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.visualize_dyn_funcs as vis_dyn
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl

if __name__ == "__main__":
    save_ani_bool = False
    dt = 0.005
    len_seq = 600
    time_vec = np.arange(0, len_seq * dt, dt)
    h_bar = 1.0
    r_bar = 0.05
    m_bar = 1.0
    d_bar = h_bar / 2
    moi = (1 / 12) * h_bar * (m_bar**2 + 3 * r_bar**2)
    dpend_sys = dyn.double_pend_abs_dyn(
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
        shoulder_act=True,
        elbow_act=True,
    )
    x_init = jnp.array([3.14, 3.14, 0.0, 0.0])
    u_vec = jnp.array([0.0, 0.0])

    x_seq = jnp.zeros((len_seq, 4))
    pot_energy_seq = jnp.zeros((len_seq, 1))
    kin_energy_seq = jnp.zeros((len_seq, 1))
    tot_energy_seq = jnp.zeros((len_seq, 1))
    x_seq = x_seq.at[0].set(x_init)
    pot_energy_seq = pot_energy_seq.at[0].set(
        dpend_sys.calculate_potential_energy(x_init)
    )
    kin_energy_seq = kin_energy_seq.at[0].set(
        dpend_sys.calculate_kinetic_energy(x_init)
    )
    tot_energy_seq = tot_energy_seq.at[0].set(dpend_sys.calculate_total_energy(x_init))
    for k in range(len_seq - 1):
        x_seq = x_seq.at[k + 1].set(
            gen_ctrl.step_rk4(dpend_sys.cont_dyn_func, dt, x_seq[k], u_vec)
        )
        pot_energy_seq = pot_energy_seq.at[k + 1].set(
            dpend_sys.calculate_potential_energy(x_seq[k + 1])
        )
        kin_energy_seq = kin_energy_seq.at[k + 1].set(
            dpend_sys.calculate_kinetic_energy(x_seq[k + 1])
        )
        tot_energy_seq = tot_energy_seq.at[k + 1].set(
            dpend_sys.calculate_total_energy(x_seq[k + 1])
        )

    fig1 = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(0, len_seq * dt, dt), pot_energy_seq, label="potential")
    plt.plot(np.arange(0, len_seq * dt, dt), kin_energy_seq, label="kinetic")
    plt.plot(np.arange(0, len_seq * dt, dt), tot_energy_seq, label="total")
    plt.legend()

    fig = plt.figure(figsize=(10, 8))
    pend_animation = vis_dyn.double_pm_pend_animation(
        dpend_sys.l1, dpend_sys.l2, np.array(x_seq), dt, fig
    )
    pend_animation.create_double_pend_animation()
    pend_animation.show_plot()

    if save_ani_bool:
        print("saving animation...")
        filename: str | os.PathLike = "/Users/thomasmoriarty/Desktop/dpend_passive"
        pend_animation.save_animation_gif(filename)
        print("animation saved!")
