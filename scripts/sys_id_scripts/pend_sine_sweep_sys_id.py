import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import ctrl_sandbox.sysid.freq_resp_funcs as freq
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.integrate_funcs as integrate
import ctrl_sandbox.simulate_funcs as sim
import ctrl_sandbox.gen_graphing_funcs as gen_graph


if __name__ == "__main__":
    ts_s = 0.01
    freq_0_hz = 0.0
    freq_1_hz = 5.0
    amplitude = 0.1
    sweep_duration_s = 1000.0

    freq_res = 0.025

    fs_hz = 1 / ts_s
    nperseg = int(fs_hz / freq_res)
    num_segs = int(sweep_duration_s / ts_s / nperseg)

    pend_dyn_obj = dyn.single_pm_pend_dyn(g=9.8, b=0.2, l=1.0)
    disc_dyn_func = lambda k, x, u: integrate.step_rk4(
        pend_dyn_obj.cont_dyn_func, ts_s, x, u
    )

    u_sweep_seq = freq.sine_sweep_up_down(
        freq_0_hz, freq_1_hz, amplitude, duration=sweep_duration_s, ts=ts_s
    )

    x_init = jnp.array([0.0, 0.0], dtype=float)

    control_func = sim.prep_ff_u_for_sim(jnp.array(u_sweep_seq, dtype=float))
    measure_func = sim.direct_pass
    disturb_func = sim.direct_pass
    noise_func = sim.direct_pass
    x_sweep_seq, _, u_sim_seq = sim.sim_dyn(
        disc_dyn_func,
        control_func,
        measure_func,
        disturb_func,
        noise_func,
        x_init,
        len(u_sweep_seq) + 1,
    )

    t = jnp.linspace(0, sweep_duration_s, int(sweep_duration_s / ts_s), endpoint=True)

    freqs, frequency_response = freq.calc_freq_resp(
        np.array(u_sweep_seq), np.array(x_sweep_seq[:, 0]), fs=fs_hz, nperseg=nperseg
    )
    freq_resp_mag, freq_resp_phase = freq.calculate_mag_and_phase_from_complex(
        frequency_response
    )

    # calculate approximate transfer function
    num_order = 0
    den_order = 2
    params_initial = np.array([1.0, 1.0, 1.0])
    num_coeffs, den_coeffs = freq.fit_lin_tf_to_fr_data_least_squares(
        ts_s, freqs[1:], frequency_response[1:], num_order, den_order, params_initial
    )
    print("numerator coeffs : ", num_coeffs)
    print("denominator coeffs : ", den_coeffs)

    tf = freq.create_tf_for_fit(num_order, den_order)
    # z_domain_freqs = freq.create_z_transform_for_freqs(freqs[1:], ts_s)
    z_domain_freqs = np.complex128(freqs[1:] * 1j)
    H_fit_out = tf(np.concatenate((num_coeffs, den_coeffs)), z_domain_freqs)
    H_fit_mag, H_fit_phase = freq.calculate_mag_and_phase_from_complex(H_fit_out)

    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle("Pendulum time series")
    ax1.plot(t, u_sweep_seq)
    ax2.plot(t, x_sweep_seq[:-1, 0])
    plt.savefig("media_output/sweep_pend_time.png")

    x_lim = (0, 1)
    y_lim = gen_graph.set_lim_based_on_defined_lim(x_lim, freqs, freq_resp_mag)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(
        freqs, freq_resp_mag, marker="o", linestyle="", label="frequency response data"
    )
    plt.plot(freqs[1:], H_fit_mag, color="r", label="fit transfer function")
    plt.title("Transfer Function Estimate")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend()
    plt.text(
        3 * np.sum(x_lim) / 4, float(1 * np.sum(y_lim) / 3), f"num_segs = {num_segs}"
    )
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.degrees(freq_resp_phase))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [radians]")
    plt.xlim(x_lim)
    plt.ylim(-180, 180)
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig("media_output/sweep_pend_freq.png")
