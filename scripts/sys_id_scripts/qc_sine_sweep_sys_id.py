import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode

import ctrl_sandbox.sysid.freq_resp_funcs as freq
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.integrate_funcs as integrate
import ctrl_sandbox.simulate_funcs as sim
import ctrl_sandbox.gen_graphing_funcs as gen_graph


if __name__ == "__main__":
    ts_s = 0.005
    freq_0_hz = 0.01
    freq_1_hz = 25.0
    amplitude = 1
    sweep_duration_s = 500.0
    type = "logarithmic"
    freq_res = 0.025

    g = 9.81
    ms = 400
    mu = 50
    ks = 35000
    bs = 1000
    kt = 190000

    fs_hz = 1 / ts_s
    nperseg = int(fs_hz / freq_res)
    num_segs = int(sweep_duration_s / ts_s / nperseg)

    qcar_dyn_obj = dyn.qc_model_dyn(ms=ms, mu=mu, ks=ks, kt=kt, bs=bs)
    disc_dyn_func = lambda k, x, u: integrate.step_rk4(
        qcar_dyn_obj.cont_dyn_func, ts_s, x, u
    )

    road_profile = freq.sine_sweep_up_down(
        freq_0_hz, freq_1_hz, amplitude, duration=sweep_duration_s, ts=ts_s, type=type
    )

    u_sweep_seq = np.zeros((len(road_profile), 2))
    u_sweep_seq[:, 1] = road_profile

    x_init = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=float)

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

    input_sig = u_sweep_seq[:, 1]
    t = jnp.linspace(0, sweep_duration_s, int(sweep_duration_s / ts_s), endpoint=True)

    freqs, frequency_response = freq.calc_freq_resp(
        np.array(input_sig), np.array(x_sweep_seq[:, 0]), fs=fs_hz, nperseg=nperseg
    )
    freq_resp_mag, freq_resp_phase = freq.calculate_mag_and_phase_from_complex(
        frequency_response
    )

    print("frequency response data collected. analyzing...")
    ############### calculate actual transfer function #############
    # Transfer function coefficients (high to low)

    num = [kt * bs / (mu * ms), ks * kt / (mu * ms)]
    den = [
        1,
        ((mu + ms) * bs) / (mu * ms),
        (ms * ks + ms * kt + mu * ks - bs**2) / (mu * ms),
        (bs * kt - bs * ks) / (mu * ms),
        (ks * kt) / (mu * ms),
    ]

    # Create the transfer function
    system = TransferFunction(num, den)
    # Generate frequency range
    w = np.logspace(-2, 2, 100)

    # Compute Bode plot
    w, mag, phase = bode(system, w)
    w = w / (2 * np.pi)

    ############## calculate transfer function fit from fr data ##############

    # set tf order for fit and initial parameter guess (low to high)
    num_order = 1
    den_order = 4
    num_coeff_guess = np.array([num[1], num[0]]) * 1.0
    den_coeff_guess = np.array([den[3], den[2], den[1], den[0]]) * 1.0
    # num_coeff_guess = np.array([1.0, 1.0])
    # den_coeff_guess = np.array([1.0, 1.0, 1.0, 1.0])

    # run tf estimation function
    params_initial = np.concatenate((num_coeff_guess, den_coeff_guess))
    num_coeffs, den_coeffs = freq.fit_lin_tf_to_fr_data(
        freqs, frequency_response, num_order, den_order, params_initial
    )
    print("numerator coeffs : ", num_coeffs)
    print("denominator coeffs : ", den_coeffs)
    # create transfer function from fit parameters
    tf_fit = lambda z: freq.transfer_function_model(num_coeffs, den_coeffs, z)
    fr_fit = freq.compute_tf_response_at_freqs_hz(tf_fit, freqs)
    fr_fit_mag, fr_fit_phase = freq.calculate_mag_and_phase_from_complex(fr_fit)

    ########### plot system outputs ##################
    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle("Pendulum time series")
    ax1.plot(t, input_sig)
    ax2.plot(t, x_sweep_seq[:-1, 0])
    plt.savefig("media_output/sweep_pend_time.png")

    x_lim = (0.01, 100)
    y_lim = gen_graph.set_lim_based_on_defined_lim(
        x_lim, freqs, freq_resp_mag, border=0.1
    )
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 1, 1)
    plt.semilogx(
        freqs, freq_resp_mag, marker="o", linestyle="", label="frequency response data"
    )
    plt.semilogx(w, mag, color="r", label="actual transfer function")
    plt.semilogx(freqs, fr_fit_mag, color="g", linestyle="--", label="fit model")
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
    plt.semilogx(
        freqs,
        np.degrees(freq_resp_phase),
        marker="o",
        linestyle="",
        label="frequency response data",
    )
    plt.semilogx(w, phase, color="r", label="actual transfer function")
    plt.semilogx(
        freqs, np.degrees(fr_fit_phase), color="g", linestyle="--", label="fit model"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [radians]")
    plt.xlim(x_lim)
    plt.ylim(-200, 200)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig("media_output/sweep_pend_freq.png")
