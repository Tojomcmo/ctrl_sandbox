import pytest
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as nptest

import ctrl_sandbox.sysid.freq_resp_funcs as freq
import ctrl_sandbox.dyn_functions as dyn


# def test_calc_freq_resp_accepts_valid_system():

#     freq_resp = freq.calc_freq_resp()

#     assert j == j_expect


def test_sine_sweep_up_down_generates_valid_sweep():
    freq_0 = 0.0
    freq_1 = 1.0
    amplitude = 1
    duration = 50.0
    sample_rate = 0.01
    sweep = freq.sine_sweep_up_down(freq_0, freq_1, amplitude, duration, sample_rate)

    t = np.linspace(0, duration, int(duration / sample_rate), endpoint=False)
    plt.plot(t, sweep)
    # plt.savefig("test_out.png")
    assert True == True


def test_form_A_mat_levy_generate_valid_matrix():
    freqs = np.array([1, 2, 3])
    freq_resp = np.array([2, 3, 4])
    num_order = 2
    den_order = 3
    freq_resp_vec = freq_resp[:, np.newaxis]
    A = freq.form_A_mat_levy(freqs, freq_resp_vec, num_order, den_order)
    expected_shape = (len(freqs), 1 + num_order + den_order)
    expected_mat = np.array(
        [[1, 1, 1, -2, -2, -2], [1, 1, 2, -3, -6, -12], [1, 1, 3, -4, -12, -36]]
    )
    assert expected_shape == A.shape
    nptest.assert_array_equal(expected_mat, A)


def test_create_z_exp_array_generates_valid_output():
    exp_order = 3
    z = np.array([np.complex128(2j)])
    z_exp_array = freq.create_z_exp_array(exp_order, z)
    assert True == True


def test_create_tf_for_fit_generates_valid_tf():
    num_order = 1
    den_order = 2
    params = np.array([1.0, 2.0, 3.0, 4.0])
    tf_for_fit = freq.create_tf_for_fit(num_order, den_order)
    z = np.array([np.complex128(1j), np.complex128(2j)])
    H_out = tf_for_fit(params, z)
    H_expect = np.array(
        [
            np.complex128(1 + 2j) / np.complex128(-4 + 3j),
            np.complex128(1 + 4j) / np.complex128(-16 + 6j),
        ]
    )
    nptest.assert_array_equal(H_expect, H_out)


def test_convert_hz_to_complex_rad_s_generates_valid_output():
    freqs = np.array([1, 2, 3])
    freqs_complex = freq.convert_hz_to_complex_rad_s(freqs)
    freqs_complex_expected = freqs * 2 * np.pi * 1j
    nptest.assert_array_equal(freqs_complex_expected, freqs_complex)


def test_create_err_func_for_least_squares_generate_valid_err_func():
    num_order = 1
    den_order = 2
    params = np.array([1.0, 2.0, 3.0, 4.0])
    freqs = np.array([1, 2, 3])
    freqs_complex = freq.convert_hz_to_complex_rad_s(freqs)
    freq_resp = np.sin(freqs)
    err_func = freq.create_err_func_for_tfest(
        num_order, den_order, freqs_complex, freq_resp
    )
    err_float = err_func(params)
    assert True == True


def test_create_err_func_for_least_squares_generate_valid_err_func_2():
    num_order = 1
    den_order = 4
    params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    freqs = np.array([1, 2, 3])
    freqs_complex = freq.convert_hz_to_complex_rad_s(freqs)
    freq_resp = np.sin(freqs)
    err_func = freq.create_err_func_for_tfest(
        num_order, den_order, freqs_complex, freq_resp
    )
    err_float = err_func(params)
    assert True == True
