#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:43:15 2022

@author: thomasmoriarty
"""

import numpy as np
from scipy.signal import welch, csd, chirp, get_window
from scipy.fft import fft, fftfreq
from scipy.optimize import least_squares, minimize
import math
from typing import Tuple, Callable, Union, Optional
import numpy.typing as npt
from functools import partial

import ctrl_sandbox.gen_typing as gt

# other functions

# get window type function

# get


def Calc_Psd(sig1, sig2, sampleFreq_Hz):

    # Signals must be the same length, of even length, and of same sample frequency
    sampleTime_s = 1 / sampleFreq_Hz
    sampleLen = len(sig1)

    # create ffts of the input signals
    sig1_fft = fft(sig1)
    sig2_fft = fft(sig2)

    # create conjugate instance of signal 2
    sig1_fft_conj = np.conj(sig1_fft)

    # create vector of fft frequencies referenced by the DFT, Truncated to half the length
    sig_freqs = fftfreq(sampleLen, sampleTime_s / 2 / np.pi)[: sampleLen // 2]

    fft_mult = sig1_fft_conj * sig2_fft

    # psd return in watts/rads/sample
    # 2/(sampleFreq * SampleLength) * (fft(x(n)) * fft*(x(n)))
    # sample frequency scaled by 2pi to convert Hz to radians
    # psd returned
    psd = 2.0 / sampleLen / (2 * np.pi * sampleFreq_Hz) * fft_mult[0 : sampleLen // 2]
    psd[0] = psd[0] / 2
    psd[-1] = psd[-1] / 2
    return psd, sig_freqs


def Calc_fr_from_fft(sig1, sig2, sampleFreq_Hz):

    # Signals must be the same length, of even length, and of same sample frequency
    sampleTime_s = 1 / sampleFreq_Hz
    sampleLen = len(sig1)

    # create ffts of the input signals
    sig1_fft = fft(sig1)
    sig2_fft = fft(sig2)

    # create vector of fft frequencies referenced by the DFT, Truncated to half the length
    sig_freqs = fftfreq(sampleLen, sampleTime_s / 2 / np.pi)[: sampleLen // 2]

    fr_fft = (sig2_fft[0 : sampleLen // 2]) / (sig1_fft[0 : sampleLen // 2])

    return fr_fft, sig_freqs


def calc_freq_resp(
    in_sig_vec: gt.npArr64,
    out_sig_vec: gt.npArr64,
    fs: float,
    nperseg: int,
) -> Tuple[gt.npArr64, gt.npCpx128]:
    """
    **Calculate frequency response from input signal to output signal** \n
    This function uses welch's method of windowing and segmentation to reduce noise
    with averaging power spectral densities. Uses half segment overlap and Hann
    windowing.
    -[in] in_sig_vec - Input signal as an array dim(n,)
    -[in] out_sig_vec - Output signal as an array dim(n,)
    -[in] fs - sample frequency in Hz(?) TODO check this
    -[in] nperseg - samples per segment for welch averaging
    -[out] freqs - array of frequencies expressed in output, Hz TODO check this
    -[out] freq_resp - frequency response, calculated as ratio of psds

    Resources:
    - Ljung - System identification (6.4) - ETFE
        -https://typeset.io/pdf/system-identification-theory-for-the-user-1copg8s5nw.pdf
    - https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem
    -

    """
    # calculate auto power spectral density of input using welch's method
    # freqs, ipsd = welch(in_sig_vec, fs, nperseg=nperseg)
    # calculate cross power spectral density of input and output using welch's method
    freqs, ipsd = csd(in_sig_vec, in_sig_vec, fs, nperseg=nperseg)
    _, cpsd = csd(in_sig_vec, out_sig_vec, fs, nperseg=nperseg)
    # divide cpsd by apsd to calculate frequency response
    return freqs, (cpsd / ipsd)


def sine_sweep_up_down(
    freq_0: float,
    freq_1: float,
    amplitude: float,
    duration: float,
    ts: float,
    type: str,
) -> gt.npArr64:
    half_duration = duration / 2
    t = np.linspace(0, half_duration, int(half_duration / ts), endpoint=False)
    sweep_up = chirp(
        t,
        f0=freq_0,
        f1=freq_1,
        t1=half_duration,
        method=type,
        phi=-90.0,  # type:ignore #TODO post error in scipy about chirp typechecking
    )
    sweep_down = np.flip(-sweep_up)
    sweep = np.concatenate((sweep_up, sweep_down))
    return amplitude * sweep


def calculate_mag_and_phase_from_complex(
    fr_complex: gt.npCpx128,
) -> Tuple[gt.npArr64, gt.npArr64]:

    return (20 * np.log10(np.abs(fr_complex)), np.angle(fr_complex))


def fit_lin_tf_to_fr_data(
    freqs: gt.npArr64,
    freq_resp: gt.npCpx128,
    num_order: int,
    den_order: int,
    params_init: gt.npArr64,
    bounds: Optional[gt.npArr64] = None,
) -> Tuple[gt.npArr64, gt.npArr64]:
    """
    **Fit a transfer function of type num_order / den_order to supplied
    frequency response data.** \n

    - [in] freqs       : Array dim(k,) of frequencies matched with frequency response
    values in Hz TODO check this
    - [in] freq_resp   : Array dim(k,) of corresponding frequency response, complex values
    - [in] num_order   : int order of fitted transfer function numerator, must be
    positive or zero
    - [in] den_order   : int order of fitted transfer function denominator, must be
    positive or zero
    - [in] params_init : initial list of tf parameter guesses
    - [out] num_coeffs : coefficient values of numerator, array dim(num_order, )
    - [out] den_coeffs : coefficient values of denominator, array dim(den_order, ) \n

    algorithm: Levy Method\n

    - uses least squares fitting to match response data to transfer function response
    at each frequency
        - N(s) = (a0 + a1s + a2s^2 ... + ans^n)
        - D(s) = (1  + b1s + b2s^2 ... + bms^m)
        - e(s) = FR(s) - N(s)/D(s)
        - e(s)D(s) = FR(s)D(s) - N(s) -> 0 as fit improves
        - form as Ax=b system for k data samples
            - Ak = [1 sk ... sk^n -skF(sk) ... -sk^mF(sk)]
            - x  = [a0 a1 ... an b1 b2 ... bm].T
            - b  = [F(s1) ... F(sk)].T
        - least squares solution using pseudo inverse
            - x = (A.T @ A).inv @ A.T b

    """
    freqs_complex: gt.npCpx128 = convert_hz_to_complex_rad_s(freqs)
    err_func = create_err_func_for_tfest(num_order, den_order, freqs_complex, freq_resp)
    if not bounds:
        bounds = np.tile([0, np.inf], (len(params_init), 1))
    result = minimize(err_func, params_init, bounds=bounds)
    x: gt.npArr64 = result.x
    return split_params_to_num_den(x, num_order)


def create_z_transform_for_freqs(freqs, ts):
    return np.exp(freqs * 1j * ts)


def convert_hz_to_complex_rad_s(
    freq: Union[np.float64, gt.npArr64]
) -> Union[np.complex128, gt.npCpx128]:
    return 2 * np.pi * freq * 1j


def transfer_function_model(
    num_coeffs: gt.npArr64, den_coeffs: gt.npArr64, s: np.complex128
) -> np.complex128:
    num = np.complex128(np.polyval(num_coeffs[::-1], s))
    den = np.complex128(np.polyval(den_coeffs[::-1], s))
    return num / den


def split_params_to_num_den(
    params: gt.npArr64, num_order: int
) -> Tuple[gt.npArr64, gt.npArr64]:
    """
    params order as 1D vecto:
        - numerator coefficients: low to high
        - denominator coefficients: low to high [excluding highest of coefficient 1]
    output:
        - numerator coefficients as 1D vector: low to high
        - denominator coefficients as 1D vector: low to high including highest 1)
    """
    return params[: num_order + 1], np.append(params[num_order + 1 :], [1])


def create_err_func_for_tfest(
    num_order: int,
    den_order: int,
    freqs: gt.npCpx128,
    freq_resp: gt.npCpx128,
) -> Callable[[gt.npArr64], np.float64]:

    def err_func(params: gt.npArr64) -> np.float64:
        num_coeffs, den_coeffs = split_params_to_num_den(params, num_order)
        tf_curried = lambda params: transfer_function_model(
            num_coeffs, den_coeffs, params
        )

        H_model: gt.npCpx128 = np.array(list(map(tf_curried, freqs)))
        error: gt.npArr64 = np.abs(H_model - freq_resp)
        error_squared: gt.npArr64 = np.square(error)
        sum_squared_error: np.float64 = np.sum(error_squared)
        return sum_squared_error

    return err_func


def compute_tf_response_at_freqs_hz(tf, freqs):
    freqs_complex_rad_s = convert_hz_to_complex_rad_s(freqs)
    return np.array(list(map(tf, freqs_complex_rad_s)))


#######################################


def create_z_exp_array(exp_order: int, z: gt.npCpx128) -> gt.npCpx128:
    return z[:, np.newaxis] ** np.arange(exp_order + 1)


def fit_lin_tf_to_fr_data_levy(
    freqs: gt.npArr64, freq_resp: gt.npArr64, num_order: int, den_order: int
) -> Tuple[gt.npArr64, gt.npArr64]:
    """
    **Fit a transfer function of type num_order / den_order to supplied
    frequency response data.** \n

    - [in] freqs       : Array dim(k,) of frequencies matched with frequency response
    values in Hz TODO check this
    - [in] freq_resp   : Array dim(k,) of frequency response, complex values
    - [in] num_order   : int order of fitted transfer function numerator, must be
    positive or zero
    - [in] den_order   : int order of fitted transfer function denominator, must be
    positive or zero
    - [out] num_coeffs : coefficient values of numerator, array dim(num_order, )
    - [out] den_coeffs : coefficient values of denominator, array dim(den_order, ) \n

    algorithm: Levy Method\n

    - uses least squares fitting to match response data to transfer function response
    at each frequency
        - N(s) = (a0 + a1s + a2s^2 ... + ans^n)
        - D(s) = (1  + b1s + b2s^2 ... + bms^m)
        - e(s) = FR(s) - N(s)/D(s)
        - e(s)D(s) = FR(s)D(s) - N(s) -> 0 as fit improves
        - form as Ax=b system for k data samples
            - Ak = [1 sk ... sk^n -skF(sk) ... -sk^mF(sk)]
            - x  = [a0 a1 ... an b1 b2 ... bm].T
            - b  = [F(s1) ... F(sk)].T
        - least squares solution using pseudo inverse
            - x = (A.T @ A).inv @ A.T b

    """
    freqs_complex = freqs * 1j
    freq_resp_vec = freq_resp[:, np.newaxis]
    A = form_A_mat_levy(freqs_complex, freq_resp_vec, num_order, den_order)
    x = np.linalg.inv(A.T @ A) @ A.T @ freq_resp_vec
    return x[: num_order + 1], x[num_order + 1 :]


def form_A_mat_levy(
    freqs: gt.npArr64, freq_resp_vec: gt.npArr64, num_order: int, den_order: int
) -> gt.npArr64:
    if num_order > den_order:
        n = num_order
    else:
        n = den_order
    column_indices = np.arange(n)
    freq_powers_mat = freqs[:, np.newaxis] ** column_indices
    A_1 = np.ones((len(freqs), 1))
    A_2 = freq_powers_mat[:, :num_order]
    A_3 = -(freq_powers_mat[:, :den_order] * freq_resp_vec)
    A = np.concatenate((A_1, A_2, A_3), axis=1)
    return A


def create_tf_for_fit(
    num_order: int, den_order: int
) -> Callable[[gt.npArr64, gt.npCpx128], gt.npCpx128]:
    if num_order > den_order:
        exp_order = num_order
    else:
        exp_order = den_order

    def tf_for_fit(params: gt.npArr64, z: gt.npCpx128) -> gt.npCpx128:
        num_coeffs = params[: num_order + 1]
        den_coeffs = params[num_order + 1 :]
        z_exp_array = create_z_exp_array(exp_order, z)
        num_transfer = z_exp_array[:, : num_order + 1] * num_coeffs
        den_transfer = z_exp_array[:, 1 : den_order + 1] * den_coeffs
        H = np.sum(num_transfer, axis=1) / np.sum(den_transfer, axis=1)
        return H

    return tf_for_fit
