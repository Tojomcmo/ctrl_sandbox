#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:43:15 2022

@author: thomasmoriarty
"""

import numpy as np
from scipy.signal import welch, csd, chirp, get_window
from scipy.fft import fft, fftfreq
import math
from typing import Tuple
import numpy.typing as npt

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
) -> Tuple[gt.npArr64, gt.npArr64]:
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
    freqs, ipsd = welch(in_sig_vec, fs, nperseg=nperseg)
    # calculate cross power spectral density of input and output using welch's method
    _, cpsd = csd(in_sig_vec, out_sig_vec, fs, nperseg=nperseg)
    # divide cpsd by apsd to calculate frequency response
    return freqs, (cpsd / ipsd)


def sine_sweep_up_down(
    freq_0: float, freq_1: float, duration: float, ts: float
) -> gt.npArr64:
    t = np.linspace(0, duration, int(duration / ts), endpoint=False)
    half_duration = duration / 2
    sweep_up = chirp(
        t[: int(len(t) / 2)],
        f0=freq_0,
        f1=freq_1,
        t1=half_duration,
        method="linear",
        phi=-90.0,  # type:ignore #TODO post error in scipy about chirp typechecking
    )
    sweep_down = chirp(
        t[: int(len(t) / 2)],
        f0=freq_1,
        f1=freq_0,
        t1=half_duration,
        method="linear",
        phi=90.0,  # type:ignore #TODO post error in scipy about chirp typechecking
    )
    sweep = np.concatenate((sweep_up, sweep_down))
    return sweep
