#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:43:15 2022

@author: thomasmoriarty
"""

import numpy as np
from scipy.signal import welch
from scipy.signal import get_window
from scipy.fft import fft, fftfreq
import math


# other functions

# get window type function

# get 

def Calc_Psd(sig1, sig2, sampleFreq_Hz):
    
    # Signals must be the same length, of even length, and of same sample frequency
    sampleTime_s  = 1 / sampleFreq_Hz
    sampleLen     = len(sig1)
    
    # create ffts of the input signals
    sig1_fft      = fft(sig1)
    sig2_fft      = fft(sig2)
    
    #create conjugate instance of signal 2
    sig1_fft_conj = np.conj(sig1_fft)
    
    #create vector of fft frequencies referenced by the DFT, Truncated to half the length
    sig_freqs = fftfreq(sampleLen, sampleTime_s/2/np.pi)[:sampleLen//2]
   
    fft_mult =  sig1_fft_conj * sig2_fft
    
    # psd return in watts/rads/sample
    # 2/(sampleFreq * SampleLength) * (fft(x(n)) * fft*(x(n)))
    # sample frequency scaled by 2pi to convert Hz to radians
    # psd returned 
    psd      =  2.0/sampleLen/(2 * np.pi * sampleFreq_Hz) * fft_mult[0:sampleLen//2]
    psd[ 0]  = psd[ 0]/2
    psd[-1]  = psd[-1]/2
    return psd, sig_freqs

def Calc_fr_from_fft(sig1, sig2, sampleFreq_Hz):
    
    # Signals must be the same length, of even length, and of same sample frequency
    sampleTime_s  = 1 / sampleFreq_Hz
    sampleLen     = len(sig1)
    
    # create ffts of the input signals
    sig1_fft      = fft(sig1)
    sig2_fft      = fft(sig2)
    
    #create vector of fft frequencies referenced by the DFT, Truncated to half the length
    sig_freqs = fftfreq(sampleLen, sampleTime_s/2/np.pi)[:sampleLen//2]
    
    fr_fft   = (sig2_fft[0:sampleLen//2]) / (sig1_fft[0:sampleLen//2])
 
    return fr_fft, sig_freqs


def Calc_Welch_Psd(sig1, sig2, sampleFreq_Hz, freqMin_Hz, ratioOverlap):    

    # truncate signals to same length and even value
    lenSignals_samp = 10;

    # define window length from minimum frequency and sample frequency:
    # the lowest frequency an FFT can sample is a period twice the duration of the window
    lenWindow_samp = math.floor(1 / (2 * freqMin_Hz) * sampleFreq_Hz)
    
    # define step size from the overlap ratio (exp. 1/2)
    lenStep_samp   = lenWindow_samp * ratioOverlap 
    
    # define number of windows from step size, window length, and signal lengths
    numWindows = math.floor((lenSignals_samp - lenWindow_samp) / lenStep_samp)
    
    # check for well-conditioned setup (enough windows etc.)
    
    # select which window form to use (exp. 'Hann')
    windowFunc = get_window('hann', lenWindow_samp)
    
    # for each window instance, calculate window psd: Ir, store result
   # for k <= numWindows:
        #create local signal vectors
   #     sig1_k = 
        
        # multiply windowed signals by window function
        
        # calculate fft of each windowed signal
   #     yf = fft(y)
   #     xf = fftfreq(N, T)[:N//2]
        # take the complex conjugate of FFT of signal 1???
        # Multiply FFT signals together and normalize by the sample frequency
            # this process is analogos to taking the FFT of the correlation of the two signals wrt eachother
        # Take the average of all window PSDs
    
    return windowFunc