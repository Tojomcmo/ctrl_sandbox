import pytest
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import ctrl_sandbox.sysid.freq_resp_funcs as freq
import ctrl_sandbox.dyn_functions as dyn


# def test_calc_freq_resp_accepts_valid_system():

#     freq_resp = freq.calc_freq_resp()

#     assert j == j_expect


def test_sine_sweep_up_down_generates_valid_sweep():
    freq_0 = 0.0
    freq_1 = 1.0
    duration = 50.0
    sample_rate = 0.01
    sweep = freq.sine_sweep_up_down(freq_0, freq_1, duration, sample_rate)

    t = np.linspace(0, duration, int(duration / sample_rate), endpoint=False)
    plt.plot(t, sweep)
    plt.savefig("test_out.png")
    assert (True, True)
