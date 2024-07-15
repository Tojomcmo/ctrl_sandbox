import jax.numpy as jnp

import ctrl_sandbox.sysid.freq_resp_funcs as freq
import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.integrate_funcs as integrate
import ctrl_sandbox.simulate_funcs as sim


ts = 0.01
freq_0 = 0.0
freq_1 = 20.0
sweep_duration = 100.0

pend_dyn_obj = dyn.single_pm_pend_dyn(g=9.8, b=0.1, l=1.0)
disc_dyn_func = lambda x, u: integrate.step_rk4(pend_dyn_obj.cont_dyn_func, ts, x, u)

sweep_u_seq = freq.sine_sweep_up_down(freq_0, freq_1, duration=sweep_duration, ts=ts)

x_init = jnp.array([0, 0])
x_seq = jnp.zeros((len(sweep_u_seq), 2))
