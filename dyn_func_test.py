import dyn_functions as dyn
import ilqr_funcs as ilqr
from jax import numpy as jnp



time_step = 0.1


state = jnp.array([0.1, 0.1])
control = jnp.array([0.1])
C = jnp.array([[1,0]])
D = jnp.array([[0]])

pend_params = dict(l = 1.0, g = 2.0, b = 1.0)
pend_dyn_with_params = (lambda x, u: dyn.pend_dyn_nl(x, u, **pend_params))
pend_dyn = pend_dyn_with_params(state, control)

A_lin, B_lin = ilqr.linearize_dynamics(pend_dyn_with_params, state, control)

print(B_lin.shape)

ss_pend_lin_continuous = ilqr.state_space(A_lin, B_lin, C, D)

ss_pend_lin_discrete, fe = ilqr.discretize_state_space(ss_pend_lin_continuous, time_step, method='zohCombined')

print(ss_pend_lin_continuous.A)
print(ss_pend_lin_continuous.B)

print(ss_pend_lin_discrete.A)
print(ss_pend_lin_discrete.B)
print('is error :', fe)
