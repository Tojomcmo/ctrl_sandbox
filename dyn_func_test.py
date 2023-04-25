from jax import numpy as jnp
import dyn_functions as dyn
import ilqr_funcs as ilqr


time_step = 0.1
state = jnp.array([0.1, 0.1])
control = jnp.array([0.1])
time_dyn = 0
C = jnp.array([[1,0]])
D = jnp.array([[0]])

pend_params = dict(l = 1.0, g = 2.0, b = 1.0)
######################## Discretization

# pend_dyn_with_params = (lambda x, u: dyn.pend_dyn_nl(time_dyn, x, u, **pend_params))
# pend_dyn = pend_dyn_with_params(state, control)

# A_lin, B_lin = ilqr.linearize_dynamics(pend_dyn_with_params, state, control)
# ss_pend_lin_continuous = ilqr.state_space(A_lin, B_lin, C, D)
# ss_pend_lin_discrete, fe = ilqr.discretize_state_space(ss_pend_lin_continuous, time_step, method='zohCombined')

# print(ss_pend_lin_continuous.A)
# print(ss_pend_lin_continuous.B)

# print(ss_pend_lin_discrete.A)
# print(ss_pend_lin_discrete.B)
# print('is error :', fe)

########################
# time_step   = 0.1
# num_steps   = 20
# control_seq = jnp.ones((num_steps, 1))
# state_init  = jnp.array([[0.1, 0.1]])
# pend_dyn_with_params = (lambda t, x, u: dyn.pend_dyn_nl(t, x, u, **pend_params))
# state_seq, time_seq  = ilqr.simulate_forward_dynamics(pend_dyn_with_params,control_seq,state_init,time_step)
# A_lin_array, B_lin_array = ilqr.calculate_linearized_state_space_seq(pend_dyn_with_params, state_seq, control_seq, time_step)

########################

num_steps   = 20
control_seq = jnp.ones((num_steps, 1))
state_init  = jnp.array([[0.1, 0.1]])
kwparams    = {'sim_method':'solve_ivp_zoh', 'c2d_method':'zohCombined'}
pend_dyn_with_params = (lambda t, x, u: dyn.pend_dyn_nl(t, x, u, **pend_params))
Ad_seq, Bd_seq, time_seq = ilqr.calculate_forward_rollout(pend_dyn_with_params, state_init, control_seq, time_step, **kwparams)
print(Ad_seq[-1])
print(Bd_seq[-1])
print(time_seq[-1])