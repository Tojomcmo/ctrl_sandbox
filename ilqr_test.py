import scipy
import jax.numpy as jnp
import ilqr_funcs as ilqr
import time
from pprint import PrettyPrinter as pp



A = jnp.array([[1,1,1],[0,1,1],[0,0,1]])
B = jnp.array([[1,1],[0,1],[0,0]])
C = jnp.array([[1,0,1]])
D = jnp.array([[0,0]])
ss_c = ilqr.state_space(A,B,C,D)

time_step = 0.1

ss_Euler, func_error_Euler = ilqr.discretize_state_space(ss_c,time_step,method = 'Euler')

ss_zohC, func_error_zohC = ilqr.discretize_state_space(ss_c,time_step,method = 'zohCombined')

ss_zoh, func_error_zoh = ilqr.discretize_state_space(ss_c,time_step,method = 'zoh')

print('Ad_Euler = ')
print(ss_Euler.A)
print('Bd_Euler = ')
print(ss_Euler.B)
print('is Euler error :', func_error_Euler)

print('Ad_zohC = ')
print(ss_zohC.A)
print('Bd_zohC = ')
print(ss_zohC.B)
print('is zohC error :', func_error_zohC)

print('Ad_zoh = ')
print(ss_zoh.A)
print('Bd_zoh = ')
print(ss_zoh.B)
print('is zoh error :', func_error_zoh)


