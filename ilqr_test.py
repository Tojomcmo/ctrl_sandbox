import scipy
import jax.numpy as jnp
import ilqr_funcs as ilqr
from pprint import PrettyPrinter as pp


#controller_obj = ilqr.ilqr_controller(2, 1, 1, 1, 1, 1, 1, 1, 1)

#ilqr.calculate_backwards_pass()

Ac = jnp.array([[1,1,1],[1,1,1],[0,0,1]])
Bc = jnp.array([[1,1],[0,1],[0,0]])
Cc = jnp.array([[1,0,1]])
Dc = jnp.array([[0,0]])
time_step = 0.1

Ad_Euler,Bd_Euler,Cd_Euler,Dd_Euler = ilqr.discretize_state_space(Ac,Bc,Cc,Dc,time_step,method = 'Euler')

Ad_zohC,Bd_zohC,Cd_zohC,Dd_zohC = ilqr.discretize_state_space(Ac,Bc,Cc,Dc,time_step,method = 'zohCombined')

Ad_zoh,Bd_zoh,Cd_zoh,Dd_zoh = ilqr.discretize_state_space(Ac,Bc,Cc,Dc,time_step,method = 'zoh')

print('Ad_Euler = ')
print(Ad_Euler)
print('Bd_Euler = ')
print(Bd_Euler)

print('Ad_zohC = ')
print(Ad_zohC)
print('Bd_zohC = ')
print(Bd_zohC)

print('Ad_zoh = ')
print(Ad_zoh)
print('Bd_zoh = ')
print(Bd_zoh)