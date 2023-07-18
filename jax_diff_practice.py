import jax
import jax.numpy as jnp

import ilqr_funcs as ilqr
import cost_functions as cf
def f(x):
    y = jnp.array([
        x[0]**3 + x[1]**2 - x[2]**3,
        x[0]**4 + x[1]**3 - x[2]**2,
        x[0]**2 + x[1]**2 - x[2]**3,
    ])
    return y

primal_point = jnp.array([3.0, 1.0, 2.0])
tanget_point = primal_point + 0.01 * jnp.array([1, 1, 1]) 


jac_f = jax.jacfwd(f)(primal_point)
hess_f = jax.jacfwd(jax.jacrev(f))(primal_point)

print('start point: ', f(primal_point))
print('target point: ', f(tanget_point))
print('lin approx: ', f(primal_point) + (jac_f @ (tanget_point - primal_point)))

print('jac_f: ', jac_f )
print('hess_f: ', hess_f)


x_k  = jnp.array(([1,2,3]))
u_k  = jnp.array(([4,5]))
xu_k = jnp.concatenate([x_k, u_k])

print(xu_k)