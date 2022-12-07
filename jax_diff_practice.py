import jax
import jax.numpy as jnp

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

print(f(primal_point))
print(f(tanget_point))
print(f(primal_point) + (jac_f @ (tanget_point - primal_point)))
