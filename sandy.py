from jax import numpy as jnp
import numpy as np




a = np.array([[1,2,3,4,5],[0,0,0,0,0]])

a_jax = jnp.asarray(a)
x = jnp.array([[1.,1.,1.,1.,1.]])
k = 0

a_k = (a[k]).reshape(-1,1)

print('a_k: ', a[k][:].transpose())
print('a_k: ', a_k)
print('a_jax_k: ', (a_jax[k]).reshape(1,-1))

x_t = x.transpose()
print(x_t)
print('a mult x: ',  a @ x_t )
print('a_jax mult x: ',  a_jax @ x_t )
print('a mult a_jax_k: ', a @ (a_jax[k]).reshape(-1,1))