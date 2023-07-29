from jax import numpy as jnp
import numpy as np



a = np.array([[1,2],[3,4]])

b_seq = np.ones([5,2,1])
b_seq[3,0] = 2.0

c = a @ b_seq[3]

print(c)