import jax.numpy as jnp
from jax import jit
import time

@jit
def large_matrix_operations_1():
    x = jnp.dot(jnp.ones((10000, 10000)), jnp.ones((10000, 10000)))
    return x

@jit
def large_matrix_operations_2():
    x = jnp.dot(jnp.ones((10000, 10000)), jnp.ones((10000, 10000)))
    return x

result_1 = large_matrix_operations_1
result_2 = large_matrix_operations_2


# With block_until_ready
start_time = time.time()
result_1()  # Wait for the computation to complete
print("Time no block: ", time.time() - start_time)

# With block_until_ready
start_time = time.time()
result_2().block_until_ready() # Wait for the computation to complete
print("Time with block: ", time.time() - start_time)
