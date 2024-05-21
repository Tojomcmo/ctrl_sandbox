import jax.numpy as jnp
from jax import jit, lax, ops
import time

# Dynamics function
def dynamics(state, t):
    A = jnp.array([[0.99, 0.01], [-0.01, 0.99]])
    return jnp.dot(A, state)

# Scan implementation
def propagate_scan(initial_state, timesteps):
    def step(state, t):
        state_next = dynamics(state, t)
        return state_next, state_next
    final_state, state_history = lax.scan(step, initial_state, timesteps)
    return state_history

# While-loop implementation
def propagate_loop(initial_state:jnp.ndarray, timesteps):
    state = initial_state
    state_history:jnp.ndarray = jnp.zeros((len(timesteps+1),len(initial_state)))
    state_history.at[0].set(initial_state)
    for idx, t in enumerate(timesteps):
        state = dynamics(state, t)
        state_history.at[idx+1].set(state)
    return state_history

# JIT compile both functions
propagate_scan_jit = jit(propagate_scan)
propagate_loop_jit = jit(propagate_loop)

# Initialize states and timesteps
initial_state = jnp.array([1.0, 0.0])
timesteps = jnp.linspace(0.0, 1.0, num=1000)

# Timing the functions
start_time = time.time()
propagate_scan_jit(initial_state, timesteps).block_until_ready()
print("Scan time:", time.time() - start_time)

start_time = time.time()
propagate_loop_jit(initial_state, timesteps).block_until_ready()
print("Loop time:", time.time() - start_time)
