import jax.numpy as jnp
from jax import lax

# Define the dynamics function
def dynamics(state, t):
    # Simple linear dynamics: x_next = Ax + Bu
    # Assume A and B are simple for demonstration
    A = jnp.array([[1.0, t], [0, 1.0]])
    B = jnp.array([0.1, 0.1])
    u = jnp.array([1.0, 1.0])  # Constant control input for simplicity
    return jnp.dot(A, state) + B * u

# Propagate function using jax.lax.scan
def propagate(initial_state, timesteps):
    def step(state, t):
        # The new state becomes the old state in the next iteration
        new_state = dynamics(state, t)
        return new_state, new_state  # Return new state and the same as output for history tracking

    # Initial state is the first state in the sequence
    final_state, all_states = lax.scan(step, initial_state, timesteps)
    return final_state, all_states

# Example usage
initial_state = jnp.array([1.0, 0.0])  # Initial state
timesteps = jnp.linspace(0.0, 1.0, num=10)  # Time steps from 0 to 1, 10 steps

final_state, all_states = propagate(initial_state, timesteps)

print("Final state:", final_state)
print("All states:\n", all_states)
