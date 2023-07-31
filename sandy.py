from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plot on the first subplot
axes[0].plot(x, y1, label='sin(x)')
axes[0].set_title('Plot 1')
axes[0].legend()

# Plot on the second subplot
axes[1].plot(x, y2, 'r', label='cos(x)')
axes[1].set_title('Plot 2')
axes[1].legend()

# Adjust layout to prevent overlapping of titles
plt.tight_layout()

# Show the figure with subplots
plt.show()
