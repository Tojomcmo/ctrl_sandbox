import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_2d_state_scatter_sequences(x_seq, xlabel="state 1", ylabel="state 2", title="state plot"):
    """
    Plot XY data using Matplotlib from JAX arrays.
    # Convert JAX arrays to NumPy arrays for plotting
    # x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
    # y_np = np.array(y) if isinstance(y, jnp.ndarray) else y
    """
    # Create the plot
    plt.figure()
    for x_k in x_seq:
        state_1 = x_k[0]
        state_2 = x_k[1]
        plt.plot(state_1, state_2, 'b.',linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_state_batch_sequences(x_seq, legend_name= 'sequence', xlabel="state 1", ylabel="state 2", title="state plot", color='b.', fig_num = 1):
    """
    Plot XY data using Matplotlib from JAX arrays.
    # Convert JAX arrays to NumPy arrays for plotting
    # x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
    # y_np = np.array(y) if isinstance(y, jnp.ndarray) else y
    """
    # Create the plot
    x_seq_array = (x_seq[0]).reshape(-1,1)
    for idx in range(len(x_seq)-1):
        x_seq_k     = (x_seq[idx+1]).reshape(-1,1)
        x_seq_array = jnp.append(x_seq_array, x_seq_k, axis=1)
    plt.figure(fig_num)    
    plt.plot(x_seq_array[0], x_seq_array[1], color,linestyle='--', label=legend_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)


def plot_compare_state_sequences(x_seqs, x_names, x_styles, xlabel="state 1", ylabel="state 2", title="state plot", fig_num=1):
    for idx in range(len(x_seqs)):
        plot_2d_state_batch_sequences(x_seqs[idx], x_names[idx], xlabel, ylabel, title,color = x_styles[idx], fig_num=fig_num)
    plt.legend()
    return