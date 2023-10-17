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

def plot_2d_state_quiver(figure, axes, x_seq, legend_name= 'sequence', xlabel="state 1", ylabel="state 2", title="state plot", color='b.', width = 0.0015):
    """
    Plot XY data using Matplotlib from JAX arrays.
    # Convert JAX arrays to NumPy arrays for plotting
    # x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
    # y_np = np.array(y) if isinstance(y, jnp.ndarray) else y
    """
    # start array as column vector
    x_seq_array = (x_seq[0]).reshape(-1,1)
    # append each state as new columns
    for idx in range(len(x_seq)-1):
        x_seq_k     = (x_seq[idx+1]).reshape(-1,1)
        x_seq_array = np.append(x_seq_array, x_seq_k, axis=1)
    # break out each row into respective state arrays    
    x_data =  x_seq_array[0]    
    y_data =  x_seq_array[1]
    # create quiver vector
    x_start = x_data[:-1] + (0.1)*(x_data[1:]-x_data[:-1])
    y_start = y_data[:-1] + (0.1)*(y_data[1:]-y_data[:-1])
    del_x   =  (0.8) * (x_data[1:]-x_data[:-1])
    del_y   =  (0.8) * (y_data[1:]-y_data[:-1])
    #axes[axis_num].plot(x_seq_array[0], x_seq_array[1], color,linestyle='--', label=legend_name)
    plt.figure(figure) 
    axes.quiver(x_start, y_start, del_x, del_y, color=color, 
                scale_units='xy', angles='xy', scale=1, width = width, headwidth=4, headlength = 6, headaxislength = 5)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True)

def plot_2d_state_dot(figure, axes, x_seq, legend_name= 'sequence', xlabel="state 1", ylabel="state 2", title="state plot", color='b.'):
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
        x_seq_array = np.append(x_seq_array, x_seq_k, axis=1)
    plt.figure(figure) 
    axes.plot(x_seq_array[0], x_seq_array[1], color=color, label=legend_name, linestyle = 'None', marker = 'o', markersize = 3)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
    axes.grid(True)

def plot_compare_state_sequences(plot_func, figure, axes, axis_num, x_seqs, x_names, x_styles, xlabel="state 1", ylabel="state 2", title="state plot"):
    for idx in range(len(x_seqs)):
        plot_func(figure, axes, axis_num, x_seqs[idx], x_names[idx], xlabel, ylabel, title, color = x_styles[idx])
    plt.legend()
    return

def plot_compare_state_sequences_quiver_dot(figure, axes, x_seqs, x_names, x_styles_quiver,x_styles_dot, quiverwidth, xlabel="state 1", ylabel="state 2", title="state plot"):
    for idx in range(len(x_seqs)):
        plot_2d_state_quiver(figure, axes, x_seqs[idx], x_names[idx], xlabel, ylabel, title, color = x_styles_quiver[idx], width = quiverwidth[idx])
        plot_2d_state_dot(figure, axes, x_seqs[idx], x_names[idx], xlabel, ylabel, title, color = x_styles_dot[idx])
    plt.legend()
    return

def plot_x_y_sequences(figure, axes, x_seq, y_seq, xlabel="state 1", ylabel="state 2", datalabel = 'data', title="state plot", color = 'b.'):
    """
    Plot XY data using Matplotlib from JAX arrays.
    # Convert JAX arrays to NumPy arrays for plotting
    # x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
    # y_np = np.array(y) if isinstance(y, jnp.ndarray) else y
    """
    # Create the plot
    plt.figure(figure) 
    x_seq.reshape(-1)
    y_seq.reshape(-1)
    axes.plot(x_seq, y_seq, color,linestyle='-',label=datalabel)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
    axes.grid(True)