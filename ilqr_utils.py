from jax import numpy as jnp
import numpy as np

# ---------- utility functions ----------#

def is_pos_def(x):
    eigs = np.linalg.eigvals(x)
    return np.all(eigs > 0)

def reqularize_mat(mat, ro):
    return mat + ro * np.eye(mat.shape[0], mat.shape[1])

def calc_and_shape_array_diff(current_array,desired_array,shape=None):  
    if shape == 'row':
        current_minus_des_array = array_to_row(current_array - desired_array)
    elif shape == 'col':
        current_minus_des_array = array_to_col(current_array - desired_array)
    else:
        current_minus_des_array = current_array - desired_array
    return current_minus_des_array

def array_to_col(array_1D):
    col_vec = array_1D.reshape(-1,1)
    return col_vec

def array_to_row(array_1D):
    row_vec = array_1D.reshape(1,-1)
    return row_vec

def get_vec_from_seq_as_col(vec_seq, k):
    col_vec = vec_seq[k].reshape(-1,1)
    return col_vec
