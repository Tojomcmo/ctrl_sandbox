from jax import numpy as jnp

# ---------- utility functions ----------#

def is_pos_def(x):
    eigs = jnp.linalg.eigvals(x)
    return jnp.all(eigs > 0)

def reqularize_mat(mat, ro):
    return mat + ro * jnp.eye(mat.shape[0], mat.shape[1])

def calculate_current_minus_des_array(current_array,desired_array):  
    current_minus_des_array = current_array - desired_array
    return current_minus_des_array

def vec_1D_array_to_col(array_1D):
    col_vec = array_1D.reshape(-1,1)
    return col_vec

def vec_1D_array_to_row(array_1D):
    row_vec = array_1D.reshape(1,-1)
    return row_vec

def get_vec_from_seq_as_col(vec_seq, k):
    col_vec = vec_seq[k].reshape(-1,1)
    return col_vec