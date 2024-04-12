from jax import numpy as jnp
import numpy as np
import numpy.typing as npt

# ---------- utility functions ----------#

def is_pos_def(x:npt.NDArray):
    eigs = np.linalg.eigvals(x)
    return np.all(eigs > 0)

def reqularize_mat(mat:npt.NDArray, ro:float):
    return mat + ro * np.eye(mat.shape[0], mat.shape[1])

def calc_and_shape_array_diff(current_array:npt.NDArray[np.float64],desired_array:npt.NDArray[np.float64],shape=None) -> npt.NDArray[np.float64]:  
    if shape == 'row':
        
        current_minus_des_array = array_to_row(current_array) - array_to_row(desired_array)
    elif shape == 'col':
        current_minus_des_array = array_to_col(current_array) - array_to_col(desired_array)
    else:
        current_minus_des_array = current_array - desired_array
    return current_minus_des_array

def array_to_col(array_1D:npt.NDArray):
    col_vec = array_1D.reshape(-1,1)
    return col_vec

def array_to_row(array_1D:npt.NDArray):
    row_vec = array_1D.reshape(1,-1)
    return row_vec

def get_vec_from_seq_as_col(vec_seq:npt.NDArray, k:int):
    col_vec = vec_seq[k].reshape(-1,1)
    return col_vec
