from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Union

# ---------- utility functions ----------#

def is_pos_def(x:jnp.ndarray):
    eigs = jnp.linalg.eigvals(x)
    return jnp.all(eigs > 0)

def reqularize_mat(mat:jnp.ndarray, ro:float):
    return mat + ro * jnp.eye(mat.shape[0], mat.shape[1])

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

def check_inf_or_nan_array(arr:Union[np.ndarray,jnp.ndarray]) -> bool:
    if jnp.any(jnp.isinf(arr) | jnp.isnan(arr)):
        return True
    else:
        return False