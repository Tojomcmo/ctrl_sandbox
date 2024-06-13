from jax import numpy as jnp
import jax
import numpy as np
import numpy.typing as npt
from typing import Union, Callable

# ---------- utility functions ----------#


def is_pos_def(x: jnp.ndarray):
    eigs = jnp.linalg.eigvals(x)
    return jnp.all(eigs > 0)


def reqularize_mat(mat: jnp.ndarray, ro: float):
    return mat + ro * jnp.eye(mat.shape[0], mat.shape[1])


def check_inf_or_nan_array(arr: Union[np.ndarray, jnp.ndarray]) -> bool:
    if jnp.any(jnp.isinf(arr) | jnp.isnan(arr)):
        return True
    else:
        return False


def calculate_matrix_power_similarity_transform(
    A: jnp.ndarray, exponent: float
) -> jnp.ndarray:
    evals, evecs = jnp.linalg.eig(A)
    D = jnp.diag(evals)
    evecs_inv = jnp.linalg.inv(evecs)
    A_to_exponent = evecs @ D**exponent @ evecs_inv
    return A_to_exponent


def lin_interp_pair(x: jnp.ndarray, y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    return alpha * x + (1 - alpha) * y


def create_lin_interp_seq_func() -> Callable[[jnp.ndarray, float], jnp.ndarray]:
    """
    **Constructor function for linear interpolation of sequence pairs, taken
    at the float value between each pair**
    """
    interp_func = jax.vmap(lin_interp_pair, in_axes=(0, 0, None))

    def lin_interp_seq_func(seq: jnp.ndarray, alpha: float) -> jnp.ndarray:
        return interp_func(seq[:-1], seq[1:], alpha)

    return lin_interp_seq_func


def create_lin_interp_seq_mid_point_func() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    **Constructor function for linear interpolation of sequence pairs, taken
    at the midpoint between each pair**
    """
    interp_func = jax.vmap(lin_interp_pair, in_axes=(0, 0, None))

    def lin_interp_seq_func_mid_point(seq: jnp.ndarray) -> jnp.ndarray:
        return interp_func(seq[:-1], seq[1:], 0.5)

    return lin_interp_seq_func_mid_point


def interleave_jax_arrays(odd_jax: jnp.ndarray, even_jax: jnp.ndarray) -> jnp.ndarray:
    array_out = jnp.zeros(len(odd_jax) + len(even_jax))
    array_out = array_out.at[0:2].set(even_jax)
    array_out = array_out.at[1:2].set(even_jax)
    return array_out
