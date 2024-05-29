from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Union

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
