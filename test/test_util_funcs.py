import pytest
import jax.numpy as jnp
import numpy as np
import util_funcs as util


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]]), True),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), True),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), False),  # Example of a non-pos-def matrix
    ],
)
def test_is_pos_def(matrix, expected):
    assert util.is_pos_def(matrix) == expected


@pytest.mark.parametrize(
    "x, x_reg_expect, ro",
    [
        (
            jnp.array([[0.0, 1.0, 1.0], [0, 0.0, 1.0], [0, 0, 0.0]]),
            jnp.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]]),
            1,
        ),
        (
            np.array([[0.0, 1.0, 1.0], [0, 0.0, 1.0], [0, 0, 0.0]]),
            np.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]]),
            1,
        ),
    ],
)
def test_regularize_mat(x, x_reg_expect, ro):
    x_reg = util.reqularize_mat(x, ro)
    assert x_reg.tolist() == x_reg_expect.tolist()


@pytest.mark.parametrize(
    "x, expected",
    [
        (jnp.array([1.0, -1.0, 0.000001, 10000000]), False),
        (jnp.array([1.0, -1.0, jnp.inf, 10000000]), True),
        (jnp.array([1.0, -1.0, -jnp.inf, 10000000]), True),
        (jnp.array([1.0, -1.0, 0.0000001, jnp.nan]), True),
        (jnp.array([1.0, -1.0, jnp.inf, jnp.nan]), True),
        (np.array([1.0, -1.0, np.inf, np.nan]), True),
    ],
)
def test_check_inf_or_nan_array(x, expected):
    is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
    assert is_inf_or_nan_bool == expected
