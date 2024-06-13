import pytest
import jax.numpy as jnp
import jax
import numpy as np
import ctrl_sandbox.util_funcs as util


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


@pytest.mark.parametrize(
    "x, y, alpha, expected",
    [
        (
            jnp.ones((3,), dtype=float),
            jnp.ones((3,), dtype=float) * 2,
            0.5,
            jnp.array([1.5, 1.5, 1.5]),
        ),
        (
            jnp.ones((3,), dtype=float),
            jnp.ones((3,), dtype=float) * 2,
            0.2,
            jnp.array([1.2, 1.2, 1.2]),
        ),
        (
            jnp.ones((3,), dtype=float),
            jnp.ones((3,), dtype=float) * 2,
            1.2,
            jnp.array([2.2, 2.2, 2.2]),
        ),
    ],
)
def test_lin_interp_pair_returns_exp_values(x, y, alpha, expected):
    interp_x_y = util.lin_interp_pair(x, y, alpha)
    assert interp_x_y.all() == expected.all()


@pytest.mark.parametrize(
    "x, alpha, expected",
    [
        (
            jnp.linspace(0, 5, 6),
            0.5,
            jnp.linspace(0.5, 4.5, 5),
        ),
        (
            jnp.linspace(0, 5, 6),
            0.3,
            jnp.linspace(0.3, 4.3, 5),
        ),
        (
            jnp.array([[1, 2], [3, 4], [5, 6]]),
            0.5,
            jnp.array([[1.5, 2.5], [3.5, 4.5]]),
        ),
    ],
)
def test_create_interp_seq_func_output_creates_expected_results(x, alpha, expected):
    lin_interp_seq_func = util.create_lin_interp_seq_func()
    interp_x_seq = lin_interp_seq_func(x, alpha)
    assert interp_x_seq.all() == expected.all()


def test_create_interp_seq_mid_point_func_output_creates_expected_results():
    x = jnp.linspace(0, 5, 6)
    lin_interp_seq_func = util.create_lin_interp_seq_mid_point_func()
    interp_x_seq = lin_interp_seq_func(x)
    expected = jnp.linspace(0.5, 4.5, 5)
    assert interp_x_seq.all() == expected.all()


def test_create_interp_seq_mid_point_func_output_is_diff_compatible():
    x = jnp.linspace(0, 3, 4)
    lin_interp_seq_func = util.create_lin_interp_seq_mid_point_func()
    jac_vec = jax.jacfwd(lin_interp_seq_func)(x)
    jac_vec_expected = jnp.array(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5]]
    )
    assert jac_vec.all() == jac_vec_expected.all()
