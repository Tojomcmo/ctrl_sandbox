import pytest
from jax import numpy as jnp
import numpy.testing as nptest

import ctrl_sandbox.integrate_funcs as integrate


def test_step_euler_forward_returns_exp_vals_from_valid_inputs():
    def cont_dyn_func(x_k: jnp.ndarray, u_k: jnp.ndarray) -> jnp.ndarray:
        A = jnp.array([[0, 1], [1, 1]], dtype=float)
        B = jnp.array([[0], [1]], dtype=float)
        x_dot = A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)
        return x_dot.reshape(-1)

    # test function
    h = 0.1
    x_k = jnp.array([1, 1], dtype=float)
    u_k = jnp.array([1], dtype=float)
    x_kp1_test = integrate.step_euler_forward(cont_dyn_func, h, x_k, u_k)
    # create expected output
    A = jnp.array([[0, 1], [1, 1]], dtype=float)
    B = jnp.array([[0], [1]], dtype=float)
    x_kp1_expect = (
        x_k.reshape(-1, 1) + (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)) * h
    ).reshape(-1)
    assert x_kp1_test.all() == x_kp1_expect.all()


def test_step_rk4_returns_exp_vals_from_valid_inputs():
    def cont_dyn_func(x_k: jnp.ndarray, u_k: jnp.ndarray) -> jnp.ndarray:
        A = jnp.array([[0, 1], [1, 1]], dtype=float)
        B = jnp.array([[0], [1]], dtype=float)
        x_dot = A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)
        return x_dot.reshape(-1)

    def rk4_test(dyn_func, x_k, u_k, h):
        f1 = dyn_func(x_k, u_k)
        f2 = dyn_func(x_k + 0.5 * h * f1, u_k)
        f3 = dyn_func(x_k + 0.5 * h * f2, u_k)
        f4 = dyn_func(x_k + h * f3, u_k)
        return x_k + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    # test function
    h = 0.001
    x_k = jnp.array([1, 1], dtype=float)
    u_k = jnp.array([1], dtype=float)
    x_kp1_test = integrate.step_rk4(cont_dyn_func, h, x_k, u_k)
    # create expected outputs 1
    A = jnp.array([[0, 1], [1, 1]], dtype=float)
    B = jnp.array([[0], [1]], dtype=float)
    x_kp1_expect_1 = (
        x_k.reshape(-1, 1) + (A @ x_k.reshape(-1, 1) + B @ u_k.reshape(-1, 1)) * h
    ).reshape(-1)
    # create expected outputs 2
    x_kp1_expect_2 = rk4_test(cont_dyn_func, x_k, u_k, h)

    nptest.assert_allclose(x_kp1_test, x_kp1_expect_1, rtol=1e-5, atol=1e-5)
    assert x_kp1_test.all() == x_kp1_expect_2.all()
