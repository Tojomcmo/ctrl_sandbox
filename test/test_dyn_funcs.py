import pytest
import numpy as np
import jax
import jax.numpy as jnp

import ctrl_sandbox.dyn_functions as dyn
import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl


def test_nl_pend_dyn_accepts_valid_inputs():
    pend_sys = dyn.single_pm_pend_dyn(g=9.81, b=0.1, l=1.0)
    x_vec = jnp.array([1.0, 1.0])
    u_vec = jnp.array([1.0])
    x_dot = pend_sys.cont_dyn_func(x_vec, u_vec)
    assert True  # This test doesn't have specific assertions yet


def test_double_pend_accepts_valid_inputs():
    dpend_sys = dyn.double_pm_pend_dyn(
        g=9.81,
        m1=1.0,
        l1=1.0,
        m2=1.0,
        l2=1.0,
        b1=0.1,
        b2=0.1,
        shoulder_act=True,
        elbow_act=True,
    )
    x_vec = jnp.array([0.1, 0.1, 0.1, 0.1])
    u_vec = jnp.array([5.0, 1.0])
    x_dot = dpend_sys.cont_dyn_func(x_vec, u_vec)
    assert True  # This test doesn't have specific assertions yet


def test_double_pend_is_jax_compatible():
    dpend_sys = dyn.double_pm_pend_dyn(
        g=9.81,
        m1=1.0,
        l1=1.0,
        m2=1.0,
        l2=1.0,
        b1=0.1,
        b2=0.1,
        shoulder_act=True,
        elbow_act=True,
    )
    x_vec = jnp.array([0.1, 0.1, 0.1, 0.1])
    u_vec = jnp.array([1.0, 1.0])
    xu_k = jnp.concatenate([x_vec, u_vec], axis=0)
    A, B = gen_ctrl.linearize_dynamics(dpend_sys.cont_dyn_func, xu_k, len(x_vec))
    assert True  # This test doesn't have specific assertions yet


def test_double_pend_is_jax_compatible_thru_rk():
    dpend_sys = dyn.double_pm_pend_dyn(
        g=9.81,
        m1=1.0,
        l1=1.0,
        m2=1.0,
        l2=1.0,
        b1=0.1,
        b2=0.1,
        shoulder_act=True,
        elbow_act=True,
    )
    x_vec = jnp.array([0.1, 0.1, 0.1, 0.1])
    u_vec = jnp.array([1.0, 1.0])
    xu_k = jnp.concatenate([x_vec, u_vec], axis=0)
    time_step = 0.01
    dyn_func_discrete = lambda x, u: gen_ctrl.step_rk4(
        dpend_sys.cont_dyn_func, time_step, x, u
    )
    A, B = gen_ctrl.linearize_dynamics(dyn_func_discrete, xu_k, len(x_vec))
    assert True  # This test doesn't have specific assertions yet
