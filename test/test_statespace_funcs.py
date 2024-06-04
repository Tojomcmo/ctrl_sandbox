import pytest
from jax import numpy as jnp
import numpy.testing as nptest

import ctrl_sandbox.statespace_funcs as ss_funcs


def test_state_space_class_accepts_valid_continuous_system():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    ss = ss_funcs.stateSpace(A, B, C, D)
    assert ss.a.tolist() == A.tolist()
    assert ss.b.tolist() == B.tolist()
    assert ss.c.tolist() == C.tolist()
    assert ss.d.tolist() == D.tolist()
    assert ss.type == "continuous"
    assert ss.time_step is None


def test_state_space_class_accepts_valid_discrete_system():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    ss = ss_funcs.stateSpace(A, B, C, D, time_step=0.1)
    assert ss.a.tolist() == A.tolist()
    assert ss.b.tolist() == B.tolist()
    assert ss.c.tolist() == C.tolist()
    assert ss.d.tolist() == D.tolist()
    assert ss.type == "discrete"
    assert ss.time_step == 0.1


def test_state_space_rejects_nonsquare_A_matrix():
    A = jnp.array([[1, 1, 1], [0, 1, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    with pytest.raises(ValueError, match=r"A matrix must be square"):
        ss = ss_funcs.stateSpace(A, B, C, D)


def test_state_space_rejects_A_and_B_matrix_n_dim_mismatch():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    with pytest.raises(
        ValueError, match=r"A matrix row number and B matrix row number must match"
    ):
        ss = ss_funcs.stateSpace(A, B, C, D)


def test_state_space_rejects_A_and_C_matrix_m_dim_mismatch():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0]])
    D = jnp.array([[0, 0]])
    with pytest.raises(
        ValueError, match=r"A matrix row number and C matrix column number must match"
    ):
        ss = ss_funcs.stateSpace(A, B, C, D)


def test_state_space_rejects_B_and_D_matrix_m_dim_mismatch():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0, 0]])
    with pytest.raises(
        Exception, match=r"B matrix column number and D matrix column number must match"
    ):
        ss = ss_funcs.stateSpace(A, B, C, D)


def test_state_space_rejects_C_and_D_matrix_n_dim_mismatch():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0], [0, 0]])
    with pytest.raises(
        Exception, match=r"C matrix row number and D matrix row number must match"
    ):
        ss = ss_funcs.stateSpace(A, B, C, D)


def test_state_space_rejects_invalid_timestep_input():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    with pytest.raises(
        Exception,
        match=("invalid time step -0.1. Must be positive float or None"),
    ):
        ss = ss_funcs.stateSpace(A, B, C, D, time_step=-0.1)
    with pytest.raises(
        Exception,
        match="invalid time step lettuce. Must be positive float or None",
    ):
        ss = ss_funcs.stateSpace(A, B, C, D, time_step="lettuce")
    with pytest.raises(
        Exception,
        match="invalid time step 0. Must be positive float or None",
    ):
        ss = ss_funcs.stateSpace(A, B, C, D, time_step=0)
    with pytest.raises(
        Exception,
        match="invalid time step nan. Must be positive float or None",
    ):
        ss = ss_funcs.stateSpace(A, B, C, D, time_step=float("nan"))


def test_discretize_state_space_accepts_valid_system():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    ss_c = ss_funcs.stateSpace(A, B, C, D)
    time_step = 0.1
    c2d_methods = ["euler", "zoh", "zohCombined"]
    for c2d_method in c2d_methods:
        ss = ss_funcs.discretize_continuous_state_space(ss_c, time_step, c2d_method)
        assert ss.a.shape == A.shape
        assert ss.b.shape == B.shape
        assert ss.c.shape == C.shape
        assert ss.d.shape == D.shape
        assert ss.time_step == time_step
        assert ss.type == "discrete"


def test_discretize_state_space_returns_correct_euler_discretization():
    A = jnp.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    B = jnp.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    C = jnp.array([[1.0, 0.0, 1.0]])
    D = jnp.array([[0.0, 0.0]])
    ss_c = ss_funcs.stateSpace(A, B, C, D)
    time_step = 0.1
    ss = ss_funcs.discretize_continuous_state_space(ss_c, time_step, c2d_method="euler")
    A_d_expected = jnp.eye(3) + (A * time_step)
    B_d_expected = B * time_step
    C_d_expected = C
    D_d_expected = D
    assert ss.a.shape == A.shape
    assert ss.b.shape == B.shape
    assert ss.c.shape == C.shape
    assert ss.d.shape == D.shape
    assert ss.time_step == time_step
    assert ss.type == "discrete"
    assert ss.a.tolist() == A_d_expected.tolist()
    assert ss.b.tolist() == B_d_expected.tolist()
    assert ss.c.tolist() == C_d_expected.tolist()
    assert ss.d.tolist() == D_d_expected.tolist()


def test_descritize_state_space_rejects_discrete_input_system():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    time_step = 0.1
    ss_d = ss_funcs.stateSpace(A, B, C, D, time_step)
    with pytest.raises(Exception, match="input state space is already discrete"):
        ss = ss_funcs.discretize_continuous_state_space(ss_d, time_step)


def test_descritize_state_space_rejects_invalid_c2d_method():
    A = jnp.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    time_step = 0.1
    ss_d = ss_funcs.stateSpace(A, B, C, D)
    with pytest.raises(Exception, match="invalid discretization method"):
        ss = ss_funcs.discretize_continuous_state_space(
            ss_d, time_step, c2d_method="lettuce"
        )


def test_descritize_state_space_rejects_singular_A_matrix_for_zoh():
    A = jnp.array([[0, 1, 1], [0, 1, 1], [0, 0, 1]])
    B = jnp.array([[1, 1], [0, 1], [0, 0]])
    C = jnp.array([[1, 0, 1]])
    D = jnp.array([[0, 0]])
    time_step = 0.1
    ss_d = ss_funcs.stateSpace(A, B, C, D)
    with pytest.raises(Exception, match="determinant of A is excessively small"):
        ss = ss_funcs.discretize_continuous_state_space(
            ss_d, time_step, c2d_method="zoh"
        )
