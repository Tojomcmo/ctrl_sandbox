import pytest
import jax.numpy as jnp
import jax
from typing import Tuple, Callable

import ctrl_sandbox.dir_col_funcs as dir_col


def test_calculate_dir_col_cost_returns_exp_values():

    assert True


def test_breakout_opt_vec_returns_exp_values():
    len_seq = 10
    x_len = 4
    u_len = 2
    opt_vec = jnp.linspace(0, len_seq * (x_len + u_len) - 1, len_seq * (x_len + u_len))
    x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    x_expected = jnp.reshape(
        jnp.linspace(0, (len_seq * x_len) - 1, len_seq * x_len), (len_seq, x_len)
    )
    u_expected = jnp.reshape(
        jnp.linspace((len_seq * x_len), (len_seq * u_len) - 1, len_seq * u_len),
        (len_seq, u_len),
    )
    assert x_seq.all() == x_expected.all()
    assert u_seq.all() == u_expected.all()


def test_breakout_opt_vec_is_scan_compatible():

    def scan_func(
        len_seq: int, x_len: int, u_len: int, carry: None, seqs: jnp.ndarray
    ) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:

        opt_vec = seqs
        x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
        return carry, (x_seq, u_seq)

    assert True


def test_breakout_opt_vec_is_diff_compatible():
    def calc_cost_seq(x_seq: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
        return (x_seq.reshape(-1, 1)).T @ x_seq.reshape(-1, 1) + (
            u_seq.reshape(-1, 1)
        ).T @ u_seq.reshape(-1, 1)

    def diff_func(len_seq, x_len, u_len, opt_vec):
        x_seq, u_seq = dir_col.breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
        return calc_cost_seq(x_seq, u_seq)

    len_seq = 5
    x_len = 4
    u_len = 2
    diff_func_curried = lambda opt_vec: diff_func(len_seq, x_len, u_len, opt_vec)
    opt_vec = jnp.linspace(0, len_seq * (x_len + u_len) - 1, len_seq * (x_len + u_len))
    jac_vec = (jax.jacfwd(diff_func_curried)(opt_vec)).reshape(-1)
    jac_vec_expect = 2 * opt_vec
    assert jac_vec.all() == jac_vec_expect.all()
