import jax.numpy as jnp
import scipy.optimize as spopt
from typing import Callable, Tuple, Union
import jax
import jax.lax as lax
import numpy as np
import numpy.typing as npt

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.util_funcs as util


class dirColConfig:
    def __init__(
        self,
        cost_func: Callable[
            [jnp.ndarray, jnp.ndarray, int],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        time_step: float,
        x_len: int,
        u_len: int,
        len_seq: int,
        x_o: Union[npt.NDArray[np.float64], jnp.ndarray],
        x_f: Union[npt.NDArray[np.float64], jnp.ndarray],
    ) -> None:
        self.cost_func = cost_func
        self.cont_dyn_func = cont_dyn_func
        self.time_step = time_step
        self.len_seq = len_seq
        self.x_len = x_len
        self.u_len = u_len
        self.x_o = jnp.array(x_o)
        self.x_f = jnp.array(x_f)
        self.create_dir_col_cost_funcs(cost_func)
        self.create_dyn_colloc_funcs(cont_dyn_func)
        pass

    def create_dyn_colloc_funcs(
        self,
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> None:
        lin_interp_mid_point_func = util.create_lin_interp_seq_mid_point_func()
        knot_point_dyn_scan_func = lambda carry, seqs: calc_knot_point_dyn_scan_func(
            cont_dyn_func, carry, seqs
        )
        mid_point_dyn_scan_func = lambda carry, seqs: calc_mid_point_dyn_scan_func(
            cont_dyn_func, self.time_step, carry, seqs
        )
        colloc_calc_scan_func = lambda carry, seqs: calc_colloc_H_S_scan_func(
            self.time_step, carry, seqs
        )
        dyn_colloc_func = lambda opt_vec: calc_dyn_colloc_ceqs_full(
            lin_interp_mid_point_func,
            knot_point_dyn_scan_func,
            mid_point_dyn_scan_func,
            colloc_calc_scan_func,
            self.len_seq,
            self.x_len,
            self.u_len,
            self.x_o,
            self.x_f,
            opt_vec,
        )
        self.dyn_colloc_calc = prep_dyn_colloc_curried_for_calc(dyn_colloc_func)
        self.dyn_colloc_diff = prep_dyn_colloc_curried_for_diff(dyn_colloc_func)

    def create_dir_col_cost_funcs(
        self,
        cost_func: Callable[
            [jnp.ndarray, jnp.ndarray, int],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
    ) -> None:

        cost_for_calc_scan_func = gen_ctrl._curry_cost_for_calc_scan_func(
            cost_func, self.x_len
        )
        dir_col_cost: Callable = lambda opt_vec: calc_dir_col_cost_pre_decorated(
            cost_for_calc_scan_func, self.len_seq, self.x_len, self.u_len, opt_vec
        )

        self.dir_col_cost_calc = prep_dir_col_cost_curried_for_calc(dir_col_cost)
        self.dir_col_cost_diff = prep_dir_col_cost_curried_for_diff(dir_col_cost)


def create_opt_vec_from_x_u_seqs(
    x_seq: Union[npt.NDArray[np.float64], jnp.ndarray],
    u_seq: Union[npt.NDArray[np.float64], jnp.ndarray],
) -> npt.NDArray[np.float64]:
    """
    **Get x and u seqs from optimization vector**
    - [in] x_seq - state sequence, dim (N,n)
    - [in] u_seq - control sequence, dim (N,m)
    - [ret] opt_vec - optimization vector, dim(N*(n+m),)

    opt_vec will be organized as [x_k; x_kp1; - ; x_N; u_k; u_kp1; - ; u_N]\n
    Function pairs with breakout_opt_vec.

    """
    return np.concatenate([x_seq.reshape(-1), u_seq.reshape(-1)], axis=0)


def breakout_opt_vec(
    len_seq: int,
    x_len: int,
    u_len: int,
    opt_vec: Union[npt.NDArray[np.float64], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Get x and u seqs from optimization vector**
    - [in] len_seq - length of sequence (not length of vector),N, int
    - [in] x_len - length of state vector, n int
    - [in] u_len - length of control vector, m, int
    - [in] opt_vec - optimization vector, dim(N*(n+m),)
    - [ret] x_seq - state sequence, dim (N,n)
    - [ret] u_seq - control sequence, dim (N,m)

    opt_vec must be organized as [x_k; x_kp1; - ; x_N; u_k; u_kp1; - ; u_N]\n
    Function pairs with create_opt_vec_from_x_u_seqs.

    """
    x_seq = jnp.reshape(opt_vec[: len_seq * x_len], (len_seq, x_len))
    u_seq = jnp.reshape(opt_vec[len_seq * x_len :], (len_seq - 1, u_len))
    return x_seq, u_seq


def calc_dir_col_cost_pre_decorated(
    cost_for_calc_scan_func: Callable[
        [Tuple[int, jnp.ndarray], jnp.ndarray],
        Tuple[Tuple[int, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ],
    len_seq: int,
    x_len: int,
    u_len: int,
    opt_vec: jnp.ndarray,
) -> jnp.ndarray:
    """
    **Calculate cost sequence of optimization vector**\n
    Requires lax.scan compatible cost function
    - [in] cost_for_cal_scan_func - callable cost calc scan func
        - compatible with gen_ctrl.calculate_total_cost_pre_decorated
    - [in] len_seq - sequence length, N, for opt_vec parsing, int
    - [in] x_len - length of state vector, n, for opt_vec parsing, int
    - [in] u_len - length of control vector, m, for opt_vec parsing, int
    - [in] opt_vec - optimization vector, dim(N*(n+m),)
    - [ret] cost_float - calculated cost value, dim(1,)
    """
    x_seq, u_seq = breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    cost_float, _, _, _ = gen_ctrl.calculate_total_cost_pre_decorated(
        cost_for_calc_scan_func, x_seq, u_seq
    )
    return cost_float


def prep_dir_col_cost_curried_for_calc(
    dir_col_cost_curried: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    **Wrap curried dir col cost function for cost calculation**
    """

    def dir_col_cost_calc(opt_vec: npt.NDArray[np.float64]):
        return (np.array(dir_col_cost_curried(jnp.array(opt_vec)))).item()

    return dir_col_cost_calc


def prep_dir_col_cost_curried_for_diff(
    dir_col_cost_curried: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    **Wrap curried dir col cost function for cost diff wrt opt vec**
    """

    def dir_col_cost_diff(opt_vec: npt.NDArray[np.float64]):
        return (np.array(jax.jacfwd(dir_col_cost_curried)(jnp.array(opt_vec)))).reshape(
            -1
        )

    return dir_col_cost_diff


def prep_dyn_colloc_curried_for_calc(
    dyn_colloc_curried: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    **Wrap curried dir col cost function for cost calculation**
    """

    def dyn_colloc_calc(opt_vec: npt.NDArray[np.float64]):
        return np.array(dyn_colloc_curried(jnp.array(opt_vec)))

    return dyn_colloc_calc


def prep_dyn_colloc_curried_for_diff(
    dyn_colloc_curried: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    **Wrap curried dir col cost function for cost diff wrt opt vec**
    """

    def dyn_colloc_diff(opt_vec: npt.NDArray[np.float64]):
        return np.array(jax.jacfwd(dyn_colloc_curried)(jnp.array(opt_vec)))

    return dyn_colloc_diff


def calc_dyn_colloc_ceqs_full(
    lin_interp_mid_point_func: Callable[[jnp.ndarray], jnp.ndarray],
    knot_point_dyn_scan_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray]], Tuple[None, jnp.ndarray]
    ],
    mid_point_dyn_scan_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        Tuple[None, jnp.ndarray],
    ],
    colloc_calc_scan_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        Tuple[None, jnp.ndarray],
    ],
    len_seq: int,
    x_len: int,
    u_len: int,
    x_o: jnp.ndarray,
    x_f: jnp.ndarray,
    opt_vec: jnp.ndarray,
) -> jnp.ndarray:
    # lin_interp_u_mid_points
    x_seq, u_seq = breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    # append u_seq with zero vec for collocation
    u_seq = jnp.append(u_seq, (jnp.zeros((u_len,), dtype=float)).reshape(-1, 1), axis=0)
    ceq_o, ceq_f = calculate_init_final_constraint(x_seq, x_o, x_f)
    u_seq_m_p = lin_interp_mid_point_func(u_seq)
    # calc_knot_point_dyn_seq
    _, dyn_seq_k_p = lax.scan(knot_point_dyn_scan_func, None, (x_seq, u_seq))
    # formulate scan sequences for midpoint dynamics calculation
    mid_point_dyn_calc_seqs = create_mid_point_dyn_calc_seqs(
        x_seq, dyn_seq_k_p, u_seq_m_p
    )
    # Calculate collocation dynamics and stack dynamic sequence
    _, dyn_seq_m_p = lax.scan(mid_point_dyn_scan_func, None, (mid_point_dyn_calc_seqs))
    colloc_calc_seqs = create_colloc_calc_seqs(x_seq, dyn_seq_k_p, dyn_seq_m_p)
    _, ceq_seq = lax.scan(colloc_calc_scan_func, None, (colloc_calc_seqs))

    return jnp.concatenate((ceq_o, ceq_seq.reshape(-1), ceq_f), axis=0)


def calc_knot_point_dyn_scan_func(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[None, jnp.ndarray]:
    """
    **Scan proto-function for calculating sequence of dynamics at knot points**\n
    Must curry dynamics function to prepare for lax.scan
    - [in] cont_dyn_func - Continuous dynamics function, callable
    - [in] carry - Nonetype
    - [in] seqs - tuple of sequence elements:
        - x_k - state at knot point k, dim(n,)
        - u_k - control at knot point k, dim(m,)
    - [ret] carry - Nonetype
    - [ret] x_dot_k - state_derivatives at knot point k, dim (n,)
    """
    x_k, u_k = seqs
    x_dot_k = cont_dyn_func(x_k, u_k)
    return carry, x_dot_k


def calc_mid_point_dyn_scan_func(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[None, jnp.ndarray]:
    """
    **Scan proto-function for calculating sequence of midpoint dynamics**\n
    Must curry dynamics function and timestep to prepare for lax.scan
    - [in] cont_dyn_func - Continuous dynamics function, callable
    - [in] h - timestep, must be positive float
    - [in] carry - Nonetype
    - [in] seqs - tuple of sequence elements:
        - x_k - state at knot point k, dim(n,)
        - x_kp1 - state at knot point k+1, dim(n,)
        - dyn_k - state derivatives at knot point k, dim(n,)
        - dyn_kp1 - state derivatives at knot point k+1, dim(n,)
        - u_m_p - control interpolated at mid point k + 1/2, dim(m,)
    - [ret] carry - Nonetype
    - [ret] dyn_m_p - state derivatives interpolated at mid point k + 1/2, dim(n,)
    """
    x_k, x_kp1, dyn_k, dyn_kp1, u_m_p = seqs
    x_m_p = interp_state_m_p_H_S_post_dyn_calc(h, x_k, x_kp1, dyn_k, dyn_kp1)
    dyn_m_p = cont_dyn_func(x_m_p, u_m_p)
    return carry, dyn_m_p


def calc_colloc_H_S_scan_func(
    h: float,
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[None, jnp.ndarray]:
    """
    **Scan proto-function for calculating sequence of collocation constraint values**
    Must curry timestep to prepare for lax.scan
    - [in] h - timestep, must be positive float
    - [in] carry - Nonetype
    - [in] seqs - tuple of sequence elements:
        - x_k - state at knot point k, dim(n,)
        - x_kp1 - state at knot point k+1, dim(n,)
        - dyn_k - state derivatives at knot point k, dim(n,)
        - dyn_m_p - state derivatives interpolated at mid point k + 1/2, dim(n,)
        - dyn_kp1 - state derivatives at knot point k+1, dim(n,)
    - [ret] carry - Nonetype
    - [ret] ceq_m_p - collocation equality constraint value at midpoint, dim(n,)
    """
    x_k, x_kp1, dyn_k, dyn_m_p, dyn_kp1 = seqs
    state_integrate = calc_H_S_integ_post_dyn_calc(h, dyn_k, dyn_m_p, dyn_kp1)
    ceq_m_p = state_integrate - (x_kp1 - x_k)
    return carry, ceq_m_p


def calculate_init_final_constraint(
    x_seq: jnp.ndarray, x_o: jnp.ndarray, x_f: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Calculate equality constraint values for sequence initial and final**
    """
    ceq_o = x_o - x_seq[0]
    ceq_f = x_f - x_seq[-1]
    return ceq_o, ceq_f


def create_mid_point_dyn_calc_seqs(
    x_seq: jnp.ndarray, dyn_seq: jnp.ndarray, u_seq_m_p: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    **create scan compatible tuple of seqs for mid point dynamics calculation**
    - [in] x_seq - state sequence at knot points, dim(N,n)
    - [in] dyn_seq - state derivatives at knot points, dim(N,n)
    - [in] u_m_p - control sequence interpolated at mitpoints, dim(N-1,m)
    - [ret] seq - Tuple of index-corrected sequences for midpoint dynamics scan
        - (x_k_seq(N-1,n), x_kp1_seq(N-1,n), dyn_k_seq(N-1,n),
        dyn_kp1_seq(N-1,n), u_k_seq(N-1,m))
    """
    return (x_seq[:-1], x_seq[1:], dyn_seq[:-1], dyn_seq[1:], u_seq_m_p)


def create_colloc_calc_seqs(
    x_seq: jnp.ndarray, dyn_seq_k_p: jnp.ndarray, dyn_seq_m_p: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    **create scan compatible tuple of seqs for dynamics collocation calculation**
    - [in] x_seq - state sequence at knot points, dim(N,n)
    - [in] dyn_seq_k_p - state derivatives at knot points, dim(N,n)
    - [in] dyn_seq_m_p - state derivatives interpolated at mid points, dim(N-1,n)
    - [ret] seqs - Tuple of index-corrected sequences for collocation scan:
        - (x_k_seq, x_kp1_seq, dyn_k_seq, dyn_m_p_seq, dyn_kp1_seq)
        - all dim(N-1,n)
    """
    return (x_seq[:-1], x_seq[1:], dyn_seq_k_p[:-1], dyn_seq_m_p, dyn_seq_k_p[1:])


def interp_state_m_p_H_S_post_dyn_calc(
    h: float,
    x_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    dyn_k: jnp.ndarray,
    dyn_kp1: jnp.ndarray,
) -> jnp.ndarray:
    """
    **Calculate state mid point x_k+1/2 using Hermite Simpson interpolation** \n
    Calculated using knot point states and dynamics. This function does not calculate
    the dynamics.
    - [in] h - time step between knot points, required positive float
    - [in] x_k - state vector at knot point k, jax array dim (n,)
    - [in] x_kp1 - state vector at knot point kp1, jax array dim (n,)
    - [in] dyn_k - state derivatives at knot point k, dim (n,)
    - [in] dyn_kp1 - state derivatives at knot point kp1, jax array dim (n,)
    - [ret] x_m_p - interpolated state vector at mid point, jax array dim (n,)
    """
    return (0.5) * (x_k + x_kp1) + (h / 8) * (dyn_k - dyn_kp1)


def calc_H_S_integ_post_dyn_calc(
    h: float, dyn_k: jnp.ndarray, dyn_m_p: jnp.ndarray, dyn_kp1: jnp.ndarray
) -> jnp.ndarray:
    """
    **Calculate the hermite-simpson integration for knot point state change**
    - [in] h - timestep, must be positive float
    - [in] dyn_k - state derivatives at knot point k, dim (n,)
    - [in] dyn_m_p - state derivatives interpolated at mid point k + 1/2, dim(n,)
    - [in] dyn_kp1 - state derivatives at knot point k+1, dim(n,)
    - [ret] (x_kp1-x_k) - integrated state change from k to kp1, dim(n,)
    """
    return (h / 6) * (dyn_k + 4 * dyn_m_p + dyn_kp1)


#########################################


def calculate_hermite_simpson_col_point(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    u_kp1: jnp.ndarray,
):
    """
    Calculate the collocation constraint using hermite-simpson integration
    x_kp1-x_k = (hk/6)(f(k)+4*f(k+1/2)+f(k+1))
    """
    x_mid = interp_state_traj_hermite_simpson(cont_dyn_func, h, x_k, u_k, x_kp1, u_kp1)
    u_mid = u_k + (h * 0.5) * (u_kp1 - u_k)
    return (h / 6) * (
        cont_dyn_func(x_k, u_k)
        + 4 * cont_dyn_func(x_mid, u_mid)
        + cont_dyn_func(x_kp1, u_kp1)
    )


def interp_state_traj_hermite_simpson(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    u_kp1: jnp.ndarray,
) -> jnp.ndarray:
    return (0.5) * (x_k + x_kp1) + (h / 8) * (
        cont_dyn_func(x_k, u_k) - cont_dyn_func(x_kp1, u_kp1)
    )
