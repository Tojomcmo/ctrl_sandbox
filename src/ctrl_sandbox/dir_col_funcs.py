import jax.numpy as jnp
import scipy.optimize as spopt
from typing import Callable, Tuple
import jax.lax as lax

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.util_funcs as util


class dirColConfig:
    def __init__(
        self,
        cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        time_step: float,
        x_len: int,
        u_len: int,
        len_seq: int,
    ) -> None:
        self.cost_func = cost_func
        self.cont_dyn_func = cont_dyn_func
        self.time_step = time_step
        self.len_seq = len_seq
        self.x_len = x_len
        self.u_len = u_len
        pass

    def create_dyn_constraint_func(
        self,
        cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        integrate_scheme,
    ) -> None:

        self.lin_interp_seq = util.create_lin_interp_seq_func()
        self.lin_interp_seq_mid_point = util.create_lin_interp_seq_mid_point_func()

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

        def create_dir_col_cost_diff(opt_vec: jnp.ndarray) -> jnp.ndarray:
            return calculate_dir_col_cost_pre_decorated(
                cost_for_calc_scan_func,
                self.len_seq,
                self.x_len,
                self.u_len,
                opt_vec,
            )

        def create_dir_col_cost_calc(opt_vec: jnp.ndarray) -> float:
            return (
                calculate_dir_col_cost_pre_decorated(
                    cost_for_calc_scan_func,
                    self.len_seq,
                    self.x_len,
                    self.u_len,
                    opt_vec,
                )
            ).item()

        self.dir_col_cost_diff = create_dir_col_cost_diff
        self.dir_col_cost_calc = create_dir_col_cost_calc


def calculate_dir_col_cost_pre_decorated(
    cost_for_calc_scan_func: Callable[
        [Tuple[int, jnp.ndarray], jnp.ndarray],
        Tuple[Tuple[int, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ],
    len_seq: int,
    x_len: int,
    u_len: int,
    opt_vec: jnp.ndarray,
) -> jnp.ndarray:
    x_seq, u_seq = breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
    cost_float, _, _, _ = gen_ctrl.calculate_total_cost_pre_decorated(
        cost_for_calc_scan_func, x_seq, u_seq
    )
    return cost_float


def breakout_opt_vec(
    len_seq: int, x_len: int, u_len: int, opt_vec: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_seq = jnp.reshape(opt_vec[: len_seq * x_len], (len_seq, x_len))
    u_seq = jnp.reshape(opt_vec[len_seq * x_len :], (len_seq, u_len))
    return x_seq, u_seq


def calculate_dyn_seq_full(
    lin_interp_mid_point_func: Callable[[jnp.ndarray], jnp.ndarray],
    knot_point_dyn_scan_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray]], Tuple[None, jnp.ndarray]
    ],
    mid_point_dyn_scan_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        Tuple[None, jnp.ndarray],
    ],
    colloc_calc_func: Callable[
        [None, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        Tuple[None, jnp.ndarray],
    ],
    h: float,
    len_seq: int,
    x_len: int,
    u_len: int,
    x_o: jnp.ndarray,
    x_f: jnp.ndarray,
    opt_vec: jnp.ndarray,
) -> jnp.ndarray:
    # lin_interp_u_mid_points
    x_seq, u_seq = breakout_opt_vec(len_seq, x_len, u_len, opt_vec)
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
    _, ceq_seq = lax.scan(colloc_calc_func, None, (colloc_calc_seqs))

    return jnp.concatenate((ceq_o, ceq_seq, ceq_f), axis=0)


def calculate_init_final_constraint(
    x_seq: jnp.ndarray, x_o: jnp.ndarray, x_f: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ceq_o = x_o - x_seq[0]
    ceq_f = x_f - x_seq[-1]
    return ceq_o, ceq_f


def create_mid_point_dyn_calc_seqs(
    x_seq: jnp.ndarray, dyn_seq_k_p: jnp.ndarray, u_seq_m_p: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_k = (x_seq[:-1]).reshape(-1, 1)
    x_kp1 = (x_seq[1:]).reshape(-1, 1)
    dyn_k = (dyn_seq_k_p[:-1]).reshape(-1, 1)
    dyn_kp1 = (dyn_seq_k_p[1:]).reshape(-1, 1)
    u_k = u_seq_m_p.reshape(-1, 1)
    return (x_k, x_kp1, dyn_k, dyn_kp1, u_k)


def create_colloc_calc_seqs(
    x_seq: jnp.ndarray, dyn_seq_k_p: jnp.ndarray, dyn_seq_m_p: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_k = (x_seq[:-1]).reshape(-1, 1)
    x_kp1 = (x_seq[1:]).reshape(-1, 1)
    dyn_k = (dyn_seq_k_p[:-1]).reshape(-1, 1)
    dyn_m_p = dyn_seq_m_p.reshape(-1, 1)
    dyn_kp1 = (dyn_seq_k_p[1:]).reshape(-1, 1)
    return (x_k, x_kp1, dyn_k, dyn_m_p, dyn_kp1)


def calc_mid_point_dyn_scan_func(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[None, jnp.ndarray]:
    x_k, x_kp1, dyn_k, dyn_kp1, u_m_p = seqs
    x_m_p = interp_state_traj_H_S_post_dyn_calc(h, x_k, x_kp1, dyn_k, dyn_kp1)
    dyn_m_p = cont_dyn_func(x_m_p, u_m_p)
    return carry, dyn_m_p


def calc_knot_point_dyn_scan_func(
    cont_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[None, jnp.ndarray]:
    x_k, u_k = seqs
    x_dot_k = cont_dyn_func(x_k, u_k)
    return carry, x_dot_k


def interp_state_traj_H_S_post_dyn_calc(
    h: float,
    x_k: jnp.ndarray,
    x_kp1: jnp.ndarray,
    dyn_k: jnp.ndarray,
    dyn_kp1: jnp.ndarray,
) -> jnp.ndarray:
    return (0.5) * (x_k + x_kp1) + (h / 8) * (dyn_k - dyn_kp1)


def calc_colloc_H_S_scan_func(
    h: float,
    carry: None,
    seqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
):
    x_k, x_kp1, dyn_k, dyn_m_p, dyn_kp1 = seqs
    state_integrate = calc_H_S_integ_post_dyn_calc(h, dyn_k, dyn_m_p, dyn_kp1)
    ceq_m_p = state_integrate - (x_kp1 - x_k)
    return carry, ceq_m_p


def calc_H_S_integ_post_dyn_calc(
    h: float, dyn_k: jnp.ndarray, dyn_m_p: jnp.ndarray, dyn_kp1: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the collocation constraint using hermite-simpson integration
    x_kp1-x_k = (hk/6)(f(k)+4*f(k+1/2)+f(k+1))
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
