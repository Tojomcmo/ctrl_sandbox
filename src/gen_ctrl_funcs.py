import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import scipy

from numpy import typing as npt
from typing import Callable, Tuple
from scipy.integrate import solve_ivp


class stateSpace:
    def __init__(
        self,
        a_mat: jnp.ndarray,
        b_mat: jnp.ndarray,
        c_mat: jnp.ndarray,
        d_mat: jnp.ndarray,
        time_step=None,
    ):
        if a_mat.shape[0] != a_mat.shape[1]:
            raise ValueError("A matrix must be square")
        elif a_mat.shape[0] != b_mat.shape[0]:
            raise ValueError("A and B matrices must have same n(state) dimension")
        elif a_mat.shape[0] != c_mat.shape[1]:
            raise ValueError("A and C matrices must have same m(state) dimension")
        elif b_mat.shape[1] != d_mat.shape[1]:
            raise ValueError("B and D matrices must have the same m(control) dimension")
        elif c_mat.shape[0] != d_mat.shape[0]:
            raise ValueError(
                "C and D matrices must have the same n(measurement) dimension"
            )
        else:
            if time_step is None:
                self.a = a_mat
                self.b = b_mat
                self.c = c_mat
                self.d = d_mat
                self.time_step = time_step
                self.type = "continuous"
            elif isinstance(time_step, float) and time_step > 0:
                self.a = a_mat
                self.b = b_mat
                self.c = c_mat
                self.d = d_mat
                self.time_step = time_step
                self.type = "discrete"
            else:
                raise ValueError(
                    f"invalid time step definition {time_step}. time_step must be a positive float or None"
                )


def _dyn_for_sim_scan_func(
    discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    carry_x_k: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Kernel scan function for forward dynamics simulation**
     - This function decorates the discrete_dyn_func for jax.lax.scan by propagating carry and seq values as returns
    """
    x_kp1 = discrete_dyn_func(carry_x_k, u_seq)
    return x_kp1, x_kp1


def _curry_dyn_for_sim_scan_func(
    discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
) -> Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    def scan_func(
        carry_x_k: jnp.ndarray, u_seq: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _dyn_for_sim_scan_func(discrete_dyn_func, carry_x_k, u_seq)

    return scan_func


def simulate_forward_dynamics_seq_pre_decorated(
    sim_dyn_scan_func: Callable[
        [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
    ],
    x_init: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """**Forward simulate a control sequence from an initial state through a discrete dynamics function** \n
    - This function uses jax.lax.scan to efficiently generate the paired x_seq to the provided u_seq through
    sequential calculation of the discrete dynamics provided. \n
    - discrete_dyn_func - [in] - discrete dynamics function (x_k,u_k)->(x_kp1), **must be jax compatible** \n
    - x_init - [in] - initial state vector, dimension (n,) \n
    - u_seq - [in] - control sequence, dimension(len_seq-1,m) , length len_seq-1 because the final state has no control value \n
    - return x_seq - [out] state sequence, dimension(len_seq,n)
    """
    _, x_seq_out = lax.scan(sim_dyn_scan_func, x_init, u_seq)
    return jnp.append(x_init.reshape(1, -1), x_seq_out, axis=0)


def simulate_forward_dynamics_seq(
    discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_init: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """**Forward simulate a control sequence from an initial state through a discrete dynamics function** \n
    - This function uses jax.lax.scan to efficiently generate the paired x_seq to the provided u_seq through
    sequential calculation of the discrete dynamics provided. \n
    - discrete_dyn_func - [in] - discrete dynamics function (x_k,u_k)->(x_kp1), **must be jax compatible** \n
    - x_init - [in] - initial state vector, dimension (n,) \n
    - u_seq - [in] - control sequence, dimension(len_seq-1,m) , length len_seq-1 because the final state has no control value \n
    - return x_seq - [out] state sequence, dimension(len_seq,n)
    """
    discrete_dyn_func_scan = lambda carry_x_k, u_seq: _dyn_for_sim_scan_func(
        discrete_dyn_func, carry_x_k, u_seq
    )
    _, x_seq_out = lax.scan(discrete_dyn_func_scan, x_init, u_seq)
    return jnp.append(x_init.reshape(1, -1), x_seq_out, axis=0)


def linearize_dynamics(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    xu_k: jnp.ndarray,
    x_k_len,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Linearizes the dynamics function about the primals x and u** \n
    dyn_func  - [in] dynamics function, may be continuous or discrete, **must be jacfwd compatible** \n
    xu_k      - [in] concatenated primal state and control linearization point, **NOT a sequence** \n
    x_k_len   - [in] length of state vector, for splitting linearization into respective parts \n
    A_lin     - [out] continuous or discrete time linearization of dyn_func wrt x eval at x_k,u_k \n
    B_lin     - [out] continuous or discrete time linearization of dyn_func wrt u eval at x_k,u_k \n
    """
    dyn_func_xu = lambda xu_k: dyn_func(xu_k[:x_k_len], xu_k[x_k_len:])
    ab_lin_cat = jax.jacfwd(dyn_func_xu)(xu_k)
    a_lin = ab_lin_cat[:x_k_len, :x_k_len]
    b_lin = ab_lin_cat[:x_k_len, x_k_len:]
    return a_lin, b_lin


def _dyn_lin_scan_func(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_len: int,
    carry: None,
    x_and_u_k: jnp.ndarray,
) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    **Partially format linearize dynamics function for lax.scan, feeding the carry and output format**
    - dyn_func [in] - callable dynamics function, usually discrete dynamics given discrete sequences \n
    - x_len [in] - int length of state vector \n
    - carry [in] - Nonetype, necessary for lax.scan \n
    - x_and_u_k [in] - jnp array of dim(n+m,), considered as a single entry in the seqs provided to lax.scan \n

    This function must be further partialed to properly function with lax.scan to only accept arguments (carry, seqs)
    """
    a_lin, b_lin = linearize_dynamics(dyn_func, x_and_u_k, x_len)
    return carry, (a_lin, b_lin)


def _curry_dyn_lin_scan_func(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], x_len: int
) -> Callable[[None, jnp.ndarray], Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]]:
    """
    **Create curried scan function for lax.scan for linearizing a dynamic sequence of x and u states through a dyanmics function**
    - discrete_dyn_func [in] - callable discrete dynamics function for linearization
    - x_len [in] - length of state vector, for splitting linearization into state and control matrices
    - return scan_func_curried - fully formatted function for lax.scan

    This function returns a fully formatted function for lax.scan operation for dynamic sequence linearization.
    """

    def dyn_lin_scan_func_curried(carry: None, xu_k: jnp.ndarray):
        carry_out, A_B_k = _dyn_lin_scan_func(dyn_func, x_len, carry, xu_k)
        return carry_out, A_B_k

    return dyn_lin_scan_func_curried


def calculate_linearized_state_space_seq_pre_decorated(
    dyn_lin_scan_func: Callable[
        [None, jnp.ndarray], Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]
    ],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Calculate the A and B linearization matrices of a set of state and control sequences**
    - lin_scan_func - [in] - lax.scan compatible function for linearizing dynamics functions
    - x_seq [in] - state sequence for linearization of dimension (len_seq, n)
    - u_seq [in] - control sequence for linearization of dimension (len_seq-1, m)

    The lin_scan_func must accept:
     - carry of Nonetype
     - seqs of jnp.ndarray of size (len_seq-1, n+m). len_seq-1 because the final timestep has no corresponding
    dynamics.
    """
    xu_seq = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
    carry = None
    _, (a_lin_seq, b_lin_seq) = lax.scan(dyn_lin_scan_func, carry, xu_seq)
    return a_lin_seq, b_lin_seq


def calculate_linearized_state_space_seq(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    **Calculate the A and B linearization matrices of a set of state and control sequences**
    - lin_scan_func - [in] - lax.scan compatible function for linearizing dynamics functions
    - x_seq [in] - state sequence for linearization of dimension (len_seq, n)
    - u_seq [in] - control sequence for linearization of dimension (len_seq-1, m)

    The lin_scan_func must accept:
     - carry of Nonetype
     - seqs of jnp.ndarray of size (len_seq-1, n+m). len_seq-1 because the final timestep has no corresponding
    dynamics.
    """
    lin_scan_func = _curry_dyn_lin_scan_func(dyn_func, x_seq.shape[1])
    xu_seq = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
    carry = None
    _, (a_lin_seq, b_lin_seq) = lax.scan(lin_scan_func, carry, xu_seq)
    return a_lin_seq, b_lin_seq


def step_euler_forward(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], npt.NDArray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
) -> jnp.ndarray:
    """**Perform single step euler forward integration**"""
    return x_k + dyn_func(x_k, u_k) * h


def step_rk4(
    dyn_func_ad: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    h: float,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
) -> jnp.ndarray:
    """**Perform single step Runge Kutta 4th order integration**"""
    f1 = dyn_func_ad(x_k, u_k)
    f2 = dyn_func_ad(x_k + 0.5 * h * f1, u_k)
    f3 = dyn_func_ad(x_k + 0.5 * h * f2, u_k)
    f4 = dyn_func_ad(x_k + h * f3, u_k)
    return x_k + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


def step_solve_ivp(
    dyn_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray],
    h: float,
    x_k: npt.NDArray[np.float64],
    u_k: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """**Perform single step integration using scipy initial value problem**"""
    time_span = (0, h)
    dyn_func_zoh = lambda time_dyn, state: dyn_func(state, u_k.reshape(-1))
    y_0 = (x_k).reshape(-1)
    result_ivp = solve_ivp(dyn_func_zoh, time_span, y_0)
    x_kp1 = (np.array([result_ivp.y[:, -1]])).reshape(-1, 1)
    return x_kp1


def _combine_x_u_for_cost_scan(x_seq: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Combine x and u vectors into single jnp array of dim(len_seq, n+m). u_seq is
    appended with a vector of zeros to match axis 0 dimension of the two input arrays.
    This allows for proper concatenation of the arrays. \n
    - x_seq [in] - state vector sequence, dim(len_seq,n) \n
    - u_seq [in] - control vector sequence, dim(len_seq-1, m) \n
    - return x_u_seq - combined array sequence \n
    """
    u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    return jnp.concatenate([x_seq, u_seq_mod], axis=1)


def _cost_for_calc_scan_func(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ],
    x_len: int,
    carry: Tuple[int, jnp.ndarray],
    xu_k,
) -> Tuple[Tuple[int, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """ """
    count, cost_sum = carry
    (total_cost, x_cost, u_cost) = cost_func(xu_k[:x_len], xu_k[x_len:], carry[0])
    new_carry = (count + 1, cost_sum + total_cost.reshape(-1))
    return new_carry, (total_cost, x_cost, u_cost)


def _curry_cost_for_calc_scan_func(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ],
    x_len: int,
) -> Callable[
    [Tuple[int, jnp.ndarray], jnp.ndarray],
    Tuple[Tuple[int, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
]:
    def scan_func(carry: Tuple[int, jnp.ndarray], xu_k: jnp.ndarray):
        return _cost_for_calc_scan_func(cost_func, x_len, carry, xu_k)

    return scan_func


def calculate_total_cost_pre_decorated(
    cost_for_calc_scan_func: Callable[
        [Tuple[int, jnp.ndarray], jnp.ndarray],
        Tuple[Tuple[int, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    **Calculate the x and u sequence cost using a pre-decorated cost function for lax.scan**
    """
    xu_seq = _combine_x_u_for_cost_scan(x_seq, u_seq)
    init_carry: Tuple[int, jnp.ndarray] = (0, jnp.zeros([1], dtype=float))
    (_, total_cost), (total_cost_seq, x_cost_seq, u_cost_seq) = lax.scan(
        cost_for_calc_scan_func, init_carry, xu_seq
    )
    return total_cost.item(), total_cost_seq, x_cost_seq, u_cost_seq


def calculate_total_cost(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    **Calculate the x and u sequence cost from provided undecorated cost function**
    This implementation is more flexible but slower than its counterpart:
    calculate_total_cost_pre_decorated()
    """
    xu_seq = _combine_x_u_for_cost_scan(x_seq, u_seq)
    init_carry: Tuple[int, jnp.ndarray] = (0, jnp.zeros([1], dtype=float))
    scan_cost_func_for_calc_curried = _curry_cost_for_calc_scan_func(
        cost_func, x_seq.shape[1]
    )
    (_, total_cost), (total_cost_seq, x_cost_seq, u_cost_seq) = lax.scan(
        scan_cost_func_for_calc_curried, init_carry, xu_seq
    )
    return total_cost.item(), total_cost_seq, x_cost_seq, u_cost_seq


def taylor_expand_cost(
    cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    xu_k: jnp.ndarray,
    x_k_len: int,
    k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This function creates a quadratic approximation of the cost function Using taylor expansion.
    Expansion is approximated about the rolled out trajectory
    """
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu: Callable[[jnp.ndarray], jnp.ndarray] = lambda xu_k: cost_func(
        xu_k[:x_k_len], xu_k[x_k_len:], k
    )
    # calculate the concatenated jacobian vector for the cost function at the primal point
    jac_cat = (jax.jacfwd(cost_func_xu)(xu_k)).reshape(len(xu_k), 1)
    # calculate the lumped hessian matrix for the cost function at the primal point
    hes_cat = (jax.jacfwd(jax.jacrev(cost_func_xu))(xu_k)).reshape(len(xu_k), len(xu_k))
    l_x = jac_cat[:x_k_len]
    l_u = jac_cat[x_k_len:]
    l_xx = hes_cat[:x_k_len, :x_k_len]
    l_uu = hes_cat[x_k_len:, x_k_len:]
    l_ux = hes_cat[x_k_len:, :x_k_len]
    return l_x, l_u, l_xx, l_uu, l_ux


def _cost_for_exp_scan_func(
    cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    x_len: int,
    carry: int,
    xu_k: jnp.ndarray,
) -> Tuple[int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, xu_k, x_len, carry)
    carry += 1
    return carry, (l_x, l_u, l_xx, l_uu, l_ux)


def _curry_cost_for_exp_scan_func(
    cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray], x_len: int
) -> Callable[
    [int, jnp.ndarray],
    Tuple[int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
]:
    def scan_func(
        carry: int, xu_k: jnp.ndarray
    ) -> Tuple[
        int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ]:
        return _cost_for_exp_scan_func(cost_func, x_len, carry, xu_k)

    return scan_func


def taylor_expand_cost_seq_pre_decorated(
    cost_for_exp_scan_func: Callable[
        [int, jnp.ndarray],
        Tuple[
            int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ],
    ],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    xu_seq = jnp.concatenate([x_seq, u_seq_mod], axis=1)
    carry: int = 0
    carry, expanded_mats = lax.scan(cost_for_exp_scan_func, carry, xu_seq)
    return expanded_mats


def taylor_expand_cost_seq(
    cost_func: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    xu_seq = jnp.concatenate([x_seq, u_seq_mod], axis=1)
    carry: int = 0
    lin_scan_func_curried = lambda carry, xu_seq: _cost_for_exp_scan_func(
        cost_func, x_seq.shape[1], carry, xu_seq
    )
    carry, expanded_mats = lax.scan(lin_scan_func_curried, carry, xu_seq)
    return expanded_mats


def prep_cost_func_for_diff(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ]
) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    """
    Function for reducing generic cost function output to single queried value.\n
    useful for preparing functions for jax differentiation
    """

    def cost_func_for_diff(
        x_k: jnp.ndarray, u_k: jnp.ndarray, k_step: int
    ) -> jnp.ndarray:
        cost_out = cost_func(x_k, u_k, k_step)
        return cost_out[0]

    return cost_func_for_diff


def prep_cost_func_for_calc(
    cost_func: Callable[
        [jnp.ndarray, jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ]
) -> Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], int], Tuple[float, float, float]
]:

    def cost_func_for_calc(
        x_k: npt.NDArray[np.float64], u_k: npt.NDArray[np.float64], k: int
    ) -> Tuple[float, float, float]:
        total_cost, x_cost, u_cost = cost_func(jnp.array(x_k), jnp.array(u_k), k)
        return total_cost.item(), x_cost.item(), u_cost.item()

    return cost_func_for_calc


def calculate_ff_fb_u(
    k_fb_k: jnp.ndarray, u_ff_k: jnp.ndarray, x_des_k: jnp.ndarray, x_k: jnp.ndarray
) -> jnp.ndarray:
    return (
        u_ff_k.reshape(-1, 1) + k_fb_k @ (x_k.reshape(-1, 1) - x_des_k.reshape(-1, 1))
    ).reshape(-1)


def apply_cov_noise_to_vec(vec: jnp.ndarray, cov_mat: jnp.ndarray) -> jnp.ndarray:
    return vec + jax.random.multivariate_normal(
        cov_mat.shape[0], np.zeros(cov_mat.shape[0]), cov_mat
    )


######################################################################################################


def ss_2_dyn_func(
    ss_cont: stateSpace,
) -> Tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], bool]:
    if ss_cont.type == "continuous":
        is_continous_bool = True
    else:
        is_continous_bool = False
    return (
        lambda x, u: ss_cont.a @ x.reshape(-1, 1) + ss_cont.b @ u.reshape(-1, 1)
    ), is_continous_bool


def calculate_s_xx_dot(
    Q_t: npt.NDArray[np.float64],
    A_t: npt.NDArray[np.float64],
    BRinvBT_t: npt.NDArray[np.float64],
    s_xx_t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    s_xx_dt = -(Q_t - (s_xx_t @ BRinvBT_t @ s_xx_t) + (s_xx_t @ A_t) + (A_t.T @ s_xx_t))
    return s_xx_dt


def simulate_forward_dynamics_seq_simple(
    discrete_dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_init: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate discrete forward dynamics sequence from initial state with contro sequence and discrete dynamics function
    """
    len_seq = len(u_seq)
    x_seq = jnp.zeros([(len_seq + 1), len(x_init)], dtype=float)
    x_seq[0] = x_init  # type: ignore
    for idx in range(len_seq):
        x_seq.at[idx + 1].set(discrete_dyn_func(x_seq[idx], u_seq[idx]))  # type: ignore
    return x_seq


def c2d_AB_zoh_combined(
    A: jnp.ndarray, B: jnp.ndarray, time_step: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n = A.shape[0]
    m = B.shape[1]
    a_b_c = jnp.concatenate(
        (jnp.concatenate((A, B), axis=1), jnp.zeros((m, n + m))), axis=0
    )
    a_b_d = jax.scipy.linalg.expm(a_b_c * time_step)
    Ad = a_b_d[:n, :n]
    Bd = a_b_d[:n, n:]
    return Ad, Bd


def convert_d2d_A_B(
    Ad_1: jnp.ndarray, Bd_1: jnp.ndarray, dt_1: float, dt_2: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    I = jnp.eye(Ad_1.shape[0])
    A = jnp.array((1 / dt_1) * scipy.linalg.logm(Ad_1))
    A_inv = jnp.linalg.inv(A)
    B = (jnp.linalg.inv(A_inv @ (jnp.linalg.inv(Ad_1 - I)))) @ Bd_1
    Ad_2 = calculate_matrix_power_similarity_transform(Ad_1, dt_2 / dt_1)
    Bd_2 = A_inv @ (Ad_2 - I) @ B
    return Ad_2, Bd_2


def calculate_matrix_power_similarity_transform(
    A: jnp.ndarray, exponent: float
) -> jnp.ndarray:
    evals, evecs = jnp.linalg.eig(A)
    D = jnp.diag(evals)
    evecs_inv = jnp.linalg.inv(evecs)
    A_to_exponent = evecs @ D**exponent @ evecs_inv
    return A_to_exponent


######################################################################################################


def simulate_forward_dynamics_step(
    dyn_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    time_step: float,
    sim_method: str = "rk4",
) -> jnp.ndarray:
    """
    **This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory**\n
    - dyn_func[in]   - continuous dynamics function (pre-populated with *params) for integration\n
    - u_k[in]        - control input (scalar or vector depending on control dimension -- must match dyn_func control dim)\n
    - x_k[in]        - starting state for integration (vector, must match dyn_func state dim)\n
    - time_step[in]  - time interval between sequence points (scalar > 0)\n
    - sim_method[in] - step integration method, must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'\n
    - x_kp1[out]     - array shape[1,state_dim] of state vector at next time step
    """
    if sim_method == "rk4":
        x_kp1 = step_rk4(dyn_func, time_step, x_k, u_k)
    elif sim_method == "euler":
        x_kp1 = x_k + dyn_func(x_k, u_k) * time_step
    elif sim_method == "solve_ivp_zoh":
        time_span = (0, time_step)
        dyn_func_zoh = lambda time_dyn, state: dyn_func(state, u_k.reshape(-1))
        y_0 = (x_k).reshape(-1)
        result_ivp = solve_ivp(dyn_func_zoh, time_span, y_0)
        x_kp1 = jnp.array(result_ivp.y[:, -1])
    else:
        raise ValueError("invalid simulation method")
    return x_kp1.reshape(-1)


def discretize_continuous_state_space(
    input_state_space: stateSpace, time_step: float, c2d_method: str = "euler"
) -> stateSpace:
    """
    **Function for discretizing continuous state space with different discretization methods** \n
    - A - (nxn) - continuous state transition matrix \n
    - B - (nxm) - continuous control matrix \n
    - C - (pxn) - continuous state measurement matrix \n
    - D - (pxm) - continuous direct feedthrough matrix \n
    Continuous state space: \n
    - xdot(t) = A * x(t) + B * u(t) \n
    - y(t)    = C * x(t) + D * u(t) \n
    transformation for zohCombined: \n
    - e^([[A, B],[0,0]] * time_step) = [[Ad, Bd],[0,I]] \n
    Discrete state space: \n
    - x[t+timestep] = Ad * x[t] + Bd * u[t] \n
    - y[t]          = Cd * x[t] + Dd * u[t] \n
    """
    if input_state_space.type == "discrete":
        raise TypeError("input state space is already discrete")
    else:
        a_d = jnp.zeros(input_state_space.a.shape)
        b_d = jnp.zeros(input_state_space.b.shape)
        c_d = jnp.zeros(input_state_space.c.shape)
        d_d = jnp.zeros(input_state_space.d.shape)
        n = input_state_space.a.shape[0]
        m = input_state_space.b.shape[1]
        p = input_state_space.c.shape[0]
        if c2d_method == "euler":
            a_d = jnp.eye(n) + (input_state_space.a * time_step)
            b_d = input_state_space.b * time_step
            c_d = input_state_space.c
            d_d = input_state_space.d
        elif c2d_method == "zoh":
            if jax.scipy.linalg.det(input_state_space.a) > 10e-8:
                a_d = jax.scipy.linalg.expm(input_state_space.a * time_step)
                b_d = (
                    jnp.linalg.inv(input_state_space.a)
                    @ (a_d - jnp.eye(n))
                    @ input_state_space.b
                )
                c_d = input_state_space.c
                d_d = input_state_space.d
            else:
                raise ValueError(
                    "determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid"
                )
        elif c2d_method == "zohCombined":
            #   create combined A B matrix e^([[A, B],[0,0]]
            a_b_c: jnp.ndarray = jnp.concatenate(
                (
                    jnp.concatenate((input_state_space.a, input_state_space.b), axis=1),
                    jnp.zeros((m, n + m)),
                ),
                axis=0,
            )
            a_b_d: jnp.ndarray = jax.scipy.linalg.expm(a_b_c * time_step)
            a_d = a_b_d[:n, :n]
            b_d = a_b_d[:n, n:]
            c_d = input_state_space.c
            d_d = input_state_space.d
        else:
            raise ValueError("invalid discretization method")
    d_state_space = stateSpace(a_d, b_d, c_d, d_d, time_step)
    return d_state_space
