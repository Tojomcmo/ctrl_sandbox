import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
from typing import Callable, Tuple
from numpy import typing as npt

import ctrl_sandbox.integrate_funcs as integrate
import ctrl_sandbox.gen_typing as gt


def _dyn_for_sim_scan_func(
    discrete_dyn_func: gt.DynFuncType,
    carry_x_k: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> gt.JaxArrayTuple2:
    """
    **Kernel scan function for forward dynamics simulation**
     - This function decorates the discrete_dyn_func for
        jax.lax.scan by propagating carry and seq values as returns
    """
    x_kp1 = discrete_dyn_func(carry_x_k, u_seq)
    return (x_kp1, x_kp1)


def _curry_dyn_for_sim_scan_func(
    discrete_dyn_func: gt.DynFuncType,
) -> Callable[[jnp.ndarray, jnp.ndarray], gt.JaxArrayTuple2]:
    """
    **Create lax.scan compatible function for forward
        dynamic simulation through partial argument reduction**
    """

    def scan_func(carry_x_k: jnp.ndarray, u_seq: jnp.ndarray) -> gt.JaxArrayTuple2:
        return _dyn_for_sim_scan_func(discrete_dyn_func, carry_x_k, u_seq)

    return scan_func


def simulate_forward_dynamics_seq_pre_decorated(
    sim_dyn_scan_func: Callable[[jnp.ndarray, jnp.ndarray], gt.JaxArrayTuple2],
    x_init: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """**Forward simulate a control sequence from an initial state
            through a discrete dynamics function** \n
    - This function uses jax.lax.scan to efficiently generate the paired
        x_seq to the provided u_seq throughsequential calculation of
        the discrete dynamics provided. \n
    - [in] discrete_dyn_func - discrete dynamics function
        (x_k,u_k)->(x_kp1), **must be jax compatible** \n
    - [in] x_init - initial state vector, dimension (n,) \n
    - [in] u_seq - control sequence, dimension(len_seq-1,m) ,
        length len_seq-1 because the final state has no control value \n
    - [return] x_seq - state sequence, dimension(len_seq,n)
    """
    _, x_seq_out = lax.scan(sim_dyn_scan_func, x_init, u_seq)
    return jnp.append(x_init.reshape(1, -1), x_seq_out, axis=0)


def simulate_forward_dynamics_seq(
    discrete_dyn_func: gt.DynFuncType, x_init: jnp.ndarray, u_seq: jnp.ndarray
) -> jnp.ndarray:
    """
    **Simulate forward dynamics from initial state through
        control sequence with undecorated dynamics function** \n
    - Wrapper function for simulate_forward_dynamics_seq_pre_decorated().
    - Simpler interface by accepting dynamics function directly
    - Slower in iteration due to the new function definition on each call.
    """
    discrete_dyn_scan_func = lambda carry_x_k, u_seq: _dyn_for_sim_scan_func(
        discrete_dyn_func, carry_x_k, u_seq
    )
    return simulate_forward_dynamics_seq_pre_decorated(
        discrete_dyn_scan_func, x_init, u_seq
    )


def linearize_dynamics(
    dyn_func: gt.DynFuncType, xu_k: jnp.ndarray, x_k_len
) -> gt.JaxArrayTuple2:
    """
    **Linearizes the dynamics function about the primals x and u** \n
    - [in] dyn_func - dynamics function,
        may be continuous or discrete, **must be jacfwd compatible** \n
    - [in]xu_k - concatenated primal state and control
        linearization point, **NOT a sequence** \n
    - [in] x_k_len - state vector length,
        for splitting linearization into respective parts \n
    - [ret] A_lin - jacobian of dyn_func wrt x eval at x_k,u_k,
        continuous or discrete time\n
    - [ret] B_lin - jacobian of dyn_func wrt x eval at x_k,u_k,
        continuous or discrete time\n
    """
    dyn_func_xu: Callable[[jnp.ndarray], jnp.ndarray] = lambda xu_k: dyn_func(
        xu_k[:x_k_len], xu_k[x_k_len:]
    )
    ab_lin_cat = jax.jacfwd(dyn_func_xu)(xu_k)
    a_lin = ab_lin_cat[:x_k_len, :x_k_len]
    b_lin = ab_lin_cat[:x_k_len, x_k_len:]
    return a_lin, b_lin


def _dyn_lin_scan_func(
    dyn_func: gt.DynFuncType, x_len: int, carry: None, x_and_u_k: jnp.ndarray
) -> Tuple[None, gt.JaxArrayTuple2]:
    """
    **Partially format linearize dynamics function for lax.scan**
    - [in] dyn_func - callable dynamics function,
        usually discrete dynamics given discrete sequences \n
    - [in] x_len - int length of state vector \n
    - [in] carry - Nonetype, necessary for lax.scan \n
    - [in] x_and_u_k - jnp array of dim(n+m,),
        single vector from seqs provided to lax.scan \n

    This function must be further partialed to properly function
    with lax.scan to only accept arguments (carry, seqs)
    """
    a_lin, b_lin = linearize_dynamics(dyn_func, x_and_u_k, x_len)
    return carry, (a_lin, b_lin)


def _curry_dyn_lin_scan_func(
    dyn_func: gt.DynFuncType, x_len: int
) -> Callable[[None, jnp.ndarray], Tuple[None, gt.JaxArrayTuple2]]:
    """
    **Create curried scan function for lax.scan for
    linearizing a dynamic sequence of x and u states
    through a dyanmics function** \n
    - [in] discrete_dyn_func - callable discrete dynamics function
    - [in] x_len - length of state vector,
            for splitting linearization into x and u jacobians
    - [ret] scan_func_curried - fully formatted function for lax.scan

    This function returns a fully formatted function for lax.scan
    operation for dynamic sequence linearization.
    """

    def dyn_lin_scan_func_curried(carry: None, xu_k: jnp.ndarray):
        carry_out, A_B_k = _dyn_lin_scan_func(dyn_func, x_len, carry, xu_k)
        return carry_out, A_B_k

    return dyn_lin_scan_func_curried


def calculate_lin_state_space_seq_pre_decorated(
    dyn_lin_scan_func: Callable[[None, jnp.ndarray], Tuple[None, gt.JaxArrayTuple2]],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> gt.JaxArrayTuple2:
    """
    **Calculate jacobian matrices of paired state and control sequences**
    - [in] lin_scan_func - lax.scan compatible dynamics linearization func
    - [in] x_seq - state linearization sequence, dim(len_seq, n)
    - [in] u_seq - control linearization sequence, dim(len_seq-1, m)

    The lin_scan_func must accept:
    - carry of Nonetype
    - seqs of jnp.ndarray of size (len_seq-1, n+m).
        len_seq-1 because the final timestep has no corresponding
        dynamics.
    - (see _dyn_lin_scan_func)
    """
    xu_seq = jnp.concatenate((x_seq[:-1], u_seq), axis=1)
    carry = None
    _, (a_lin_seq, b_lin_seq) = lax.scan(dyn_lin_scan_func, carry, xu_seq)
    return a_lin_seq, b_lin_seq


def calculate_linearized_state_space_seq(
    dyn_func: gt.DynFuncType, x_seq: jnp.ndarray, u_seq: jnp.ndarray
) -> gt.JaxArrayTuple2:
    """
    **Calculate the A and B matrix linearizations for
    x and u through undecorated dynamics function** \n
    - Wrapper function for
        calculate_linearized_state_space_seq_pre_decorated().
    - Simpler interface by accepting dynamics function directly
    - Slower in iteration due to the
        new function definition on each call.
    """
    lin_scan_func = _curry_dyn_lin_scan_func(dyn_func, x_seq.shape[1])
    return calculate_lin_state_space_seq_pre_decorated(lin_scan_func, x_seq, u_seq)


def _combine_x_u_for_cost_scan(x_seq: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Combine x and u vectors into single jnp array of dim(len_seq, n+m). u_seq is
    appended with a vector of zeros to match axis 0 dimension of the two input arrays.
    This allows for proper concatenation of the arrays. \n
    - [in] x_seq - state vector sequence, dim(len_seq,n) \n
    - [in] u_seq - control vector sequence, dim(len_seq-1, m) or dim(len_seq,m) \n
    - [ret] x_u_seq - combined array sequence \n
    """
    if u_seq.shape[0] == (x_seq.shape[0] - 1):
        u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
        u_seq_app = jnp.append(u_seq, u_seq_end, axis=0)
    else:
        u_seq_app = u_seq
    return jnp.concatenate([x_seq, u_seq_app], axis=1)


def _cost_for_calc_scan_func(
    cost_func: gt.CostFuncType,
    x_len: int,
    carry: Tuple[int, jnp.ndarray],
    xu_k: jnp.ndarray,
) -> Tuple[Tuple[int, jnp.ndarray], gt.JaxArrayTuple3]:
    """
    ** Kernel function for lax.scan calculation of
        sequence cost and individual cost sequences**
    - [in] cost_func - callable cost function that accepts (x_k, u_k, k) and
        returns the total and individual costs as jnp.ndarrays of dim(1,)
    - [in] x_len - length of x_vector, dim n (not sequence length len_seq)
    - [in] carry - Tuple of sequence index and cost sum,
        serves as carry argument for lax.scan
    - [in] xu_k - combined x and u sequences in jnp.ndarray dim(len_seq, n+m),
        serves as seq argument for lax.scan
    """
    k, cost_sum = carry
    (total_cost, x_cost, u_cost) = cost_func(xu_k[:x_len], xu_k[x_len:], k)
    new_carry = (k + 1, cost_sum + total_cost.reshape(-1))
    return new_carry, (total_cost, x_cost, u_cost)


def _curry_cost_for_calc_scan_func(
    cost_func: gt.CostFuncType,
    x_len: int,
) -> Callable[
    [Tuple[int, jnp.ndarray], jnp.ndarray],
    Tuple[Tuple[int, jnp.ndarray], gt.JaxArrayTuple3],
]:
    """
    **Create lax.scan compatible function for cost calculation**
    """

    def scan_func(carry: Tuple[int, jnp.ndarray], xu_k: jnp.ndarray):
        return _cost_for_calc_scan_func(cost_func, x_len, carry, xu_k)

    return scan_func


def calculate_total_cost_pre_decorated(
    cost_for_calc_scan_func: Callable[
        [Tuple[int, jnp.ndarray], jnp.ndarray],
        Tuple[Tuple[int, jnp.ndarray], gt.JaxArrayTuple3],
    ],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> gt.JaxArrayTuple4:
    """
    **Calculate sequence cost using pre-decorated cost function for lax.scan**
    """
    xu_seq = _combine_x_u_for_cost_scan(x_seq, u_seq)
    init_carry: Tuple[int, jnp.ndarray] = (0, jnp.zeros([1], dtype=float))
    (_, total_cost), (total_cost_seq, x_cost_seq, u_cost_seq) = lax.scan(
        cost_for_calc_scan_func, init_carry, xu_seq
    )
    return (
        total_cost.reshape(-1),
        total_cost_seq.reshape(-1),
        x_cost_seq.reshape(-1),
        u_cost_seq.reshape(-1),
    )


def calculate_total_cost(
    cost_func: gt.CostFuncType, x_seq: jnp.ndarray, u_seq: jnp.ndarray
) -> gt.JaxArrayTuple4:
    """
    **Calculate sequence cost from provided undecorated cost function** \n
    - Wrapper function for calculate_total_cost_pre_decorated().
    - Simpler interface by accepting the cost function directly
    - Slower in iteration due to the new function definition on each call.
    """
    scan_func = _curry_cost_for_calc_scan_func(cost_func, x_seq.shape[1])
    return calculate_total_cost_pre_decorated(scan_func, x_seq, u_seq)


def taylor_expand_cost(
    cost_func: gt.CostFuncDiffType, xu_k: jnp.ndarray, x_k_len: int, k: int
) -> gt.JaxArrayTuple5:
    """
    **Create cost function quadratic approximation using taylor expansion.** \n
    Expansion is approximated about the rolled out trajectory
    """
    # create lambda function of reduced inputs to only a single vector
    cost_func_xu: Callable[[jnp.ndarray], jnp.ndarray] = lambda xu_k: cost_func(
        xu_k[:x_k_len], xu_k[x_k_len:], k
    )
    # calculate jacobian vectors for the cost function at the primal point
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
    cost_func: gt.CostFuncDiffType, x_len: int, k: int, xu_k: jnp.ndarray
) -> Tuple[int, gt.JaxArrayTuple5]:
    """
    ** kernel lax.scan function for cost function second order taylor expansion**
    - [in] cost_func - callable that accepts (x_k, u_k, k) and
        returns cost as a jnp.ndarray of dim(1,) type float
    - [in] x_len - length of x vector
    - [in] k - int index of sequence, serves as carry variable in lax.scan
    - [in] xu_k - paired x and u sequences as jnp.ndarrays
        of dim (len_seq, n+m), u_seq[-1] must be vector of zeros
    - [ret] kp1 - sequence index incremented by 1
    - [ret] (expanded_mats) - taylor expanded matrices around xu_k primal point.
        refer to taylor_expand_cost() for details
    """
    l_x, l_u, l_xx, l_uu, l_ux = taylor_expand_cost(cost_func, xu_k, x_len, k)
    kp1 = k + 1
    return kp1, (l_x, l_u, l_xx, l_uu, l_ux)


def _curry_cost_for_exp_scan_func(
    cost_func: gt.CostFuncDiffType, x_len: int
) -> Callable[[int, jnp.ndarray], Tuple[int, gt.JaxArrayTuple5]]:
    """
    **Create lax.scan compatible function for cost function taylor expansion**
    """

    def scan_func(
        carry: int, xu_k: jnp.ndarray
    ) -> Tuple[
        int, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ]:
        return _cost_for_exp_scan_func(cost_func, x_len, carry, xu_k)

    return scan_func


def taylor_expand_cost_seq_pre_decorated(
    cost_for_exp_scan_func: Callable[[int, jnp.ndarray], Tuple[int, gt.JaxArrayTuple5]],
    x_seq: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> gt.JaxArrayTuple5:
    """
    **Calculate the second order taylor expansion of provided
        undecorated cost function at x and u sequence primal points** \n
    """
    u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    xu_seq = jnp.concatenate([x_seq, u_seq_mod], axis=1)
    carry: int = 0
    carry, expanded_mats = lax.scan(cost_for_exp_scan_func, carry, xu_seq)
    return expanded_mats


def taylor_expand_cost_seq(
    cost_func: gt.CostFuncDiffType, x_seq: jnp.ndarray, u_seq: jnp.ndarray
) -> gt.JaxArrayTuple5:
    """
    **Calculate the second order taylor expansion of provided
        undecorated cost function at x and u sequence primal points** \n
    - Wrapper function for taylor_expand_cost_seq_pre_decorated().
    - Simpler interface by accepting the cost function directly
    - Slower in iteration due to the new function definition on each call.
    """
    u_seq_end = jnp.zeros((1, u_seq.shape[1]), dtype=float)
    u_seq_mod = jnp.append(u_seq, u_seq_end, axis=0)
    xu_seq = jnp.concatenate([x_seq, u_seq_mod], axis=1)
    carry: int = 0
    lin_scan_func_curried = lambda carry, xu_seq: _cost_for_exp_scan_func(
        cost_func, x_seq.shape[1], carry, xu_seq
    )
    carry, expanded_mats = lax.scan(lin_scan_func_curried, carry, xu_seq)
    return expanded_mats


def prep_cost_func_for_diff(cost_func: gt.CostFuncType) -> gt.CostFuncDiffType:
    """
    Function for reducing generic cost function output to single queried value.\n
    Useful for preparing functions for jax differentiation
    """

    def cost_func_for_diff(
        x_k: jnp.ndarray, u_k: jnp.ndarray, k_step: int
    ) -> jnp.ndarray:
        cost_out = cost_func(x_k, u_k, k_step)
        return cost_out[0]

    return cost_func_for_diff


def prep_cost_func_for_calc(cost_func: gt.CostFuncType) -> gt.CostFuncFloatType:
    """
    **function wrapper on cost function for simple cost calculation, returning floats**
    """

    def cost_func_for_calc(
        x_k: gt.npORjnpArr, u_k: gt.npORjnpArr, k: int
    ) -> Tuple[float, float, float]:
        total_cost, x_cost, u_cost = cost_func(jnp.array(x_k), jnp.array(u_k), k)
        return total_cost.item(), x_cost.item(), u_cost.item()

    return cost_func_for_calc


def calculate_ff_fb_u(
    k_fb_k: jnp.ndarray, u_ff_k: jnp.ndarray, x_des_k: jnp.ndarray, x_k: jnp.ndarray
) -> jnp.ndarray:
    """
    **calculate control value at discrete time k from feedforward-feedback structure**
    - [in] k_fb_k - feedback gain matrix at time k, dim(m,n)
        providing control interaction with state error
    - [in] u_ff_k - feedforward/nominal control value at time k,
         prescribed trajectory control sequence
    - [in] x_des_k - desired/target state trajectory at time k,
        prescribed desired state
    - [in] x_k - actual/measured/estimated state at time k,
        differenced with x_des_k to create vectored error value
    - [out] u_k - control value for time k from combined feedforward and feedback terms
    """
    return (
        u_ff_k.reshape(-1, 1) + k_fb_k @ (x_k.reshape(-1, 1) - x_des_k.reshape(-1, 1))
    ).reshape(-1)


def apply_cov_noise_to_vec(vec: jnp.ndarray, cov_mat: jnp.ndarray) -> jnp.ndarray:
    """
    **apply zero-mean multivariate gaussian white noise of cov_mat covariance to vec**
    - [in] vec - jax array of dim (n,)
    - [in] cov_mat - jax array of covariance matrix dim(n,n)
    - [out] vec+noise - jax array of provided vector with noise superimposed
    """

    return vec + jax.random.multivariate_normal(
        cov_mat.shape[0], np.zeros(cov_mat.shape[0]), cov_mat
    )


#######################################################################################


def calculate_s_xx_dot(
    Q_t: npt.NDArray[np.float64],
    A_t: npt.NDArray[np.float64],
    BRinvBT_t: npt.NDArray[np.float64],
    s_xx_t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    s_xx_dt = -(Q_t - (s_xx_t @ BRinvBT_t @ s_xx_t) + (s_xx_t @ A_t) + (A_t.T @ s_xx_t))
    return s_xx_dt


def simulate_forward_dynamics_seq_simple(
    discrete_dyn_func: gt.DynFuncType,
    x_init: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate discrete forward dynamics sequence from initial state
        with control sequence and discrete dynamics function
    """
    len_seq = len(u_seq)
    x_seq = jnp.zeros([(len_seq + 1), len(x_init)], dtype=float)
    x_seq[0] = x_init  # type: ignore
    for idx in range(len_seq):
        x_seq.at[idx + 1].set(discrete_dyn_func(x_seq[idx], u_seq[idx]))  # type: ignore
    return x_seq


def simulate_forward_dynamics_step(
    dyn_func: gt.DynFuncType,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    time_step: float,
    sim_method: str = "rk4",
) -> jnp.ndarray:
    """
    **This function integrates the nonlinear dynamics forward
        to formulate the corresponding trajectory**\n
    - dyn_func[in]   - continuous dynamics function
        (pre-populated with *params) for integration\n
    - u_k[in]        - control input (scalar or vector depending on control dimension,
        must match dyn_func control dim)\n
    - x_k[in]        - init state for integration, dim(n,),
        must match dyn_func state dim)\n
    - time_step[in]  - time interval between sequence points (scalar > 0)\n
    - sim_method[in] - step integration method,
        must be 'rk4'(default), 'euler', or 'solve_ivp_zoh'\n
    - x_kp1[out]     - array shape[1,state_dim] of state vector at next time step
    """
    if sim_method == "rk4":
        x_kp1 = integrate.step_rk4(dyn_func, time_step, x_k, u_k)
    elif sim_method == "euler":
        x_kp1 = x_k + dyn_func(x_k, u_k) * time_step
    elif sim_method == "solve_ivp_zoh":
        x_kp1 = jnp.ndarray(integrate.step_solve_ivp(dyn_func, time_step, x_k, u_k))
    else:
        raise ValueError("invalid simulation method")
    return x_kp1.reshape(-1)
