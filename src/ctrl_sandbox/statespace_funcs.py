import jax
import jax.numpy as jnp
import scipy
from typing import Callable, Tuple


import ctrl_sandbox.util_funcs as util


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
            raise ValueError("A matrix row number and B matrix row number must match")
        elif a_mat.shape[0] != c_mat.shape[1]:
            raise ValueError(
                "A matrix row number and C matrix column number must match"
            )
        elif b_mat.shape[1] != d_mat.shape[1]:
            raise ValueError(
                "B matrix column number and D matrix column number must match"
            )
        elif c_mat.shape[0] != d_mat.shape[0]:
            raise ValueError("C matrix row number and D matrix row number must match")
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
                    f"invalid time step {time_step}. Must be positive float or None",
                )


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
    Ad_2 = util.calculate_matrix_power_similarity_transform(Ad_1, dt_2 / dt_1)
    Bd_2 = A_inv @ (Ad_2 - I) @ B
    return Ad_2, Bd_2


def discretize_continuous_state_space(
    input_state_space: stateSpace, time_step: float, c2d_method: str = "euler"
) -> stateSpace:
    """
    **Function for discretizing continuous state space
        with different discretization methods** \n
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
                    "determinant of A is excessively small (<10E-8),",
                    "simple zoh method is potentially invalid",
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
