from jax import numpy as jnp
from jax import lax
import numpy as np
import numpy.typing as npt
from typing import Tuple, Union, Callable

# library of cost functions and associated functions for manipulating cost functions
# Cost functions may return multiple values:
#  - first value MUST be the float value of calculated query cost

CostFuncType = Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]


class cost_quad_x_and_u:
    def __init__(
        self,
        Q: Union[jnp.ndarray, npt.NDArray[np.float64]],
        R: Union[jnp.ndarray, npt.NDArray[np.float64]],
        Qf: Union[jnp.ndarray, npt.NDArray[np.float64]],
        x_des_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
        u_des_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    ) -> None:
        self.Q = jnp.array(Q)
        self.R = jnp.array(R)
        self.Qf = jnp.array(Qf)
        self.x_des_seq = jnp.array(x_des_seq)
        self.u_des_seq = jnp.array(u_des_seq)
        self.len_seq: int = len(x_des_seq)

    def validate_inputs(self):
        # TODO add validation criteria
        pass

    def first_index_case(self, args: Tuple[jnp.ndarray, jnp.ndarray, int]):
        x_k, u_k, k = args
        u_k_corr: jnp.ndarray = u_k.reshape(-1, 1) - (self.u_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array([[0.0]])
        u_cost = jnp.array((0.5) * (u_k_corr.T @ self.R @ u_k_corr))
        total_cost = u_cost
        return total_cost.reshape(-1), x_cost.reshape(-1), u_cost.reshape(-1)

    def last_index_case(self, args: Tuple[jnp.ndarray, jnp.ndarray, int]):
        x_k, u_k, k = args
        x_k_corr: jnp.ndarray = x_k.reshape(-1, 1) - (self.x_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array((0.5) * (x_k_corr.T @ self.Qf @ x_k_corr))
        u_cost = jnp.array([[0.0]])
        total_cost = x_cost
        return total_cost.reshape(-1), x_cost.reshape(-1), u_cost.reshape(-1)

    def default_case(self, args: Tuple[jnp.ndarray, jnp.ndarray, int]):
        x_k, u_k, k = args
        x_k_corr: jnp.ndarray = x_k.reshape(-1, 1) - (self.x_des_seq[k]).reshape(-1, 1)
        u_k_corr: jnp.ndarray = u_k.reshape(-1, 1) - (self.u_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array((0.5) * (x_k_corr.T @ self.Q @ x_k_corr))
        u_cost = jnp.array((0.5) * (u_k_corr.T @ self.R @ u_k_corr))
        total_cost = jnp.array(
            (0.5)
            * ((x_k_corr.T @ self.Q @ x_k_corr) + (u_k_corr.T @ self.R @ u_k_corr))
        )
        return total_cost.reshape(-1), x_cost.reshape(-1), u_cost.reshape(-1)

    def cost_func_quad_state_and_control_scan_compatible(
        self, x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate quadratic cost wrt state and control
        Q[in]     - State cost matrix, square(n,n) Positive semidefinite
        R[in]     - Control cost matrix, square(m,m) Positive definite
        Qf[in]    - Final state cost matrix, square(n,n) Positive semidefinite
        cost[out] - cost value Tuple [total_cost, x_cost, u_cost]
        This function is prepped for lax.scan use, and is jax diff compatible
        """
        # check that dimensions match [TODO]
        is_first_case = k == 0
        is_last_case = k == self.len_seq - 1
        return lax.cond(
            is_first_case,
            self.first_index_case,
            lambda args: lax.cond(
                is_last_case, self.last_index_case, self.default_case, args
            ),
            (x_k, u_k, k),
        )

    def cost_func_quad_state_and_control_for_calc(
        self, x_k: jnp.ndarray, u_k: jnp.ndarray, k: int
    ):
        """
        This function calculates a quadratic cost wrt state and control
        Q[in]     - State cost matrix, square(n,n) Positive semidefinite
        R[in]     - Control cost matrix, square(m,m) Positive definite
        Qf[in]    - Final state cost matrix, square(n,n) Positive semidefinite
        cost[out] - cost value Tuple [total_cost, x_cost, u_cost]
        """
        if k == 0:
            u_k_corr: jnp.ndarray = u_k.reshape(-1, 1) - (self.u_des_seq[k]).reshape(
                -1, 1
            )
            x_cost = jnp.array([[0.0]])
            u_cost = jnp.array((0.5) * (u_k_corr.T @ self.R @ u_k_corr))
            total_cost = u_cost
        elif k == self.len_seq - 1:
            x_k_corr: jnp.ndarray = x_k.reshape(-1, 1) - (self.x_des_seq[k]).reshape(
                -1, 1
            )
            x_cost = jnp.array((0.5) * (x_k_corr.T @ self.Qf @ x_k_corr))
            u_cost = jnp.array([[0.0]])
            total_cost = x_cost
            total_cost = u_cost
        else:
            x_k_corr: jnp.ndarray = x_k.reshape(-1, 1) - (self.x_des_seq[k]).reshape(
                -1, 1
            )
            u_k_corr: jnp.ndarray = u_k.reshape(-1, 1) - (self.u_des_seq[k]).reshape(
                -1, 1
            )
            x_cost = jnp.array((0.5) * (x_k_corr.T @ self.Q @ x_k_corr))
            u_cost = jnp.array((0.5) * (u_k_corr.T @ self.R @ u_k_corr))
            total_cost = jnp.array(
                (0.5)
                * ((x_k_corr.T @ self.Q @ x_k_corr) + (u_k_corr.T @ self.R @ u_k_corr))
            )
        return total_cost, x_cost, u_cost


####################################################################################


class costFuncParams:
    def __init__(self, x_des_seq: jnp.ndarray, u_des_seq: jnp.ndarray) -> None:
        self.x_des_seq = x_des_seq
        self.u_des_seq = u_des_seq
        self.len_seq = len(x_des_seq)


class costFuncQuadStateAndControlParams(costFuncParams):
    def __init__(
        self,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        Qf: jnp.ndarray,
        x_des_seq: jnp.ndarray,
        u_des_seq: jnp.ndarray,
    ) -> None:
        super().__init__(x_des_seq, u_des_seq)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        # TODO create validate inputs function


def cost_func_quad_state_and_control(
    cost_func_params: costFuncQuadStateAndControlParams,
    x_k: jnp.ndarray,
    u_k: jnp.ndarray,
    k: int,
):
    """
    This function calculates a quadratic cost wrt state and control
    Q[in]     - State cost matrix, square(n,n) Positive semidefinite
    R[in]     - Control cost matrix, square(m,m) Positive definite
    Qf[in]    - Final state cost matrix, square(n,n) Positive semidefinite
    cost[out] - cost value Tuple [total_cost, x_cost, u_cost]
    """
    Q = jnp.array(cost_func_params.Q)
    R = jnp.array(cost_func_params.R)
    Qf = jnp.array(cost_func_params.Qf)
    if k == cost_func_params.len_seq - 1:
        x_k_corr = x_k.reshape(-1, 1) - (cost_func_params.x_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array((0.5) * (x_k_corr.T @ Qf @ x_k_corr))
        u_cost = jnp.array([0.0])
        total_cost = x_cost
    elif k == 0:
        u_k_corr = u_k.reshape(-1, 1) - (cost_func_params.u_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array([0.0])
        u_cost = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
        total_cost = u_cost
    else:
        x_k_corr = x_k.reshape(-1, 1) - (cost_func_params.x_des_seq[k]).reshape(-1, 1)
        u_k_corr = u_k.reshape(-1, 1) - (cost_func_params.u_des_seq[k]).reshape(-1, 1)
        x_cost = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
        u_cost = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
        total_cost = jnp.array(
            (0.5) * ((x_k_corr.T @ Q @ x_k_corr) + (u_k_corr.T @ R @ u_k_corr))
        )
    return total_cost, x_cost, u_cost
