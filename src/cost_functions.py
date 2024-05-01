from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
import ilqr_utils as util
from typing import Tuple

# library of cost functions and associated functions for manipulating cost functions
# Cost functions may return multiple values:
#  - first value MUST be the float value of calculated query cost

class costFuncParams:
    def __init__(self,x_des_seq:npt.NDArray[np.float64], 
                 u_des_seq:npt.NDArray[np.float64]) -> None:
        self.x_des_seq = x_des_seq
        self.u_des_seq = u_des_seq

class costFuncQuadStateAndControlParams(costFuncParams):
    def __init__(self, 
                 Q:npt.NDArray[np.float64], 
                 R:npt.NDArray[np.float64], 
                 Qf:npt.NDArray[np.float64], 
                 x_des_seq:npt.NDArray[np.float64], 
                 u_des_seq:npt.NDArray[np.float64]) -> None:
        super().__init__(x_des_seq, u_des_seq)
        self.Q  = Q
        self.R  = R
        self.Qf = Qf
        # TODO create validate inputs function


def cost_func_quad_state_and_control(cost_func_params:costFuncQuadStateAndControlParams, 
                                    x_k:npt.NDArray[np.float64], 
                                    u_k:npt.NDArray[np.float64], 
                                    k_step:int, 
                                    is_final_bool:bool):
# This function calculates a quadratic cost wrt state and control
# Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
# R[in]     - Control cost matrix, square, dim(control, control) Positive definite
# Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
# cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
    Q  = jnp.array(cost_func_params.Q)
    R  = jnp.array(cost_func_params.R)
    Qf = jnp.array(cost_func_params.Qf)
    # x_des_seq_jax = np.array(cost_func_params.x_des_seq)
    # u_des_seq_jax = np.array(cost_func_params.u_des_seq)
    # check that dimensions match [TODO]
    if is_final_bool:
        x_k_corr = util.calc_and_shape_array_diff(x_k  , cost_func_params.x_des_seq[k_step], shape='col')
        x_cost     = jnp.array((0.5) * (x_k_corr.T @ Qf @ x_k_corr))
        u_cost     = jnp.array([0.0])
        total_cost = x_cost
    elif k_step == 0:
        u_k_corr   = util.calc_and_shape_array_diff(u_k, cost_func_params.u_des_seq[k_step], shape='col')
        x_cost     = jnp.array([0.0])    
        u_cost     = jnp.array((0.5) * (u_k_corr.T  @ R  @ u_k_corr))
        total_cost = u_cost
    else:
        x_k_corr = util.calc_and_shape_array_diff(x_k, cost_func_params.x_des_seq[k_step], shape='col')   
        u_k_corr = util.calc_and_shape_array_diff(u_k, cost_func_params.u_des_seq[k_step], shape='col')
        x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
        u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
        total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
    return total_cost, x_cost, u_cost     


class cost_quad_x_and_u():
    def __init__(self, 
                 Q:npt.NDArray[np.float64], 
                 R:npt.NDArray[np.float64], 
                 Qf:npt.NDArray[np.float64], 
                 x_des_seq:npt.NDArray[np.float64], 
                 u_des_seq:npt.NDArray[np.float64]) -> None:
        self.Q  = Q
        self.R  = R
        self.Qf = Qf
        self.x_des_seq = x_des_seq
        self.u_des_seq = u_des_seq

    def validate_inputs(self):    
        #TODO add validation criteria
        pass

    def cost_func_quad_state_and_control(self, 
                                    x_k:npt.NDArray[np.float64], 
                                    u_k:npt.NDArray[np.float64], 
                                    k_step:int, 
                                    is_final_bool:bool)->Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # This function calculates a quadratic cost wrt state and control
    # Q[in]     - State cost matrix, square, dim(state, state) Positive semidefinite
    # R[in]     - Control cost matrix, square, dim(control, control) Positive definite
    # Qf[in]    - Final state cost matrix, square, dim(state, state) Positive semidefinite
    # cost[out] - cost value calculated given Q,R,S, and supplied state and control vecs   
        Q  = jnp.array(self.Q)
        R  = jnp.array(self.R)
        Qf = jnp.array(self.Qf)
        # x_des_seq_jax = np.array(cost_func_params.x_des_seq)
        # u_des_seq_jax = np.array(cost_func_params.u_des_seq)
        # check that dimensions match [TODO]
        if is_final_bool:
            x_k_corr = util.calc_and_shape_array_diff(x_k  , self.x_des_seq[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Qf @ x_k_corr))
            u_cost     = jnp.array([0.0])
            total_cost = x_cost
        elif k_step == 0:
            u_k_corr   = util.calc_and_shape_array_diff(u_k, self.u_des_seq[k_step], shape='col')
            x_cost     = jnp.array([0.0])    
            u_cost     = jnp.array((0.5) * (u_k_corr.T  @ R  @ u_k_corr))
            total_cost = u_cost
        else:
            x_k_corr = util.calc_and_shape_array_diff(x_k, self.x_des_seq[k_step], shape='col')   
            u_k_corr = util.calc_and_shape_array_diff(u_k, self.u_des_seq[k_step], shape='col')
            x_cost     = jnp.array((0.5) * (x_k_corr.T @ Q @ x_k_corr))
            u_cost     = jnp.array((0.5) * (u_k_corr.T @ R @ u_k_corr))
            total_cost = jnp.array((0.5) * ((x_k_corr.T @ Q @ x_k_corr)+(u_k_corr.T @ R @ u_k_corr)))
        return total_cost, x_cost, u_cost   