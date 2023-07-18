from jax import numpy as jnp

import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost

# configuration dictionary
#   - state transition function 
#   - cost function
#   - simulation method
#   - c2d method
#   - time step
#   - max iteration
#   - convergence criteria
#
# controller state struct
#   - Seed state vector
#   - Seed control array
#   - Seed cost
#   - Iteration control array
#   - Iteration state array
#   - Iteration cost
#   - Iteration number
#



if __name__ == "__main__":
   cost_func_params    = {}
   state_trans_params  = {}
   ilqr_config = {
                    'state_trans_func'   : dyn.pend_dyn_nl,
                    'state_trans_params' : state_trans_params,
                    'cost_func'          : cost.cost_func_quad_state_and_control,
                    'cost_func_params'   : cost_func_params,
                    'sim_method'       : 'euler',
                    'c2d_method'       : 'euler',
                    'max_iter'         : 100,
                    'time_step'        : 0.01,
                    'converge_crit'    : 1e-6
                 }
   
   # define cost func parameters (simple LQR being Q,R,Qf)
   print(ilqr_config['sim_method'])