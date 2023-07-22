from jax import numpy as jnp
import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost

if __name__ == "__main__":
   cost_func_params    = {}
   state_trans_params  = {}
   ilqr_config = {
                    'state_trans_func'          : dyn.pend_dyn_nl,
                    'state_trans_func_params'   : state_trans_params,
                    'cost_func'                 : cost.cost_func_quad_state_and_control,
                    'cost_func_params'          : cost_func_params,
                    'sim_method'                : 'euler',
                    'c2d_method'                : 'euler',
                    'max_iter'                  : 100,
                    'time_step'                 : 0.01,
                    'converge_crit'             : 1e-6
                 }
   
   state_init_vec = jnp.array([])
   control_init_seq = jnp.array([])


   controller_state = ilqr.ilqrControllerState(state_init_vec, control_init_seq)
   controller_state.create_time_sequence(ilqr_config['time_step'])

   # calculate initial rollout
   # 

   
