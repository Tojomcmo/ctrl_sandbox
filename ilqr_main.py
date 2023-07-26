from jax import numpy as jnp
import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost

if __name__ == "__main__":
   cost_func_params    = {'Q'  : jnp.array([[1.,0],[0,10.]]),
                          'R'  : jnp.array([[1.]]),
                          'Qf' : jnp.array([[10.,0],[0,1.]])}
   state_trans_params  = {'b'  : 1.0,
                          'l'  : 2.0,
                          'g'  : 3.0}
   ilqr_config = {
                    'state_trans_func'          : dyn.pend_dyn_nl,
                    'state_trans_func_params'   : state_trans_params,
                    'cost_func'                 : cost.cost_func_quad_state_and_control,
                    'cost_func_params'          : cost_func_params,
                    'sim_method'                : 'euler',
                    'c2d_method'                : 'euler',
                    'max_iter'                  : 100,
                    'time_step'                 : 0.1,
                    'converge_crit'             : 1e-6,
                    'cost_ratio_bounds'         : [1e-4, 10]
                 }
   
   time_step = 0.1
   len_seq   = 11
   state_init_vec = jnp.array([[[0.1],[0.1]]])
   control_init_seq = [jnp.array([[0]])] * (len_seq - 1)


   controller_state = ilqr.ilqrControllerState(ilqr_config, state_init_vec, control_init_seq, ilqr_config['time_step'])
   # initialize controller state and configured functions
   config_funcs, controller_state.state_seq, controller_state.cost_float, controller_state.prev_cost_float \
   = ilqr.initialize_ilqr_controller(ilqr_config, controller_state)
   # run ilqr controller
   controller_output = ilqr.run_ilqr_controller(ilqr_config, config_funcs, controller_state)

   # calculate initial rollout
   # 

   
