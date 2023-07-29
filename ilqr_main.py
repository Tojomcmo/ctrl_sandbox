from jax import numpy as jnp
import numpy as np
import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost
import analyze_ilqr_output_funcs as analyze

import matplotlib.pyplot as plt

if __name__ == "__main__":
   cost_func_params    = {'Q'  : jnp.array([[10.,0],[0,1.]]),
                          'R'  : jnp.array([[.01]]),
                          'Qf' : jnp.array([[10.,0],[0,10.]])}
   state_trans_params  = {'b'  : 1.0,
                          'l'  : 1.0,
                          'g'  : 9.81}
   ilqr_config = {
                    'state_trans_func'          : dyn.pend_dyn_nl,
                    'state_trans_func_params'   : state_trans_params,
                    'cost_func'                 : cost.cost_func_quad_state_and_control,
                    'cost_func_params'          : cost_func_params,
                    'sim_method'                : 'solve_ivp_zoh', # 'euler', "solve_ivp_zoh"
                    'c2d_method'                : 'zohCombined',# 'euler', "zoh", 'zohCombined'
                    'max_iter'                  : 10,
                    'time_step'                 : 0.1,
                    'converge_crit'             : 1e-3,
                    'cost_ratio_bounds'         : [1e-6, 10],
                    'ro_reg_start'              : 0.0,
                    'ro_reg_change'             : 0.25,
                    'fp_max_iter'               : 5
                 }
   
   time_step = 0.1
   len_seq   = 30
   state_init_vec = jnp.array([[0.],[0.]])
   control_init_seq = [jnp.array([[0.]])] * (len_seq - 1)
   state_des_seq = [jnp.array([[-jnp.pi], [0]])] * (len_seq)

   controller_state = ilqr.ilqrControllerState(ilqr_config, state_init_vec, control_init_seq, state_des_seq=state_des_seq)
   # initialize controller state and configured functions
   config_funcs, controller_state.state_seq, controller_state.cost_float, controller_state.prev_cost_float \
   = ilqr.initialize_ilqr_controller(ilqr_config, controller_state)
   #save initial state_seq
   controller_state.seed_state_seq = controller_state.state_seq
   # run ilqr controller
   controller_output = ilqr.run_ilqr_controller(ilqr_config, config_funcs, controller_state)

   print(np.array(controller_output.control_seq))

   x_plot_seqs = [controller_state.seed_state_seq,
                  controller_state.state_des_seq,
                  controller_output.state_seq]
   x_plot_seq_names = ['initial_seq', 'desired_seq', 'output_seq']
   x_plot_seq_styles = ['r.', 'g.', 'b.']
   analyze.plot_compare_state_sequences(x_plot_seqs, x_plot_seq_names, x_plot_seq_styles, fig_num=1)

   # u_plot_seqs = [controller_state.seed_control_seq,
   #              controller_state.control_des_seq,
   #              controller_output.control_seq]
   # u_plot_seq_names = ['initial_seq', 'desired_seq', 'output_seq']
   # u_plot_seq_styles = ['r.', 'g.', 'b.']
   # analyze.plot_compare_state_sequences(x_plot_seqs, x_plot_seq_names, x_plot_seq_styles, fig_num=1, xlabel='time')
   # analyze.plot_compare_state_sequences(x_plot_seqs, x_plot_seq_names, x_plot_seq_styles, fig_num=1)
   plt.legend()
   plt.show()

