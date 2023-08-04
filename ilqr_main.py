from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

import src.ilqr_funcs as ilqr
import src.dyn_functions as dyn
import src.cost_functions as cost
import src.analyze_ilqr_output_funcs as analyze
import src.gen_ctrl_funcs as gen_ctrl


if __name__ == "__main__":
   #------- Define controller configuration -------#
   cost_func_params    = {'Q'  : jnp.array([[10.,0],[0.,10.]]) * 1,
                          'R'  : jnp.array([[.01]]),
                          'Qf' : jnp.array([[1.,0],[0.,1.]]) * 1}
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
                    'converge_crit'             : 1e-4,
                    'cost_ratio_bounds'         : [1e-6, 10],
                    'ro_reg_start'              : 0.0,
                    'ro_reg_change'             : 0.25,
                    'fp_max_iter'               : 5,
                    'log_ctrl_history'          : True
                 }
   
   #----- define timestep and sequence length -----#
   time_step  = 0.1
   len_seq    = 30

   #---------- define plotting condition ----------#
   plot_ctrl_output_bool = False
   plot_ctrl_history_bool = True

   #---------- create desired trajectory ----------#
   traj_gen_dyn_func_params = {'g' : 9.81,
                               'b' : 5.0,
                               'l' : 1.0}
   x_tg_init_vec = np.array([[0.1],[0.1]])
   u_tg_seq      = np.ones([len_seq-1,1,1])*(15)

   #create curried dynamics function for simulation
   traj_gen_dyn_func = lambda t,x,u: dyn.pend_dyn_nl(traj_gen_dyn_func_params,t,x,u)
   x_des_seq         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_dyn_func,x_tg_init_vec, u_tg_seq,time_step, sim_method='solve_ivp_zoh')

   #---------- set system init ----------#
   x_init_vec = x_tg_init_vec
   u_init_seq = np.zeros([len_seq-1, 1,1])
   u_des_seq  = u_tg_seq
   # u_des_seq  = np.zeros([len_seq-1, 1, 1])
   # x_des_seq  = np.zeros([len_seq, 2, 1])
   # x_des_seq[:,0] = np.pi


   controller_state = ilqr.ilqrControllerState(ilqr_config, x_init_vec, u_init_seq, x_des_seq=x_des_seq, u_des_seq=u_des_seq)
   # initialize controller state and configured functions
   config_funcs, controller_state.x_seq, controller_state.cost_float, controller_state.prev_cost_float, controller_state.cost_seq \
   = ilqr.initialize_ilqr_controller(ilqr_config, controller_state)
   #save initial state_seq
   controller_state.seed_x_seq = controller_state.x_seq
   # run ilqr controller
   controller_output = ilqr.run_ilqr_controller(ilqr_config, config_funcs, controller_state)



   if plot_ctrl_output_bool is True:
      x_plot_seqs = [controller_state.seed_x_seq,
                     controller_state.x_des_seq,
                     controller_output.x_seq]
      x_plot_seq_names = ['initial_seq', 'desired_seq', 'output_seq']
      x_plot_seq_styles_dot = ['r.', 'g.', 'b.']
      x_plot_seq_styles_quiver = ['red', 'green', 'blue']
      gs = gridspec.GridSpec(2, 2)
      fig = plt.figure(figsize=(14,6))
      ax1 = fig.add_subplot(gs[:, 0]) # row 0, col 0
      ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
      ax3 = fig.add_subplot(gs[1, 1]) # row 1, span all columns
      analyze.plot_compare_state_sequences_quiver_dot(fig, ax1, x_plot_seqs, x_plot_seq_names, x_plot_seq_styles_quiver, x_plot_seq_styles_dot, quiverwidth=0.0015, xlabel = 'angPos', ylabel = 'angVel')
      analyze.plot_x_y_sequences(fig, ax2, controller_output.time_seq[:-1], controller_output.u_seq[:,0,0], xlabel='Time', ylabel='control') # type: ignore 
      analyze.plot_x_y_sequences(fig, ax3, controller_output.time_seq, controller_output.cost_seq, xlabel='Time', ylabel='cost')
      plt.tight_layout()
      plt.show()

   elif plot_ctrl_history_bool is True:
      x_plot_seqs = controller_output.x_seq_history
      x_plot_seqs.insert(0,controller_output.seed_x_seq)
      x_plot_seqs.append(controller_output.x_des_seq)
      x_plot_seq_names = []
      x_plot_seq_styles_dot = []
      x_plot_seq_styles_quiver = []
      x_plot_seq_quiverwidth   = []
      n = len(x_plot_seqs)
      # set color map
      colormap = cm.get_cmap('rainbow')
      color = iter(colormap(np.linspace(0, 1, n)))
      for idx in range(len(x_plot_seqs)):
         if idx == 0:
            x_plot_seq_names.append('seed state seq')
            x_plot_seq_styles_quiver.append('purple')
            x_plot_seq_quiverwidth.append(0.0015)            
         elif idx == n-1:
            x_plot_seq_names.append('target state seq')
            x_plot_seq_styles_quiver.append('red')
            x_plot_seq_quiverwidth.append(0.0015)         
         elif idx == n-2:   
            x_plot_seq_names.append(f'seq iter {idx}')
            x_plot_seq_styles_quiver.append('orange')
            x_plot_seq_quiverwidth.append(0.0015)  
         elif idx == 1:
            x_plot_seq_names.append('target state seq')
            x_plot_seq_styles_quiver.append('blue')    
            x_plot_seq_quiverwidth.append(0.0015)                                   
         else:   
            x_plot_seq_names.append(f'seq iter {idx}')
            x_plot_seq_styles_quiver.append('gray')
            x_plot_seq_quiverwidth.append(0.001)  
         c = next(color)           
         x_plot_seq_styles_dot.append(c) 
      gs = gridspec.GridSpec(1, 1)
      fig = plt.figure(figsize=(12,8))
      ax1 = fig.add_subplot(gs[:, 0]) # row 0, col 0
      analyze.plot_compare_state_sequences_quiver_dot(fig, ax1, x_plot_seqs,x_plot_seq_names,x_plot_seq_styles_quiver,x_plot_seq_styles_dot, x_plot_seq_quiverwidth, xlabel = 'angPos', ylabel = 'angVel')
      plt.tight_layout()
      plt.show()