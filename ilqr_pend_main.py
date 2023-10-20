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
   cost_func_params    = {'Q'  : jnp.array([[10.,0],[0.,1.]]) * 10,
                          'R'  : jnp.array([[0.1]]),
                          'Qf' : jnp.array([[10.,0],[0.,1.]]) * 50}
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
                    'max_iter'                  : 20,
                    'time_step'                 : 0.1,
                    'converge_crit'             : 1e-4,
                    'ff_gain_tol'               : 1e-3,
                    'cost_ratio_bounds'         : [1e-6, 10],
                    'ro_reg_start'              : 0.0,
                    'ro_reg_change'             : 0.5,
                    'fp_max_iter'               : 8,
                    'ls_scale_alpha_param'      : 0.5,
                    'log_ctrl_history'          : True
                 }
   
   #----- define timestep and sequence length -----#
   time_step  = 0.1
   len_seq    = 40

   #---------- create desired trajectory system ----------#
   traj_gen_dyn_func_params = {'g' : 9.81,
                               'b' : 5.0,
                               'l' : 1.0}
   x_tg_init_vec = np.array([[0.1],[0.1]]) * (-1)
   u_tg_seq      = np.ones([len_seq-1,1,1])*(10)

   #---------- create simulation system ----------#
   sim_dyn_func_params = {'g' : 9.81,
                          'b' : 1.0,
                          'l' : 1.0}
   x_sim_init_vec = np.array([[0.1],[0.1]])* (-1)

   #---- generate desired trajectory from traj gen system----#
   traj_gen_dyn_func = lambda t,x,u: dyn.pend_dyn_nl(traj_gen_dyn_func_params,t,x,u)
   x_des_seq         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_dyn_func,x_tg_init_vec, u_tg_seq,time_step, sim_method='solve_ivp_zoh')

   #---------- set system init ----------#
   x_init_vec = x_tg_init_vec
   # x_des_seq  = np.zeros([len_seq, 2, 1])
   # x_des_seq[:,0] = np.pi

   u_init_seq = np.ones([len_seq-1, 1,1]) * (-1)
   u_des_seq  = u_tg_seq
   # u_des_seq  = np.zeros([len_seq-1, 1, 1])



   #----- Run iLQR algorithm -----#
   controller_state = ilqr.ilqrControllerState(ilqr_config, x_init_vec, u_init_seq, x_des_seq=x_des_seq, u_des_seq=u_des_seq)
   # initialize controller state and configured functions
   config_funcs, controller_state.x_seq, controller_state.cost_float, controller_state.prev_cost_float, controller_state.cost_seq \
   = ilqr.initialize_ilqr_controller(ilqr_config, controller_state)
   #save initial state_seq
   controller_state.seed_x_seq = controller_state.x_seq
   # run ilqr controller
   controller_output = ilqr.run_ilqr_controller(ilqr_config, config_funcs, controller_state)

   #------- Simulate controller output --------#
   sim_dyn_func = lambda t,x,u: dyn.pend_dyn_nl(sim_dyn_func_params,t,x,u)
   x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_dyn_func, controller_output, x_sim_init_vec, sim_method='solve_ivp_zoh')

   #------- plot simulation and controller outputs ------#
   analyze.plot_ilqr_iter_sim_ctrl_cost(controller_output, x_sim_seq, u_sim_seq)

plt.show()