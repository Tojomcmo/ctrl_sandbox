from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.animation as animate
import numpy.typing as npt

import src.ilqr_funcs as ilqr
import src.dyn_functions as dyn
import src.cost_functions as cost
import src.analyze_ilqr_output_funcs as analyze
import src.gen_ctrl_funcs as gen_ctrl

import mujoco as mujoco
import src.mjcf_models as mj_models
import src.mujoco_funcs as mj_funcs

if __name__ == "__main__":
   #------- Define controller configuration -------#
   cost_func_params      = {'Q'  : jnp.array([[10.,0],[0.,1.]]) * 1,
                            'R'  : jnp.array([[0.1]]),
                            'Qf' : jnp.array([[10.,0],[0.,1.]]) * 10}
   cont_dyn_func_params  = {'b'  : 1.0,
                            'l'  : 1.0,
                            'g'  : 9.81}
   ilqr_config_cont = {
                    'mj_ctrl'                   : False,   
                    'cont_dyn_func'             : dyn.pend_dyn_nl,
                    'cont_dyn_func_params'      : cont_dyn_func_params,
                    'cost_func'                 : cost.cost_func_quad_state_and_control,
                    'cost_func_params'          : cost_func_params,
                    'integrate_func'            : gen_ctrl.step_rk4, # 'step_rk4',"step_euler_forward", "step_solve_ivp"
                    'max_iter'                  : 10,
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
   ilqr_config_mj = {
                    'mj_ctrl'                   : True,   
                    'mjcf_model'                : mj_models.create_MJCF_single_pend_m_d_mod(1,1,1),  
                    'timestep_sim'              : 0.002, 
                    'cost_func'                 : cost.cost_func_quad_state_and_control,
                    'cost_func_params'          : cost_func_params,
                    'max_iter'                  : 6,
                    'time_step'                 : 0.1,
                    'converge_crit'             : 1e-5,
                    'ff_gain_tol'               : 1e-4,
                    'cost_ratio_bounds'         : [1e-6, 10],
                    'ro_reg_start'              : 0.0,
                    'ro_reg_change'             : 0.5,
                    'fp_max_iter'               : 8,
                    'ls_scale_alpha_param'      : 0.5,
                    'log_ctrl_history'          : True 
                 }
   
   # ilqr_config = ilqr_config_mj
   ilqr_config = ilqr_config_cont

   #----- define timestep and sequence length -----#
   time_step  = ilqr_config['time_step']
   len_seq    = 30
   num_states = 4
   num_controls = 2

   #---------- create desired trajectory system ----------#
   traj_gen_dyn_func_params = {'g' : 9.81,
                               'b' : 5.0,
                               'l' : 1.0}
   x_tg_init_vec = np.array([[0.0],[0.0]])
   u_tg_seq      = np.ones([len_seq-1,1,1])*(2)

   #---------- create simulation system ----------#
   sim_dyn_func_params = {'g' : 9.81,
                          'b' : 1.0,
                          'l' : 1.0}
   x_sim_init_vec = np.array([[-1.0],[-2.0]])

   #---- generate desired trajectory from traj gen system----#
   traj_gen_dyn_func = lambda t,x,u: dyn.pend_dyn_nl(traj_gen_dyn_func_params,t,x,u)
   x_des_seq         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_dyn_func,x_tg_init_vec, u_tg_seq,time_step, sim_method='solve_ivp_zoh')

   #---------- set system init ----------#
   x_init_vec = x_tg_init_vec

   ctrl_target_condition = 2

   if ctrl_target_condition == 1:
      u_init_seq = np.ones([len_seq-1, 1,1]) * (-1)
      u_des_seq  = u_tg_seq   

   elif ctrl_target_condition == 2:  
      x_des_seq  = np.zeros([len_seq, 2, 1])
      x_des_seq[:,0] = np.pi
      u_init_seq = np.ones([len_seq-1, 1,1]) * (-1)
      u_des_seq  = np.zeros([len_seq-1, 1, 1])

   else:
      raise ValueError('invalid ctrl_target_condition')   


   ilqr_config = ilqr_config_mj


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
   if ilqr_config['mj_ctrl'] is True:
      ts_sim = 0.002
      mjsim_model, _ , mjsim_data = mj_funcs.create_mujoco_model(ilqr_config['mjcf_model'])
      sim_dyn_func_step = lambda x, u: (mj_funcs.fwd_sim_mj_w_ctrl(mjsim_model, mjsim_data, x, u, ts_sim, ilqr_config['time_step']))[-1]
   else:
      sim_dyn_func = lambda t,x,u: dyn.pend_dyn_nl(sim_dyn_func_params,x,u)
      sim_dyn_func_step = lambda x, u: gen_ctrl.simulate_forward_dynamics_step(sim_dyn_func, x, u,controller_output.time_step, sim_method='solve_ivp_zoh')
   x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_dyn_func_step, controller_output, x_sim_init_vec)

   #------ Create mujoco video --------#
   framerate = 30
   fig1, ax1 = plt.subplots()
   mjvid_model, mjvid_renderer , mjvid_data = mj_funcs.create_mujoco_model(ilqr_config['mjcf_model'])
   mjvid_renderer.update_scene(mjvid_data, "fixed")
   scene_option = mujoco.MjvOption()
   scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
   img_set, frames = mj_funcs.create_mj_video_w_ctrl(mjvid_model, mjvid_data, mjvid_renderer, scene_option, framerate, x_sim_init_vec, controller_output, ilqr_config)

   #------- plot simulation and controller outputs ------#
   ani = animate.ArtistAnimation(fig1, img_set, interval = int(1/framerate * 1000))
   analyze.plot_ilqr_iter_sim_ctrl_cost(controller_output, x_sim_seq, u_sim_seq)
   plt.show()