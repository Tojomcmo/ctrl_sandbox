from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.animation as animate
import numpy.typing as npt

import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost
import analyze_ilqr_output_funcs as analyze
import gen_ctrl_funcs as gen_ctrl

import mujoco as mujoco
import mjcf_models as mj_models
import mujoco_funcs as mj_funcs
import visualize_mj_funcs as mj_vis

if __name__ == "__main__":
   #------- Define controller configuration -------#
   time_step  = 0.01 
   len_seq    = 10
   num_states = 4
   num_controls = 2
   mj_ctrl = True
   Q_cost  = np.array([[10,0 ,0 ,0 ],
                       [0 ,10,0 ,0 ],
                       [0 ,0 ,1 ,0 ],
                       [0 ,0 ,0 ,1 ]], dtype=float) * 10.0
   # R_cost  = np.array([[1]], dtype=float) * (0.01)
   R_cost  = np.eye(2, dtype=float) * (1.0)   
   Qf_cost = np.array([[10,0 ,0 ,0 ],
                       [0 ,10,0 ,0 ],
                       [0 ,0 ,1 ,0 ],
                       [0 ,0 ,0 ,1 ]], dtype=float) * 1000.0

   #---------- initialize ilqr configuration object
   ilqr_config   = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
   ilqr_config.max_iter = 10
   ilqr_config.fp_max_iter = 20
   #---------- create desired trajectory ----------#
   # traj_gen_dyn_func_params = dyn.nlPendParams(g=9.81,b=5.0,l=1.0)
   # x_tg_init_vec = np.array([[0.0],[0.0],[0.0],[0.0]])
   # u_tg_seq      = np.ones([len_seq-1,1,1]) 
   # traj_gen_cont_dyn_func = lambda x,u: dyn.pend_dyn_nl(traj_gen_dyn_func_params,x,u)
   # traj_gen_disc_dyn_func = lambda x,u: gen_ctrl.step_rk4(traj_gen_cont_dyn_func, ilqr_config.time_step, x, u)
   # x_des_seq_traj_gen         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_disc_dyn_func,x_tg_init_vec, u_tg_seq)

   #---------- create simulation system ----------#
   x_sim_init_vec = np.array([0.0,0.0,0.0,0.0])

   #---------- set system init ----------#
   x_tg_init_vec = np.array([0.0,0.0,0.0,0.0])
   x_init_vec = x_tg_init_vec

   ctrl_target_condition = 2

   if ctrl_target_condition == 1:
      # u_init_seq = np.ones([len_seq-1, num_controls,1]) * (-1.0)
      # u_des_seq  = u_tg_seq 
      # x_des_seq  = x_des_seq_traj_gen  
      pass
   elif ctrl_target_condition == 2:  
      x_des_seq  = np.zeros([len_seq, num_states])
      x_des_seq[:,0] = np.pi
      u_init_seq = np.ones([len_seq-1, num_controls]) * (2.0)
      u_des_seq  = np.zeros([len_seq-1, num_controls])

   else:
      raise ValueError('invalid ctrl_target_condition')   

   #------- complete ilqr configuration --------#
   cost_func_obj = cost.cost_quad_x_and_u(Q_cost,R_cost,Qf_cost,
                                          x_des_seq=x_des_seq,
                                          u_des_seq=u_des_seq)
   if mj_ctrl is True:
      ilqr_config.config_for_mujoco(mj_models.create_MJCF_double_pend_m_d_mod(1,1,1))
   else:
      pass
   ilqr_config.config_cost_func(cost_func_obj.cost_func_quad_state_and_control)
   ilqr_config.create_curried_funcs()   

   #----- Run iLQR algorithm -----#
   controller_state = ilqr.ilqrControllerState(ilqr_config, x_init_vec, u_init_seq)
   # initialize controller state and configured functions
   controller_state.x_seq, controller_state.cost_float, controller_state.prev_cost_float, controller_state.cost_seq \
   = ilqr.initialize_ilqr_controller(ilqr_config, controller_state)
   controller_state.seed_x_seq = controller_state.x_seq
   # run ilqr controller
   controller_output = ilqr.run_ilqr_controller(ilqr_config, controller_state)

   #------- Simulate controller output --------#
   if ilqr_config.mj_ctrl is True:
      mjsim_model, _ , mjsim_data = mj_funcs.create_mujoco_model(ilqr_config.mjcf_model, ilqr_config.time_step)
      sim_dyn_disc_func = lambda x, u: (mj_funcs.fwd_sim_mj_w_ctrl_step(mjsim_model, mjsim_data, x, u))
   else:
      pass
   x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_dyn_disc_func, controller_output, x_sim_init_vec)

   #------ Create mujoco video --------#
   if ilqr_config.mj_ctrl is True:
      framerate = 30
      fig1, ax1 = plt.subplots()
      mjvid_model, mjvid_renderer , mjvid_data = mj_funcs.create_mujoco_model(ilqr_config.mjcf_model, ilqr_config.time_step)
      mjvid_renderer.update_scene(mjvid_data, "fixed")
      scene_option = mujoco.MjvOption()
      scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
      img_set, frames = mj_vis.create_mj_video_ilqr_w_ctrl(mjvid_model, mjvid_data, mjvid_renderer, scene_option, framerate, x_sim_init_vec, controller_output, ilqr_config)
      ani = animate.ArtistAnimation(fig1, img_set, interval = int(1/framerate * 1000))
   #------- plot simulation and controller outputs ------#

   analyze.plot_ilqr_iter_sim_ctrl_cost(ilqr_config, controller_output, cost_func_obj.x_des_seq, x_sim_seq, u_sim_seq)
   plt.show()