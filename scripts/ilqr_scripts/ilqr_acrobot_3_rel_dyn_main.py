import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.typing as npt
import os

import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost
import gen_ctrl_funcs as gen_ctrl
import visualize_dyn_funcs as vis_dyn
import mujoco_funcs as mj_funcs
import mjcf_models as mjcf
import simulate_funcs as sim
import cProfile
import pstats


if __name__ == "__main__":
   #start profiling
   profiler = cProfile.Profile()
   profiler.enable()
   jax.config.update("jax_enable_x64", True)
   #------- Define controller configuration -------#
   save_ani_bool = False
   sim_with_mj = False
   ani_save_location = "media_output/"
   ani_save_name = "acrobot"
   os.makedirs(ani_save_location, exist_ok=True)
   time_step  = 0.01
   len_seq    = 500
   num_states = 4
   num_controls = 1
   shoulder_act = False 
   elbow_act = True
   Q_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,1. ,0   , 0  ],
                       [0   ,0   ,0.1   ,0  ],
                       [0   ,0   ,0   ,5.0  ]],
                       dtype=float) * 0.05  
   R_cost  = np.array([[1.0]],dtype=float)*5.0
   # R_cost  = np.array([[1.0, 0],[0, 1.0]],dtype=float)*0.5
   Qf_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,10. ,0   ,0  ],
                       [0   ,0   ,1.0   ,0  ],
                       [0   ,0   ,0   ,1.0  ]],
                       dtype=float) * 10000.0
   
   h_bar = 1.0
   r_bar = 0.05
   m_bar = 1.0
   d_bar = h_bar/2
   moi = (1/12)*h_bar*(m_bar**2 + 3*r_bar**2) 
   dyn_func_sys_ctrl = dyn.double_pend_rel_dyn(g=9.81, m1=m_bar, moi1=moi, d1=d_bar, l1=h_bar, m2=m_bar, moi2=moi, d2=d_bar, l2=h_bar, b1=0.0, b2=0.0,
                                                 shoulder_act=shoulder_act, elbow_act=elbow_act)

   #---------- initialize ilqr configuration object --------------#
   ilqr_config   = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
   ilqr_config.converge_crit = 1e-6
   ilqr_config.max_iter = 500
   ilqr_config.fp_max_iter = 10

   #---------- create simulation system for post algorithm test ----------#
   dyn_func_sys_sim = dyn_func_sys_ctrl
   # dyn_func_sys_sim = dyn.double_pend_abs_dyn(g=9.81, m1=1.0, moi1=1.0, d1=0.5, l1=1.0, m2=1.0, moi2=1.0, d2=0.5, l2=1.0, b1=0.0, b2=0.0,
   #                                               shoulder_act=shoulder_act, elbow_act=elbow_act)


   #---------- set system state init and desired trajectories ----------#
   x_init_vec = jnp.array([0.0,0.0,0.0,0.0])
   x_sim_init_vec = jnp.array([0.0,0.0,0.0,0.0])


   ctrl_target_condition = 2

   if ctrl_target_condition == 1:
      dyn_func_sys_traj_dyn = dyn.double_pm_pend_dyn(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.0, b2=0.0,
                                                      shoulder_act=shoulder_act, elbow_act=elbow_act)
      u_tg_seq      = jnp.ones([len_seq-1,num_controls]) 
      traj_gen_disc_dyn_func = lambda x,u: gen_ctrl.step_rk4(dyn_func_sys_traj_dyn.cont_dyn_func, ilqr_config.time_step, x, u)
      x_des_seq_traj_gen         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_disc_dyn_func,x_init_vec, u_tg_seq)

      u_init_seq = jnp.ones([len_seq-1, num_controls]) * (0.001)
      u_des_seq  = u_tg_seq 
      x_des_seq  = x_des_seq_traj_gen  

   elif ctrl_target_condition == 2:  
      x_des_seq  = jnp.zeros([len_seq, num_states], dtype=float)
      x_des_seq  = x_des_seq.at[:,0].set(jnp.pi)
      x_des_seq  = x_des_seq.at[:,1].set(0.0)
      u_init_seq = jnp.ones([len_seq-1, num_controls], dtype=float)*0.01
      u_des_seq  = jnp.zeros([len_seq-1, num_controls], dtype=float)

   else:
      raise ValueError('invalid ctrl_target_condition')   

   #------- complete ilqr configuration --------#
   cost_func_obj = cost.cost_quad_x_and_u(Q_cost,R_cost,Qf_cost,
                                          x_des_seq=x_des_seq,
                                          u_des_seq=u_des_seq)

   ilqr_config.config_for_dyn_func(dyn_func_sys_ctrl.cont_dyn_func, gen_ctrl.step_rk4)
   ilqr_config.config_cost_func(cost_func_obj.cost_func_quad_state_and_control_scan_compatible)
   ilqr_config.create_curried_funcs()   


   # end code profiling
   profiler.disable()
   profiler.dump_stats('profiler_out.stats')
   #----- Run iLQR algorithm -----#
   controller_state = ilqr.ilqrControllerState(ilqr_config, x_init_vec, u_init_seq)
   ctrl_out = ilqr.run_ilqr_controller(ilqr_config, controller_state)
   #------- Simulate controller output --------#


   if sim_with_mj is True:
      mjcf_obj = mjcf.mjcf_dpend(shoulder_act, elbow_act)
      mj_model, _, mj_data = mj_funcs.create_mujoco_model(mjcf_obj.get_mjcf_model(), time_step=time_step)
      sim_disc_dyn_func = lambda x_k, u_k: mj_funcs.fwd_sim_mj_w_ctrl_step(mj_model, mj_data, np.array(x_k), np.array(u_k))
      print('simulating with mujoco...')
      x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_disc_dyn_func, ctrl_out, x_sim_init_vec)
   else:   
      sim_disc_dyn_func = lambda k, x_k,u_k: gen_ctrl.step_rk4(dyn_func_sys_sim.cont_dyn_func, ilqr_config.time_step, x_k, u_k)
      control_func = sim.prep_ff_fb_u_for_sim(ctrl_out.u_seq, ctrl_out.x_seq, ctrl_out.K_seq)
      measure_func = sim.direct_pass
      disturb_func = sim.direct_pass
      noise_func   = sim.direct_pass
      print('simulating with dpend func...')
      x_sim_seq, x_est_sim_seq, u_sim_seq = sim.sim_dyn(sim_disc_dyn_func,
                                                      control_func,
                                                      measure_func,
                                                      disturb_func,
                                                      noise_func,
                                                      x_sim_init_vec, ctrl_out.len_seq)


   #------- plot simulation and controller outputs ------#
   print("animating simulation output...")
   fig = plt.figure(figsize=(16,8))
   gs =  gridspec.GridSpec(2, 2)
   ax1 = fig.add_subplot(gs[:, 0]) # row 0, col 0
   ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
   ax3 = fig.add_subplot(gs[1, 1]) # row 1, span all columns
   
   vis_dyn.plot_ilqr_dpend_act_and_cost_axes(shoulder_act, elbow_act, ctrl_out, u_sim_seq, ax2, ax3)  

   pend_animation = vis_dyn.double_pend_animation(dyn_func_sys_sim.get_animate_value_dict(), np.array(x_sim_seq), time_step, fig, ax1, th2='rel')
   pend_animation.create_double_pend_animation()
   plt.tight_layout()
   plt.show()

   if save_ani_bool == True: 
      print('saving animation...')
      filename:str | os.PathLike = ani_save_location+ani_save_name
      pend_animation.save_animation_gif(filename)
      print('animation saved!')

