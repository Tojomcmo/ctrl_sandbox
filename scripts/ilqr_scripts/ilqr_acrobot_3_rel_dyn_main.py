from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.typing as npt
import os

import ilqr_funcs as ilqr
import dyn_functions as dyn
import cost_functions as cost
import analyze_ilqr_output_funcs as analyze
import gen_ctrl_funcs as gen_ctrl
import visualize_dyn_funcs as vis_dyn

if __name__ == "__main__":
   #------- Define controller configuration -------#
   save_ani_bool = True
   ani_save_location = "media_output/"
   ani_save_name = "acrobot"
   os.makedirs(ani_save_location, exist_ok=True)
   time_step  = 0.04
   len_seq    = 65
   num_states = 4
   num_controls = 2
   shoulder_act = True
   elbow_act = True
   Q_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,10. ,0   ,0  ],
                       [0   ,0   ,0.1   ,0  ],
                       [0   ,0   ,0   ,0.1  ]],
                       dtype=float) * 10.0
   # R_cost  = np.array([[1.0]],dtype=float)*1.0
   R_cost  = np.array([[5.0, 0],[0, 1.0]],dtype=float)*0.5
   Qf_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,10. ,0   ,0  ],
                       [0   ,0   ,1.0   ,0  ],
                       [0   ,0   ,0   ,1.0  ]],
                       dtype=float) * 4000.0
   
   h_bar = 1.0
   r_bar = 0.05
   m_bar = 1.0
   d_bar = h_bar/2
   moi = (1/12)*h_bar*(m_bar**2 + 3*r_bar**2) 
   dyn_func_sys_ctrl = dyn.double_pend_rel_dyn(g=9.81, m1=m_bar, moi1=moi, d1=d_bar, l1=h_bar, m2=m_bar, moi2=moi, d2=d_bar, l2=h_bar, b1=0.0, b2=0.0,
                                                 shoulder_act=shoulder_act, elbow_act=elbow_act)

   #---------- initialize ilqr configuration object --------------#
   ilqr_config   = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
   ilqr_config.converge_crit = 1e-5
   ilqr_config.max_iter = 50


   #---------- create simulation system for post algorithm test ----------#
   dyn_func_sys_sim = dyn_func_sys_ctrl
   # dyn_func_sys_sim = dyn.double_pend_abs_dyn(g=9.81, m1=1.0, moi1=1.0, d1=0.5, l1=1.0, m2=1.0, moi2=1.0, d2=0.5, l2=1.0, b1=0.0, b2=0.0,
   #                                               shoulder_act=shoulder_act, elbow_act=elbow_act)

   x_sim_init_vec = np.array([0.0,0.0,0.0,0.0])

   #---------- set system state init and desired trajectories ----------#
   x_tg_init_vec = np.array([0.0,0.0,0.0,0.0])
   x_init_vec = x_tg_init_vec

   ctrl_target_condition = 2

   if ctrl_target_condition == 1:
      dyn_func_sys_traj_dyn = dyn.double_pm_pend_dyn(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.0, b2=0.0,
                                                      shoulder_act=shoulder_act, elbow_act=elbow_act)
      u_tg_seq      = np.ones([len_seq-1,num_controls]) 
      traj_gen_disc_dyn_func = lambda x,u: gen_ctrl.step_rk4(dyn_func_sys_traj_dyn.cont_dyn_func, ilqr_config.time_step, x, u)
      x_des_seq_traj_gen         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_disc_dyn_func,x_tg_init_vec, u_tg_seq)

      u_init_seq = np.ones([len_seq-1, num_controls]) * (-1)
      u_des_seq  = u_tg_seq 
      x_des_seq  = x_des_seq_traj_gen  

   elif ctrl_target_condition == 2:  
      x_des_seq  = np.zeros([len_seq, num_states], dtype=float)
      x_des_seq[:,0] = np.pi
      x_des_seq[:,1] = 0    
      u_init_seq = np.ones([len_seq-1, num_controls], dtype=float)*0.01
      u_des_seq  = np.zeros([len_seq-1, num_controls], dtype=float)

   else:
      raise ValueError('invalid ctrl_target_condition')   

   #------- complete ilqr configuration --------#
   cost_func_obj = cost.cost_quad_x_and_u(Q_cost,R_cost,Qf_cost,
                                                            x_des_seq=x_des_seq,
                                                            u_des_seq=u_des_seq)

   ilqr_config.config_for_dyn_func(dyn_func_sys_ctrl.cont_dyn_func, gen_ctrl.step_rk4)
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

   sim_dyn_disc_func = lambda x,u: gen_ctrl.step_rk4(dyn_func_sys_sim.cont_dyn_func, ilqr_config.time_step, x, u)
   x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_dyn_disc_func, controller_output, x_sim_init_vec)

   #------- plot simulation and controller outputs ------#

   fig = plt.figure(figsize=[16,8])
   gs =  gridspec.GridSpec(2, 2)
   ax1 = fig.add_subplot(gs[:, 0]) # row 0, col 0
   ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
   ax3 = fig.add_subplot(gs[1, 1]) # row 1, span all columns
   
   if shoulder_act is True and elbow_act is True:
      ax2.plot(controller_output.time_seq[:-1],u_sim_seq[:,0], label = 'shoulder control effort')
      ax2.plot(controller_output.time_seq[:-1],u_sim_seq[:,1], label = 'elbow control effort')
      ax2.legend() 
   else:
      ax2.plot(controller_output.time_seq[:-1],u_sim_seq, label = 'control effort')
      ax2.legend() 
   ax3.plot(controller_output.time_seq, controller_output.cost_seq, label='total cost')
   ax3.plot(controller_output.time_seq, controller_output.x_cost_seq, label='state cost')
   ax3.plot(controller_output.time_seq, controller_output.u_cost_seq, label='control cost')    
   ax3.legend()  

   pend_animation = vis_dyn.double_pend_animation(dyn_func_sys_sim.get_animate_value_dict(), x_sim_seq, time_step, fig, ax1, th2='rel')
   pend_animation.create_double_pend_animation()
   plt.tight_layout()
   plt.show()

   if save_ani_bool == True: 
      print('saving animation...')
      filename:str | os.PathLike = ani_save_location+ani_save_name
      pend_animation.save_animation_gif(filename)
      print('animation saved!')

