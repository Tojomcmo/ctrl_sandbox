from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

import src.ilqr_funcs as ilqr
import src.dyn_functions as dyn
import src.cost_functions as cost
import src.analyze_ilqr_output_funcs as analyze
import src.gen_ctrl_funcs as gen_ctrl

import src.visualize_dyn_funcs as vis_dyn

if __name__ == "__main__":
   #------- Define controller configuration -------#
   save_ani_bool = True
   time_step  = 0.03
   len_seq    = 100
   num_states = 4
   num_controls = 2
   shoulder_act = True
   elbow_act = True
   Q_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,10. ,0   ,0  ],
                       [0   ,0   ,1   ,0  ],
                       [0   ,0   ,0   ,1  ]],
                       dtype=float) * 100.0
   # R_cost  = np.array([[1.0]],dtype=float)*0.1
   R_cost  = np.array([[10.0, 0],[0, 1.0]],dtype=float)*0.1
   Qf_cost  = np.array([[10. ,0   ,0   ,0  ],
                       [0   ,10. ,0   ,0  ],
                       [0   ,0   ,1   ,0  ],
                       [0   ,0   ,0   ,1  ]],
                       dtype=float) * 1000.0
   dyn_func_params_ctrl = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.05, b2=0.05,
                                                 shoulder_act=shoulder_act, elbow_act=elbow_act)

   #---------- initialize ilqr configuration object
   ilqr_config   = ilqr.ilqrConfigStruct(num_states, num_controls, len_seq, time_step)
   ilqr_config.max_iter = 60

   #---------- create desired trajectory ----------#
   x_tg_init_vec = np.array([0.0,0.0,0.0,0.0])
   # traj_gen_dyn_func_params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.1, b2=0.1,
   #                                                   shoulder_act=shoulder_act, elbow_act=elbow_act)
   # u_tg_seq      = np.ones([len_seq-1,num_controls]) 
   # traj_gen_cont_dyn_func = lambda x,u: dyn.double_pend_no_damp_full_act_dyn(traj_gen_dyn_func_params,x,u)
   # traj_gen_disc_dyn_func = lambda x,u: gen_ctrl.step_rk4(traj_gen_cont_dyn_func, ilqr_config.time_step, x, u)
   # x_des_seq_traj_gen         = gen_ctrl.simulate_forward_dynamics_seq(traj_gen_disc_dyn_func,x_tg_init_vec, u_tg_seq)

   #---------- create simulation system ----------#
   sim_dyn_func_params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0.05, b2=0.05,
                                                shoulder_act=shoulder_act, elbow_act=elbow_act)
   x_sim_init_vec = np.array([0.0,0.0,0.0,0.0])

   #---------- set system init ----------#
   x_init_vec = x_tg_init_vec

   ctrl_target_condition = 2

   if ctrl_target_condition == 1:
      u_init_seq = np.ones([len_seq-1, num_controls]) * (-1)
      u_des_seq  = u_tg_seq 
      x_des_seq  = x_des_seq_traj_gen  

   elif ctrl_target_condition == 2:  
      x_des_seq  = np.zeros([len_seq, num_states], dtype=float)
      x_des_seq[:,0] = np.pi
      x_des_seq[:,1] = np.pi      
      u_init_seq = np.ones([len_seq-1, num_controls], dtype=float)*0.1
      u_des_seq  = np.zeros([len_seq-1, num_controls], dtype=float)

   else:
      raise ValueError('invalid ctrl_target_condition')   

   #------- complete ilqr configuration --------#
   cost_func_params = cost.costFuncQuadStateAndControlParams(Q_cost,R_cost,Qf_cost,
                                                            x_des_seq=x_des_seq,
                                                            u_des_seq=u_des_seq)

   ilqr_config.config_for_dyn_func(dyn.double_pend_no_damp_full_act_dyn, dyn_func_params_ctrl, gen_ctrl.step_rk4)
   ilqr_config.config_cost_func(cost.cost_func_quad_state_and_control, cost_func_params)
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

   sim_dyn_cont_func = lambda x,u: dyn.double_pend_no_damp_full_act_dyn(sim_dyn_func_params,x,u)
   sim_dyn_disc_func = lambda x,u: gen_ctrl.step_rk4(sim_dyn_cont_func, ilqr_config.time_step, x, u)
   x_sim_seq, u_sim_seq = ilqr.simulate_ilqr_output(sim_dyn_disc_func, controller_output, x_sim_init_vec)

   #------- plot simulation and controller outputs ------#

   fig = plt.figure(figsize=[10,8])
   pend_animation = vis_dyn.double_pend_animation(sim_dyn_func_params, controller_output.x_seq, time_step, fig)
   pend_animation.create_double_pend_animation()
   pend_animation.show_animation()

   if save_ani_bool == True: 
      print('saving animation...')
      filename:str | os.PathLike = '/Users/thomasmoriarty/Desktop/dpend_full_act'
      pend_animation.save_animation_gif(filename)
      print('animation saved!')

   plt.show()