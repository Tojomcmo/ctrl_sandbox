#@title Import packages for plotting and creating graphics
import numpy as np
import mujoco as mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animate

import src.mjcf_models as mjcf
import src.mujoco_funcs as mj_funcs
import src.gen_ctrl_funcs as gen_ctrl
# More legible printing from numpy.

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    mjcf_model = mjcf.MJCF_single_pend_2
    model, renderer, data = mj_funcs.create_mujoco_model(mjcf_model)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, "fixed")

    nu = model.nu
    nx = model.nv

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    duration = 5  # (seconds)
    framerate = 3  # (Hz)

    frames = []
    img_set = []
    mujoco.mj_resetData(model, data)
    data.ctrl = 4.0

    # finite difference prep
    eps = 1e-4
    A = np.zeros((2*nx, 2*nx))
    B = np.zeros((2*nx,   nu))
    flg_centered = False
    state_next_pred = np.hstack((data.qpos, data.qvel)).reshape(-1,1)


    time_step = 0.01
    model.opt.timestep = time_step
    # u_seq     = np.ones((1,int(duration/time_step)))*0.1
    data.ctrl = 1
    mujoco.mj_forward(model,data)

    spec = 3
    state_get = np.zeros((2*nx, 1))

####################################

    # fig1, ax1 = plt.subplots()

###################################

#     while data.time < duration:
#         print('state_predict: ', state_next_pred)         
#         mujoco.mj_step(model, data)
#         mujoco.mj_getState(model, data,state_get, spec)
#         state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)      
#         print('state_actual: ', state_vec)  
# #        if len(frames) < data.time * framerate:
#         mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
#         # print('A: ', A)
#         # print('B: ', B)    
#         state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)
#         control_vec = np.array(data.ctrl).reshape(-1,1)
#         state_next_pred = A @ state_vec + B @   control_vec
#         mj_funcs.update_plt_frame(renderer, data, frames, img_set)


#     # ani = animate.ArtistAnimation(fig1, img_set, interval = int(1/framerate * 1000))
#     ani = animate.ArtistAnimation(fig1, img_set, interval = int(1/(1/time_step) * 1000))
#     plt.show()

###################################

    # while data.time < duration:
    #     print('state_predict: ', state_next_pred)         
    #     mujoco.mj_step(model, data)
    #     state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)      
    #     print('state_actual: ', state_vec)  
    #     mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
    #     print('A: ', A)
    #     print('B: ', B)    
    #     state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)
    #     control_vec = np.array(data.ctrl).reshape(-1,1)
    #     state_next_pred = A @ state_vec + B @   control_vec
    #     # mj_funcs.update_plt_frame(renderer, data, frames, img_set)


###################################


    time_step = 0.01
    u_seq     = np.arange(1,10)*0.1

    x_seq = mj_funcs.fwd_sim_mj_w_ctrl(model, data,u_seq, time_step)

    timestep_lin = 0.1
    A_seq, B_seq = mj_funcs.linearize_mj_seq(model, data, x_seq, u_seq,timestep_lin)

    print(x_seq)
    print(A_seq)
    print(B_seq)

