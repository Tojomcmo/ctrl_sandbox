import numpy as np
import mujoco as mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animate

import src.mujoco_funcs as mj_funcs
import src.mjcf_models as mj_models

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    mjcf_model  = mj_models.create_MJCF_double_pend_m_d_mod(1,1,1)
    # mjcf_model  = mj_models.MJCF_single_pend_2  
    # mjcf_model  = mj_models.MJCF_dpend_outer_motor
    model = mujoco.MjModel.from_xml_string(mjcf_model)
    data = mujoco.MjData(model)

    # renderer and visualization
    show_plot = True

    # finite difference prep
    nu = model.nu
    nx = model.nv    
    eps = 1e-6
    A = np.zeros((2*nx, 2*nx))
    B = np.zeros((2*nx,   nu))
    flg_centered = True

    # update model and data parameters
    time_step = 0.01
    model.opt.timestep = time_step
    steps     = 100
    data.ctrl = np.array([50.0,10.0])
    mujoco.mj_forward(model,data)

    #initial linearization and predictiuon
    state_vec = np.hstack((data.qpos.copy(), data.qvel.copy())).reshape(-1,1) 
    control_vec = np.array(data.ctrl).reshape(-1,1)
    mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
    state_next_pred = state_next_pred = (A @ state_vec) + (B @ control_vec)


    if show_plot is True:
        renderer = mujoco.Renderer(model, 480, 480)
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        renderer.update_scene(data, "fixed", scene_option=scene_option)
        frames = []
        img_set = []
        fig1, ax1 = plt.subplots()

    for idx in range(steps):      
        mujoco.mj_step(model, data)
        state_vec = np.hstack((data.qpos.copy(), data.qvel.copy())).reshape(-1,1)      
        mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
        control_vec = np.array(data.ctrl).reshape(-1,1)
        state_next_pred = (A @ state_vec) + (B @ control_vec)
        # update frame
        if show_plot is True:  
            mj_funcs.update_plt_frame(renderer, data, frames, img_set)
        idx += 1    


    if show_plot is True:
        ani = animate.ArtistAnimation(fig1, img_set, interval = time_step*1000)
        plt.show()

###################################

    for idx in range(steps):
        print('state_predict: ', state_next_pred)         
        mujoco.mj_step(model, data)
        state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)      
        print('state_actual: ', state_vec)  
        mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
        print('A: ', A)
        print('B: ', B)    
        state_vec = np.hstack((data.qpos, data.qvel)).reshape(-1,1)
        control_vec = np.array(data.ctrl).reshape(-1,1)
        state_next_pred = A @ state_vec + B @   control_vec


###################################


    # ts_sim = 0.01
    # ts_ctrl = 0.1
    # u_seq     = np.arange(1,10)*1
    # x_init = np.array([[0.0],[0.0],[0.0],[0.0]])

    # x_seq        = mj_funcs.fwd_sim_mj_w_ctrl(model, data,x_init, u_seq, ts_sim, ts_ctrl)
    # A_seq, B_seq = mj_funcs.linearize_mj_seq(model, data, x_seq, u_seq, ts_ctrl)

    # print(x_seq)
    # print(A_seq)
    # print(B_seq)

