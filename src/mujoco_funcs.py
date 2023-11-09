import mujoco as mujoco
import numpy as np
import matplotlib.pyplot as plt

from . import gen_ctrl_funcs as gen_ctrl

def create_mujoco_model(mjcf_model:str):
    model = mujoco.MjModel.from_xml_string(mjcf_model)
    renderer = mujoco.Renderer(model, 480, 480)
    data = mujoco.MjData(model)
    return model,renderer,data

def update_plt_frame(renderer:mujoco.Renderer, data:mujoco._structs.MjData, frames:list, img_set:list):
    renderer.update_scene(data, "fixed")    
    pixels = renderer.render()
    frames.append(pixels)
    img_set.append([plt.imshow(pixels)])


def fwd_sim_mj_w_ctrl(model, data, u_seq, ts_sim, ts_ctrl):
    '''
        function simulates provided model and data forward in time using u sequence
        and returns state trajectory x_seq and linearized state and control matricies A
        and B at each pair x_seq, u_seq
    '''
    if int(ts_ctrl % ts_sim) != 0:
        raise ValueError('the control timestep must be an integer multiple of the simulation timestep')
    nx            = model.nv
    t_final       = len(u_seq) * ts_ctrl
    sim_steps     = int(t_final / ts_sim)
    steps_zoh     = int(ts_ctrl / ts_sim)
    x_seq         = np.zeros((len(u_seq)+1, nx*2, 1   ))
    time_seq      = np.zeros(len(u_seq)+1)

    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    mujoco.mj_forward(model,data)

    x_seq[0]   = get_state_vec(data)
    data.ctrl  = u_seq[0]
    ctrl_idx   = 0
    ctrl_count = 0  
    
    for sim_idx in range(sim_steps+1):
        if ctrl_count == steps_zoh:
            ctrl_count = 0  
            ctrl_idx  += 1               
            x_seq[ctrl_idx]    = get_state_vec(data)
            time_seq[ctrl_idx] = data.time    
            if sim_idx < sim_steps:             
                data.ctrl = u_seq[ctrl_idx]
                mujoco.mj_step(model, data)
                ctrl_count += 1  
        else:
            mujoco.mj_step(model, data)   
            ctrl_count += 1         
    model.opt.timestep = prev_timestep       
    return x_seq, time_seq

def linearize_mj_seq(model, data,x_seq, u_seq, ts_ctrl):
    '''
        function simulates provided model and data forward in time using u sequence
        and returns state trajectory x_seq and linearized state and control matricies A
        and B at each pair x_seq, u_seq
    '''
    nu      = model.nu
    nx      = model.nv
    prev_timestep = model.opt.timestep
    mujoco.mj_resetData(model, data)
    model.opt.timestep = ts_ctrl
    mujoco.mj_forward(model, data)
    x_len   = len(x_seq[0])
    x_vec_split = int((x_len/2))
    A_seq = np.zeros((len(u_seq)  , nx*2, nx*2))
    B_seq = np.zeros((len(u_seq)  , nx*2, nu  ))
    for idx in range(len(u_seq)):           
        # x_vec_k   = x_seq[idx]
        #reset model data
        mujoco.mj_resetData(model, data)  
        data.ctrl = u_seq[idx]
        data.qpos = (x_seq[idx,:x_vec_split])[:,0]
        data.qvel = (x_seq[idx,x_vec_split:])[:,0]
        A_idx, B_idx = linearize_mujoco_state_and_control(model, data)
        A_seq[idx] = A_idx
        B_seq[idx] = B_idx
    model.opt.timestep = prev_timestep    
    return A_seq, B_seq


def linearize_mujoco_state_and_control(model, data, eps=1e-6, flg_centered=True):
    nu = model.nu
    nx = model.nv
    A = np.zeros((2*nx, 2*nx))
    B = np.zeros((2*nx,   nu))
    mujoco.mj_forward(model, data)
    mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)
    return A, B

def get_state_vec(data):
    return np.hstack((data.qpos, data.qvel)).reshape(-1,1)  