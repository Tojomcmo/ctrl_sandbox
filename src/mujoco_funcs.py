import mujoco as mujoco
import numpy as np
import matplotlib.pyplot as plt

from . import gen_ctrl_funcs as gen_ctrl
from . import ilqr_funcs as ilqr

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


def fwd_sim_mj_w_ctrl(model, data, x_init, u_seq, ts_sim, ts_ctrl):
    '''
        function simulates provided model and data forward in time using u sequence
        and returns state trajectory x_seq and linearized state and control matricies A
        and B at each pair x_seq, u_seq
    '''
    if int(ts_ctrl % ts_sim) != 0:
        raise ValueError('the control timestep must be an integer multiple of the simulation timestep')
    elif ts_ctrl < ts_sim:
        raise ValueError('the control timestep must be larger than or equal to the simulation timestep')
    nx            = model.nv
    t_final       = len(u_seq) * ts_ctrl
    sim_steps     = int(t_final / ts_sim)
    steps_zoh     = int(ts_ctrl / ts_sim)
    x_seq         = np.zeros((len(u_seq)+1, nx*2, 1   ))
    time_seq      = np.zeros(len(u_seq)+1)

    mujoco.mj_resetData(model, data)
    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    set_mj_state_vec(data, x_init)    
    mujoco.mj_forward(model,data)
    x_seq[0]   = get_state_vec(data)
    assert (x_seq[0]).tolist() == x_init.tolist(), "x_init vector improperly initialized or queried"
    data.ctrl  = (u_seq[0]).reshape(-1)
    ctrl_idx   = 0
    ctrl_count = 0  
    
    for sim_idx in range(sim_steps+1):
        if ctrl_count == steps_zoh:
            ctrl_count = 0  
            ctrl_idx  += 1               
            x_seq[ctrl_idx]    = get_state_vec(data)
            time_seq[ctrl_idx] = data.time    
            if sim_idx < sim_steps:             
                data.ctrl = (u_seq[ctrl_idx]).reshape(-1)
            else:
                break    
        mujoco.mj_step(model, data)   
        ctrl_count += 1         
    model.opt.timestep = prev_timestep       
    return x_seq

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
        data.ctrl = (u_seq[idx]).reshape(-1)
        set_mj_state_vec(data, x_seq[idx])
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

def set_mj_state_vec(data, x_vec):
    x_vec_split = int(len(x_vec)/2)
    data.qpos = (x_vec[:x_vec_split])[:,0]
    data.qvel = (x_vec[x_vec_split:])[:,0]



def create_mj_sim_video(model, data, framerate, sim_dyn_step):
    renderer = mujoco.Renderer(model, 480, 480)
    renderer.update_scene(data, "fixed")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    frames = []
    img_set = []
    fig1, ax1 = plt.subplots()

def create_mj_video_w_ctrl(model, data, renderer, scene_option, framerate, x_init, ctrl_out, ilqr_config):
    '''
        function simulates provided model and data forward in time using u sequence
        and returns state trajectory x_seq and linearized state and control matricies A
        and B at each pair x_seq, u_seq
    '''
    ts_sim = ilqr_config['timestep_sim']
    ts_ctrl = ilqr_config['time_step']
    if int(ts_ctrl % ts_sim) != 0:
        raise ValueError('the control timestep must be an integer multiple of the simulation timestep')
    elif ts_ctrl < ts_sim:
        raise ValueError('the control timestep must be larger than or equal to the simulation timestep')
    frames = []
    img_set = []
    nx            = model.nv
    nu            = model.nu
    t_final       = (ctrl_out.len_seq-1) * ts_ctrl
    sim_steps     = int(t_final / ts_sim)
    steps_zoh     = int(ts_ctrl / ts_sim)
    x_sim_seq         = np.zeros((ctrl_out.len_seq, nx*2, 1   ))
    u_sim_seq     = np.zeros((ctrl_out.len_seq-1, nu, 1   ))
    mujoco.mj_resetData(model, data)
    prev_timestep = model.opt.timestep
    model.opt.timestep = ts_sim
    set_mj_state_vec(data, x_init)    
    mujoco.mj_forward(model,data)
    x_sim_seq[0]   = get_state_vec(data)
    assert (x_sim_seq[0]).tolist() == x_init.tolist(), "x_init vector improperly initialized or queried"
    ctrl_idx   = 0    
    u_sim_seq[0] = ilqr.calculate_u_star(ctrl_out.K_seq[ctrl_idx], ctrl_out.u_seq[ctrl_idx], ctrl_out.x_seq[ctrl_idx], x_sim_seq[ctrl_idx]) # type: ignore 
    data.ctrl  = (u_sim_seq[0]).reshape(-1)
    ctrl_count = 0  
    for sim_idx in range(sim_steps+1):
        if ctrl_count == steps_zoh:
            ctrl_count = 0  
            ctrl_idx  += 1               
            x_sim_seq[ctrl_idx] = get_state_vec(data) 
            if sim_idx < sim_steps:      
                u_sim_seq[ctrl_idx] = ilqr.calculate_u_star(ctrl_out.K_seq[ctrl_idx], ctrl_out.u_seq[ctrl_idx], ctrl_out.x_seq[ctrl_idx], x_sim_seq[ctrl_idx]) # type: ignore       
                data.ctrl           = (u_sim_seq[ctrl_idx]).reshape(-1)
            else:
                break 
        if len(frames) < data.time * framerate:  
            update_plt_frame(renderer, data, frames, img_set)
        # data.ctrl = 0.0        
        mujoco.mj_step(model, data)   
        ctrl_count += 1         
    model.opt.timestep = prev_timestep       
    return img_set, frames