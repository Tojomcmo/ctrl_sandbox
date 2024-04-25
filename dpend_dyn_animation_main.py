import numpy as np
import matplotlib.pyplot as plt
import os

from src import dyn_functions as dyn
from src import visualize_dyn_funcs as vis_dyn
from src import gen_ctrl_funcs as gen_ctrl


if __name__== "__main__":
    save_ani_bool = False
    lock_arm_num = 0
    dt = 0.01
    len_seq = 1500
    pend_params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=0, b2=0)
    dyn_func = lambda x, u: dyn.double_pend_no_damp_full_act_dyn(pend_params, x, u)
    x_init = np.array([[3.14],[3.14],[0.0],[0.0]])
    u_vec = np.array([[0.0],[0.0]]) 

    x_seq = np.zeros((len_seq, 4, 1))
    x_seq[0]= x_init
    for k in range(len_seq-1):
        if lock_arm_num == 1:
            x_seq[k,0] = 0.0
            x_seq[k,2] = 0.0
        elif lock_arm_num == 2:    
            x_seq[k,1] = x_seq[k,0]
            x_seq[k,3] = x_seq[k,2]
        x_seq[k+1] = gen_ctrl.step_rk4(dyn_func, dt, x_seq[k], u_vec)

    fig = plt.figure(figsize=[10,8])
    pend_animation = vis_dyn.double_pend_animation(pend_params, x_seq, dt,fig)
    pend_animation.create_double_pend_animation()
    pend_animation.show_animation()

    if save_ani_bool == True: 
        print('saving animation...')
        filename:str | os.PathLike = '/Users/thomasmoriarty/Desktop/dpend_passive'
        pend_animation.save_animation_gif(filename)
        print('animation saved!')

