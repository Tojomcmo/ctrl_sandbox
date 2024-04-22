import numpy as np
import matplotlib.pyplot as plt

from src import dyn_functions as dyn
from src import visualize_dyn_funcs as vis_dyn
from src import gen_ctrl_funcs as gen_ctrl


if __name__== "__main__":
    dt = 0.01
    len_seq = 2000
    pend_params = dyn.nlDoublePendParams(g=9.81, m1=1.0, l1=1.0, m2=1.0, l2=1.0, b1=100.0, b2=1.0)
    dyn_func = lambda x, u: dyn.double_pend_no_damp_full_act_dyn(pend_params, x, u)
    x_init = np.array([[1.0],[1.0],[1.0],[1.0]])
    u_vec = np.array([[0.0],[2.0]]) 

    x_seq = np.zeros((len_seq, 4, 1))
    x_seq[0]= x_init
    for k in range(len_seq-1):
        x_seq[k+1] = gen_ctrl.step_rk4(dyn_func, dt, x_seq[k], u_vec)

    fig = plt.figure(figsize=[5,4])
    pend_animation = vis_dyn.double_pend_animation(pend_params, x_seq, dt,fig)
    ani = pend_animation.create_double_pend_animation()
    plt.show()