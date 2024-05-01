import numpy as np
import numpy.typing as npt
from functools import partial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import numpy.typing as npt
import os
from typing import Optional

import dyn_functions as dyn
import analyze_ilqr_output_funcs as analyze

class double_pend_animation():
    def __init__(self,l1,l2,
                 x_seq:npt.NDArray[np.float64],
                 dt:float, fig:Figure,
                 u_seq:Optional[npt.NDArray[np.float64]] = None,
                 cost_seqs:Optional[npt.NDArray[np.float64]] = None) -> None:
        self.l1      = l1
        self.l2      = l2
        self.dt      = dt
        self.x_seq   = x_seq        
        self.fig     = fig
        self.min_fps:int = 30
        self.set_fps_for_animation()
        self.create_cartesian_sequence()    
        if u_seq is None or cost_seqs is None:
            self.config_for_ani()
            self.plt_w_ctrl_cost = False 
        else:
            self.config_for_ani_ctrl_cost()
            self.plt_w_ctrl_cost = True    

    def config_for_ani(self):
        L = self.l1 + self.l2
        self.ax1 = self.fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
        self.ax1.set_aspect('equal')
        self.ax1.grid()
        self.line, = self.ax1.plot([], [], 'o-', lw=2)
        self.trace, = self.ax1.plot([], [], '.-', lw=1, ms=2)
        self.time_template = 'time = %.2fs'
        self.time_text = self.ax1.text(0.05, 0.9, '', transform=self.ax1.transAxes)

    def config_for_ani_ctrl_cost(self):
        L = self.l1 + self.l2
        gs = gridspec.GridSpec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[:, 0],autoscale_on=False, xlim=(-L, L), ylim=(-L, L)) # row 0, col 0
        self.ax2 = self.fig.add_subplot(gs[0, 1]) # row 0, col 1
        self.ax3 = self.fig.add_subplot(gs[1, 1]) # row 1, span all columns
        # ax = self.fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
        self.ax1.set_aspect('equal')
        self.ax1.grid()
        self.line = self.ax1.plot([], [], 'o-', lw=2)
        self.trace = self.ax1.plot([], [], '.-', lw=1, ms=2)
        self.time_template = 'time = %.2fs'
        self.time_text = self.ax1.text(0.05, 0.9, '', transform=self.ax1.transAxes)

    def create_cartesian_sequence(self)->None:
        x1 =  self.l1*np.sin(self.x_seq_ani[:, 0])
        y1 = -self.l1*np.cos(self.x_seq_ani[:, 0])
        x2 =  self.l2*np.sin(self.x_seq_ani[:, 1]) + x1
        y2 = -self.l2*np.cos(self.x_seq_ani[:, 1]) + y1
        self.cartesian_vecs = (x1, y1, x2, y2)


    def animate_double_pend(self, i, line, trace, time_text):
        thisx = [0, (self.cartesian_vecs[0][i]).item(), (self.cartesian_vecs[2][i]).item()]
        thisy = [0, (self.cartesian_vecs[1][i]).item(), (self.cartesian_vecs[3][i]).item()]
        history_x = self.cartesian_vecs[2][:i]
        history_y = self.cartesian_vecs[3][:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(self.time_template % (i / self.fps))
        return line, trace, time_text
    
    def set_fps_for_animation(self):
        skipMod:int = 1
        if 1/(self.dt) / self.min_fps <= 1:
            pass
        else:
            skipMod = int(np.floor(1/(self.dt) / self.min_fps))
        self.x_seq_ani = self.x_seq[0::skipMod]
        self.fps:int = int(1 / (self.dt * skipMod))

    def update_fps_for_animation(self, new_min_fps):
        self.min_fps = new_min_fps
        self.set_fps_for_animation()
        self.create_cartesian_sequence()  

    def create_double_pend_animation(self):
        self.ani = animation.FuncAnimation(self.fig, partial(self.animate_double_pend, 
                                                line = self.line,
                                                trace = self.trace,
                                                time_text = self.time_text), 
                                                len(self.x_seq_ani), interval=self.fps, blit=True)

    def show_plot(self):
        # self.fig.show()
        plt.show()


    def save_animation_gif(self, filename: str | os.PathLike):
        writer = animation.PillowWriter(fps=self.fps,
                                    metadata=dict(artist='Me'))
        self.ani.save((filename+'.gif'), writer=writer)

    def save_animation_mp4(self, filename: str | os.PathLike):
        writer = animation.FFMpegWriter(fps=self.fps,
                                        metadata=dict(artist='Me'))
        self.ani.save((filename+'.mp4'), writer=writer)