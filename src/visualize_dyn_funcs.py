import numpy as np
import numpy.typing as npt
from functools import partial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy.typing as npt

from . import dyn_functions as dyn
from typing import Tuple


class double_pend_animation():
    def __init__(self,double_pend_params:dyn.nlDoublePendParams,x_seq:npt.NDArray[np.float64],dt:float, fig:Figure) -> None:
        self.params = double_pend_params
        self.dt     = dt
        self.x_seq  = x_seq        
        self.fig    = fig
        self.framerate:int = 30

        self.cartesian_vecs:Tuple = self.create_cartesian_sequence()
        self.configure_figure()

    def configure_figure(self):
        L = self.params.l1 + self.params.l2
        ax = self.fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
        ax.set_aspect('equal')
        ax.grid()
        self.line, = ax.plot([], [], 'o-', lw=2)
        self.trace, = ax.plot([], [], '.-', lw=1, ms=2)
        self.time_template = 'time = %.1fs'
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def create_cartesian_sequence(self)->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        x1 =  self.params.l1*np.sin(self.x_seq[:, 0])
        y1 = -self.params.l1*np.cos(self.x_seq[:, 0])
        x2 =  self.params.l2*np.sin(self.x_seq[:, 1]) + x1
        y2 = -self.params.l2*np.cos(self.x_seq[:, 1]) + y1
        return (x1, y1, x2, y2)

    def animate_double_pend(self, i, line, trace, time_text):
        thisx = [0, (self.cartesian_vecs[0][i]).item(), (self.cartesian_vecs[2][i]).item()]
        thisy = [0, (self.cartesian_vecs[1][i]).item(), (self.cartesian_vecs[3][i]).item()]

        history_x = self.cartesian_vecs[2][:i]
        history_y = self.cartesian_vecs[3][:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(self.time_template % (i*self.dt))
        return line, trace, time_text
    

    def create_double_pend_animation(self):
        ani = animation.FuncAnimation(self.fig, partial(self.animate_double_pend, 
                                                line = self.line,
                                                trace = self.trace,
                                                time_text = self.time_text), 
                                                len(self.x_seq), interval=self.dt*1000, blit=True)
        return ani
