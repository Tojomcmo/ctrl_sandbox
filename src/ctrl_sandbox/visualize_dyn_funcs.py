import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from functools import partial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
import numpy.typing as npt
import os
from typing import Optional, Union, Tuple


import ctrl_sandbox.ilqr_funcs as ilqr


class pm_pend_animation:
    def __init__(
        self,
        l,
        x_seq: npt.NDArray[np.float64],
        dt: float,
        fig: Figure,
        u_seq: Optional[npt.NDArray[np.float64]] = None,
        cost_seqs: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        self.l = l
        self.dt = dt
        self.x_seq = x_seq
        self.fig = fig
        self.min_fps: int = 20
        self.set_fps_for_animation()
        self.create_cartesian_sequence()
        if u_seq is None or cost_seqs is None:
            self.config_for_ani()
            self.plt_w_ctrl_cost = False
        else:
            self.config_for_ani_ctrl_cost()
            self.plt_w_ctrl_cost = True

    def config_for_ani(self):
        L = self.l * 1.1
        self.ax1 = self.fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
        self.ax1.set_aspect("equal")
        self.ax1.grid()
        (self.line,) = self.ax1.plot([], [], "o-", lw=2)
        (self.trace,) = self.ax1.plot([], [], ".-", lw=1, ms=2)
        self.time_template = "time = %.2fs"
        self.time_text = self.ax1.text(0.05, 0.9, "", transform=self.ax1.transAxes)

    def config_for_ani_ctrl_cost(self):
        L = self.l
        gs = gridspec.GridSpec(2, 2)
        self.ax1 = self.fig.add_subplot(
            gs[:, 0], autoscale_on=False, xlim=(-L, L), ylim=(-L, L)
        )  # row 0, col 0
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # row 0, col 1
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # row 1, span all columns
        # ax = self.fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
        self.ax1.set_aspect("equal")
        self.ax1.grid()
        self.line = self.ax1.plot([], [], "o-", lw=2)
        self.trace = self.ax1.plot([], [], ".-", lw=1, ms=2)
        self.time_template = "time = %.2fs"
        self.time_text = self.ax1.text(0.05, 0.9, "", transform=self.ax1.transAxes)

    def create_cartesian_sequence(self) -> None:
        x1 = self.l * np.sin(self.x_seq_ani[:, 0])
        y1 = -self.l * np.cos(self.x_seq_ani[:, 0])
        self.cartesian_vecs = (x1, y1)

    def animate_double_pend(self, i, line, trace, time_text):
        thisx = [
            0,
            (self.cartesian_vecs[0][i]).item(),
        ]
        thisy = [
            0,
            (self.cartesian_vecs[1][i]).item(),
        ]
        history_x = self.cartesian_vecs[0][:i]
        history_y = self.cartesian_vecs[1][:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(self.time_template % (i / self.fps))
        return line, trace, time_text

    def set_fps_for_animation(self):
        skipMod: int = 1
        if 1 / (self.dt) / self.min_fps <= 1:
            pass
        else:
            skipMod = int(np.floor(1 / (self.dt) / self.min_fps))
        self.x_seq_ani = self.x_seq[0::skipMod]
        self.fps: int = int(1 / (self.dt * skipMod))

    def update_fps_for_animation(self, new_min_fps):
        self.min_fps = new_min_fps
        self.set_fps_for_animation()
        self.create_cartesian_sequence()

    def create_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            partial(
                self.animate_double_pend,
                line=self.line,
                trace=self.trace,
                time_text=self.time_text,
            ),
            len(self.x_seq_ani),
            interval=self.fps,
            blit=True,
        )

    def show_plot(self):
        # self.fig.show()
        plt.show()

    def save_animation_gif(self, filename: str | os.PathLike):
        writer = animation.PillowWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.gif'}"
        self.ani.save(full_filename, writer=writer)

    def save_animation_mp4(self, filename: str | os.PathLike):
        writer = animation.FFMpegWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.mp4'}"
        self.ani.save(full_filename, writer=writer)


class double_pm_pend_animation:
    def __init__(
        self,
        l1,
        l2,
        x_seq: npt.NDArray[np.float64],
        dt: float,
        fig: Figure,
        ax: Axes,
        th2="abs",
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.dt = dt
        self.x_seq = x_seq
        self.fig = fig
        self.ax = ax
        self.min_fps: int = 30
        self.set_fps_for_animation()
        if th2 == "abs":
            self.create_abs_cartesian_sequence()
        elif th2 == "rel":
            self.create_rel_cartesian_sequence()
        self.config_for_ani()

    def config_for_ani(self):
        L_ax = (self.l1 + self.l2) * 1.1
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-L_ax, L_ax)
        self.ax.set_ylim(-L_ax, L_ax)
        self.ax.set_autoscale_on(False)
        (self.line,) = self.ax.plot([], [], "o-", lw=2)
        (self.trace,) = self.ax.plot([], [], ".-", lw=1, ms=2)
        self.time_template = "time = %.2fs"
        self.time_text = self.ax.text(0.05, 0.9, "", transform=self.ax.transAxes)

    def create_abs_cartesian_sequence(self) -> None:
        x1 = self.l1 * np.sin(self.x_seq_ani[:, 0])
        y1 = -self.l1 * np.cos(self.x_seq_ani[:, 0])
        x2 = self.l2 * np.sin(self.x_seq_ani[:, 1]) + x1
        y2 = -self.l2 * np.cos(self.x_seq_ani[:, 1]) + y1
        self.cartesian_vecs = (x1, y1, x2, y2)

    def create_rel_cartesian_sequence(self) -> None:
        x1 = self.l1 * np.sin(self.x_seq_ani[:, 0])
        y1 = -self.l1 * np.cos(self.x_seq_ani[:, 0])
        x2 = self.l2 * np.sin(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + x1
        y2 = -self.l2 * np.cos(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + y1
        self.cartesian_vecs = (x1, y1, x2, y2)

    def animate_double_pend(self, i, line, trace, time_text):
        thisx = [
            0,
            (self.cartesian_vecs[0][i]).item(),
            (self.cartesian_vecs[2][i]).item(),
        ]
        thisy = [
            0,
            (self.cartesian_vecs[1][i]).item(),
            (self.cartesian_vecs[3][i]).item(),
        ]
        history_x = self.cartesian_vecs[2][:i]
        history_y = self.cartesian_vecs[3][:i]
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(self.time_template % (i / self.fps))
        return line, trace, time_text

    def set_fps_for_animation(self):
        skipMod: int = 1
        if 1 / (self.dt) / self.min_fps <= 1:
            pass
        else:
            skipMod = int(np.floor(1 / (self.dt) / self.min_fps))
        self.x_seq_ani = self.x_seq[0::skipMod]
        self.fps: int = int(1 / (self.dt * skipMod))

    def update_fps_for_animation(self, new_min_fps):
        self.min_fps = new_min_fps
        self.set_fps_for_animation()
        self.create_abs_cartesian_sequence()

    def create_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            partial(
                self.animate_double_pend,
                line=self.line,
                trace=self.trace,
                time_text=self.time_text,
            ),
            len(self.x_seq_ani),
            interval=self.fps,
            blit=True,
        )

    def show_plot(self):
        plt.show()

    def save_animation_gif(self, filename: str | os.PathLike):
        writer = animation.PillowWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.gif'}"
        self.ani.save(full_filename, writer=writer)

    def save_animation_mp4(self, filename: str | os.PathLike):
        writer = animation.FFMpegWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.mp4'}"
        self.ani.save(full_filename, writer=writer)


class double_pend_animation:
    def __init__(
        self,
        animate_vals: dict,
        x_seq: npt.NDArray[np.float64],
        dt: float,
        fig: Figure,
        ax: Axes,
        th2="abs",
    ) -> None:
        self.l1 = animate_vals["l1"]
        self.l2 = animate_vals["l2"]
        self.d1 = animate_vals["d1"]
        self.d2 = animate_vals["d2"]
        self.dt = dt
        self.x_seq = x_seq
        self.fig = fig
        self.ax = ax
        self.min_fps: int = 30
        self.set_fps_for_animation()
        if th2 == "abs":
            self.create_abs_cartesian_sequence()
        elif th2 == "rel":
            self.create_rel_cartesian_sequence()
        self.config_for_ani()

    def config_for_ani(self):
        L = self.l1 + self.l2
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_autoscale_on(False)
        (self.origin,) = self.ax.plot([], [], marker="s", ms=8, color="black")
        (self.line,) = self.ax.plot([], [], "o-", lw=2, ms=3)
        (self.CoMs,) = self.ax.plot([], [], "ro", ms=8)
        (self.trace,) = self.ax.plot([], [], ".-", lw=1, ms=2)
        self.time_template = "time = %.2fs"
        self.time_text = self.ax.text(0.05, 0.9, "", transform=self.ax.transAxes)

    def create_abs_cartesian_sequence(self) -> None:
        x1 = self.d1 * np.sin(self.x_seq_ani[:, 0])
        y1 = -self.d1 * np.cos(self.x_seq_ani[:, 0])
        x2 = self.l1 * np.sin(self.x_seq_ani[:, 0])
        y2 = -self.l1 * np.cos(self.x_seq_ani[:, 0])
        x3 = self.d2 * np.sin(self.x_seq_ani[:, 1]) + x2
        y3 = -self.d2 * np.cos(self.x_seq_ani[:, 1]) + y2
        x4 = self.l2 * np.sin(self.x_seq_ani[:, 1]) + x2
        y4 = -self.l2 * np.cos(self.x_seq_ani[:, 1]) + y2
        self.cartesian_vecs = (x1, x2, x3, x4, y1, y2, y3, y4)

    def create_rel_cartesian_sequence(self) -> None:
        x1 = self.d1 * np.sin(self.x_seq_ani[:, 0])
        y1 = -self.d1 * np.cos(self.x_seq_ani[:, 0])
        x2 = self.l1 * np.sin(self.x_seq_ani[:, 0])
        y2 = -self.l1 * np.cos(self.x_seq_ani[:, 0])
        x3 = self.d2 * np.sin(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + x2
        y3 = -self.d2 * np.cos(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + y2
        x4 = self.l2 * np.sin(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + x2
        y4 = -self.l2 * np.cos(self.x_seq_ani[:, 0] + self.x_seq_ani[:, 1]) + y2
        self.cartesian_vecs = (x1, x2, x3, x4, y1, y2, y3, y4)

    def animate_double_pend(self, i, origin, line, CoMs, trace, time_text):
        thisx = [
            0,
            (self.cartesian_vecs[0][i]).item(),
            (self.cartesian_vecs[1][i]).item(),
            (self.cartesian_vecs[2][i]).item(),
            (self.cartesian_vecs[3][i]).item(),
        ]
        thisy = [
            0,
            (self.cartesian_vecs[4][i]).item(),
            (self.cartesian_vecs[5][i]).item(),
            (self.cartesian_vecs[6][i]).item(),
            (self.cartesian_vecs[7][i]).item(),
        ]
        thisCoMx = [
            (self.cartesian_vecs[0][i]).item(),
            (self.cartesian_vecs[2][i]).item(),
        ]
        thisCoMy = [
            (self.cartesian_vecs[4][i]).item(),
            (self.cartesian_vecs[6][i]).item(),
        ]
        history_x = self.cartesian_vecs[3][:i]
        history_y = self.cartesian_vecs[7][:i]
        origin.set_data([0], [0])
        line.set_data(thisx, thisy)
        CoMs.set_data(thisCoMx, thisCoMy)
        trace.set_data(history_x, history_y)
        time_text.set_text(self.time_template % (i / self.fps))
        return line, origin, CoMs, trace, time_text

    def set_fps_for_animation(self):
        skipMod: int = 1
        if 1 / (self.dt) / self.min_fps <= 1:
            pass
        else:
            skipMod = int(np.floor(1 / (self.dt) / self.min_fps))
        self.x_seq_ani = self.x_seq[0::skipMod]
        self.fps: int = int(1 / (self.dt * skipMod))

    def update_fps_for_animation(self, new_min_fps):
        self.min_fps = new_min_fps
        self.set_fps_for_animation()
        self.create_abs_cartesian_sequence()

    def create_double_pend_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            partial(
                self.animate_double_pend,
                origin=self.origin,
                line=self.line,
                CoMs=self.CoMs,
                trace=self.trace,
                time_text=self.time_text,
            ),
            len(self.x_seq_ani),
            interval=self.fps,
            blit=True,
        )

    def show_plot(self):
        # self.fig.show()
        plt.show()

    def save_animation_gif(self, filename: str | os.PathLike):
        writer = animation.PillowWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.gif'}"
        self.ani.save(full_filename, writer=writer)

    def save_animation_mp4(self, filename: str | os.PathLike):
        writer = animation.FFMpegWriter(fps=self.fps, metadata=dict(artist="Me"))
        full_filename = f"{filename}{'.mp4'}"
        self.ani.save(full_filename, writer=writer)


def plot_ilqr_dpend_act_and_cost_axes(
    shoulder_act: bool,
    elbow_act: bool,
    ctrl_out: ilqr.ilqrControllerState,
    u_sim_seq,
    ax2: Axes,
    ax3: Axes,
):
    if shoulder_act is True and elbow_act is True:
        ax2.plot(
            ctrl_out.time_seq[:-1],
            ctrl_out.u_seq[:, 0],
            label="shoulder control ff torque",
        )
        ax2.plot(
            ctrl_out.time_seq[:-1], u_sim_seq[:, 0], label="shoulder control sim effort"
        )
        ax2.plot(
            ctrl_out.time_seq[:-1],
            ctrl_out.u_seq[:, 1],
            label="elbow control ff effort",
        )
        ax2.plot(
            ctrl_out.time_seq[:-1], u_sim_seq[:, 1], label="elbow control sim effort"
        )

        ax2.legend()
    else:
        ax2.plot(ctrl_out.time_seq[:-1], u_sim_seq, label="control effort")
        ax2.plot(ctrl_out.time_seq[:-1], ctrl_out.u_seq, label="control ff effort")
        ax2.legend()
    ax3.plot(ctrl_out.time_seq, ctrl_out.cost_seq, label="total cost")
    ax3.plot(ctrl_out.time_seq, ctrl_out.x_cost_seq, label="state cost")
    ax3.plot(ctrl_out.time_seq, ctrl_out.u_cost_seq, label="control cost")
    ax3.legend()


def plot_dpend_control_to_ax(
    ax: Axes,
    x_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    u_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    time_vec: Union[jnp.ndarray, npt.NDArray[np.float64]],
    shoulder_ctrl_bool: bool,
    elbow_ctrl_bool: bool,
):
    if shoulder_ctrl_bool is True and elbow_ctrl_bool is True:
        ax.plot(
            time_vec[:-1], u_seq[:, 0], label="shoulder ctrl", color="r", marker="o"
        )
        ax.plot(
            time_vec[:-1], u_seq[:, 1], label="elbow control", color="b", marker="s"
        )
    else:
        if shoulder_ctrl_bool is True:
            ax.plot(
                time_vec[:-1], u_seq, label="shoulder control", color="r", marker="o"
            )
        else:
            ax.plot(time_vec[:-1], u_seq, label="elbow control", color="r", marker="o")
    ax.set_title("control input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [Nm]")
    ax.grid(True)
    ax.legend()


def plot_dpend_pos_to_ax(
    ax: Axes,
    x_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    time_vec: Union[jnp.ndarray, npt.NDArray[np.float64]],
):
    ax.plot(time_vec, x_seq[:, 0], label="th1", marker="o")
    ax.plot(time_vec, x_seq[:, 1], label="th2", marker="s")
    ax.set_title("position states")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Positions [rad]")
    ax.grid(True)
    ax.legend()


def plot_dpend_state_control_to_fig(
    fig: Figure,
    x_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    u_seq: Union[jnp.ndarray, npt.NDArray[np.float64]],
    time_vec: Union[jnp.ndarray, npt.NDArray[np.float64]],
    shoulder_ctrl_bool: bool,
    elbow_ctrl_bool: bool,
):
    ax1 = fig.add_subplot(2, 1, 1)
    plot_dpend_pos_to_ax(ax1, x_seq, time_vec)
    ax2 = fig.add_subplot(2, 1, 2)
    plot_dpend_control_to_ax(
        ax2, x_seq, u_seq, time_vec, shoulder_ctrl_bool, elbow_ctrl_bool
    )
    plt.tight_layout()
