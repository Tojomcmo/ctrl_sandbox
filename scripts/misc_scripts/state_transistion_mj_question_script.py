import numpy as np
import mujoco as mujoco
from scipy.signal import StateSpace

# TODO Fix the statespace discretization statements which are commented out.


class undamped_simple_pend_dyn:
    def __init__(self, g: float, l: float) -> None:
        self.g = g
        self.l = l

    def cont_dyn_func(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """non-linear continuous time pendulum dynamics"""
        state_dot = np.array(
            [state[1], -(np.sin(state[0]) * self.g / self.l) + control[0]]
        )
        return state_dot

    def cont_lti_pend_ss_down(self) -> StateSpace:
        """linearized continuous time pendulum dynamics about the down position"""
        A = np.array([[0, 1], [-self.g / self.l, 0]])
        B = np.array([[0], [1]])
        C = np.eye(2)
        D = np.zeros([2, 1])
        ss_cont = StateSpace(A, B, C, D)
        return ss_cont


def step_rk4(cont_dyn_func, dt, state_k, control_k):
    # rk4 integration with zero-order hold on u
    f1 = cont_dyn_func(state_k, control_k)
    f2 = cont_dyn_func(state_k + 0.5 * dt * f1, control_k)
    f3 = cont_dyn_func(state_k + 0.5 * dt * f2, control_k)
    f4 = cont_dyn_func(state_k + dt * f3, control_k)
    return state_k + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


# define pendulum of approximately point mass, length 1.0, -9.81 gravity
mjcf_pend_model = """<mujoco>
                        <option gravity="0 0 -1.0" integrator="RK4">
                        </option>
                        <worldbody>
                            <body name="pendulum1" pos="0 0 0">
                                <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                                <geom type="cylinder" size="0.05 0.5" pos="0 0 -1.0"/>
                                <inertial pos="0 0 -1.0" mass="1" diaginertia="0.00001 0.00001 0.000001"/>/>
                            </body>   
                        </worldbody>
                        <actuator>
                            <motor joint="joint1" name="shoulder_motor" gear="1" ctrllimited="false"/>                    
                        </actuator>
                    </mujoco>
                    """

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    dt = 0.01
    dt_mj = dt
    state_lin = np.array([0.0, 0.0])
    control_lin = np.array([0.0])
    state_k = state_lin
    control_k = np.array([10.0])

    mj_model = mujoco.MjModel.from_xml_string(mjcf_pend_model)
    mj_model.opt.timestep = dt_mj
    mj_data = mujoco.MjData(mj_model)
    nu = mj_model.nu
    nx = mj_model.nv

    pend_sys = undamped_simple_pend_dyn(g=1.0, l=1.0)
    ss_cont = pend_sys.cont_lti_pend_ss_down()
    # ss_disc = ss_cont.to_discrete(dt)

    # --------- compare state transition and control matrices ------------#
    print("/------compare transition matrices--------/")
    # get discretized linear dynamic model of pendulum
    ss_cont = pend_sys.cont_lti_pend_ss_down()
    # ss_disc = ss_cont.to_discrete(dt)

    # perform finite difference mujoco function to obtain transition and control matrices
    eps = 1e-6
    flg_centered = False
    A_mj = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    B_mj = np.zeros((2 * mj_model.nv, mj_model.nu))
    mj_data.qpos = state_lin[0]
    mj_data.qvel = state_lin[1]
    mj_data.ctrl = control_lin[0]
    mujoco.mj_forward(mj_model, mj_data)
    mujoco.mjd_transitionFD(
        mj_model, mj_data, eps, flg_centered, A_mj, B_mj, None, None
    )

    # print("linear discrete model A matrix: \n", ss_disc.A)
    print("mujoco transition FD A matrix: \n", A_mj)
    # print("diff A matrices: \n", (ss_disc.A - A_mj))
    # print("linear discrete model B matrix: \n", ss_disc.B)
    print("mujoco transition FD B matrix: \n", B_mj)

    # --------- compare next states ------------#
    print("/--------compare next states------------/")

    # Calculate x_kp1 = A_d@x_k + B_d@u_k state propogation for state space systems
    x_kp1_pred_mj = A_mj @ state_k.reshape(-1, 1) + B_mj @ control_k.reshape(-1, 1)
    # x_kp1_pred_lin = ss_disc.A @ state_k.reshape(-1, 1) + ss_disc.B @ control_k.reshape(
    # -1, 1
    # )

    # perform RK4 step on non-linear dynamics to compare with linearized dynamics
    x_kp1_pred_nl = step_rk4(pend_sys.cont_dyn_func, dt, state_k, control_k)

    # step mujoco forward to compare with transition_FD output
    mj_data.ctrl = control_k[0]
    mujoco.mj_step(mj_model, mj_data)
    x_kp1_mj = (np.array([mj_data.qpos, mj_data.qvel])).reshape(-1)

    # print("lin sys step x(kp1|k): \n", x_kp1_pred_lin.reshape(-1))
    print("rk4 on nl dyn x(kp1|k): \n", x_kp1_pred_nl.reshape(-1))
    print("mj predicted x(kp1|k): \n", x_kp1_pred_mj.reshape(-1))
    print("mj simulated step x_kp1: \n", x_kp1_mj)
