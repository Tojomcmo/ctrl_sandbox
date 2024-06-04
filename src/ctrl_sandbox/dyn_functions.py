from jax import numpy as jnp

import ctrl_sandbox.gen_ctrl_funcs as gen_ctrl
import ctrl_sandbox.statespace_funcs as ss_funcs


class single_pm_pend_dyn:
    def __init__(self, g: float, b: float, l: float) -> None:
        self.g = g
        self.b = b
        self.l = l

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        continuous time dynamic equation for simple pendulum
        time[in]       - time component, necessary prarmeter for ode integration
        state[in]      - vector of state variables, 2 values,
                            [0]: theta, [1]: theta dot, pend down is zero
        control[in]    - vector of control variables, 1 value,
                            [0]: torque, positive torque causes
                            counter clockwise rotation (Right hand rule out of page)
        params[in]     - g: gravity[m/s^2] (positive down),
                            l: pend length[m], b: damping[Ns/m]
        state_dot[out] - vector of state derivatives, [0]: theta dot, [1]: theta ddot
        """
        x_jax = jnp.array(x_vec)
        u_jax = jnp.array(u_vec)
        state_dot = jnp.array(
            [
                x_vec[1],
                -(self.b / self.l) * x_jax[1]
                - (jnp.sin(x_jax[0]) * self.g / self.l)
                + u_jax[0],
            ]
        )
        return state_dot

    def cont_pend_dyn_lin_down(
        self, x_vec: jnp.ndarray, u_vec: jnp.ndarray
    ) -> jnp.ndarray:
        """
        continuous time dynamic equation for simple pendulum

        time[in]       - time component, necessary prarmeter for ode integration
        state[in]      - vector of state variables, 2 values,
                        [0]: theta, [1]: theta dot, pend down is zero
        control[in]    - vector of control variables, 1 value,
                        [0]: torque, positive torque causes
                        counter clockwise rotation (Right hand rule out of page)
        params[in]     - g: gravity[m/s^2] (positive down),
                        l: pend length[m], b: damping[Ns/m]
        state_dot[out] - vector of state derivatives, [0]: theta dot, [1]: theta ddot
        """
        x_jax = jnp.array(x_vec)
        u_jax = jnp.array(u_vec)
        state_dot = jnp.array(
            [
                x_jax[1],
                -((self.b / self.l) * x_jax[1])
                - ((self.g / self.l) * x_jax[0])
                + u_jax[0],
            ]
        )
        return state_dot

    def cont_lti_pend_ss_down(self) -> ss_funcs.stateSpace:
        A = jnp.array([[0, 1], [-self.g / self.l, -self.b / self.l]])
        B = jnp.array([[0], [1]])
        C = jnp.eye(2)
        D = jnp.zeros([2, 1])
        ss_cont = ss_funcs.stateSpace(A, B, C, D)
        return ss_cont


class single_pend_dyn:
    def __init__(self, m: float, moi: float, g: float, b: float, l: float) -> None:
        self.m = m
        self.moi = moi
        self.g = g
        self.b = b
        self.d = l
        self.create_lumped_parameters()

    def create_lumped_parameters(self):
        self.m_divide = 1 / (self.m * self.d**2 + self.moi)
        self.B_1 = self.b * self.m_divide
        self.G_1 = self.m * self.g * self.d * self.m_divide

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        continuous time dynamic equation for simple pendulum
        time[in]       - time component, necessary prarmeter for ode integration
        state[in]      - vector of state variables, 2 values,
                            [0]: theta, [1]: theta dot, pend down is zero
        control[in]    - vector of control variables, 1 value,
                            [0]: torque, positive torque causes
                            counter clockwise rotation (Right hand rule out of page)
        params[in]     - g: gravity[m/s^2] (positive down),
                            l: pend length[m], b: damping[Ns/m]
        state_dot[out] - vector of state derivatives, [0]: theta dot, [1]: theta ddot
        """
        state_dot = jnp.array(
            [
                x_vec[1],
                -self.B_1 * x_vec[1]
                - self.G_1 * jnp.sin(x_vec[0])
                + self.m_divide * u_vec[0],
            ]
        )
        return state_dot


class double_pm_pend_dyn:
    def __init__(
        self,
        g: float,
        m1: float,
        l1: float,
        m2: float,
        l2: float,
        b1: float,
        b2: float,
        shoulder_act: bool,
        elbow_act: bool,
    ):
        super().__init__()
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2
        self.shoulder_act = shoulder_act
        self.elbow_act = elbow_act
        # configure system matrices and static lumped terms
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def validate_inputs(self):
        # TODO fill in validation criteria
        pass

    def reconfigure_double_pend(self):
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def _configure_B_mat(self):
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, -1], [0, 1]], dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1], [0]], dtype=float)
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[-1], [1]], dtype=float)
        else:
            print(
                "double pendulum dynamics function is fully passive. Control array must be len(1,)"
            )
            self.B_mat = jnp.zeros([2, 1])

    def _calculate_lumped_terms(self):
        self.M_1 = (self.m1 + self.m2) * self.l1**2
        self.M_2 = self.m2 * self.l2**2
        self.M_3 = self.m2 * self.l1 * self.l2
        self.G_1 = (self.m1 + self.m2) * self.l1 * self.g
        self.G_2 = self.m2 * self.l2 * self.g

    def _calc_mass_matrix_cross_term(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        return self.M_3 * jnp.cos(x_vec[0] - x_vec[1])

    def calculate_mass_matrix(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        cross_term = self._calc_mass_matrix_cross_term(x_vec)
        return jnp.array([[self.M_1, cross_term], [cross_term, self.M_2]])

    def calculate_mass_matrix_inv(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        cross_term = self._calc_mass_matrix_cross_term(x_vec)
        return (1 / (self.M_1 * self.M_2 - cross_term**2)) * jnp.array(
            [[self.M_2, -cross_term], [-cross_term, self.M_1]]
        )

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        The state vector has form x = [th1, th2, thdot1, thdot2]'
        The control vector has form u = [tq1, tq2]'
        The output vector has form xdot = [thdot1, thdot2, thddot1, thddot2]
        helpful resource https://dassencio.org/33
        Derivation in github wiki https://github.com/Tojomcmo/ctrl_sandbox/wiki
        Thetas are both reference to intertial frame down position
        """
        x_jax = jnp.array(x_vec)
        u_jax = jnp.array(u_vec)
        s1m2 = jnp.sin(x_jax[0] - x_jax[1])

        m_inv_mat = self.calculate_mass_matrix_inv(x_vec)
        gen_dyn_mat = jnp.array(
            [
                [
                    -self.M_3 * x_jax[3] ** 2 * s1m2
                    - (self.b1 + self.b2) * x_jax[2]
                    + (self.b2) * x_jax[3]
                ],
                [
                    self.M_3 * x_jax[2] ** 2 * s1m2
                    + (self.b2) * x_jax[2]
                    - (self.b2) * x_jax[3]
                ],
            ]
        )

        grav_mat = jnp.array(
            [[-self.G_1 * jnp.sin(x_jax[0])], [-self.G_2 * jnp.sin(x_jax[1])]]
        )

        theta_ddots = (
            m_inv_mat @ (gen_dyn_mat + grav_mat + self.B_mat @ u_jax.reshape(-1, 1))
        ).reshape(-1)
        state_dot = jnp.array([x_jax[2], x_jax[3], theta_ddots[0], theta_ddots[1]])
        return state_dot

    def calculate_potential_energy(self, x_vec: jnp.ndarray) -> float:
        pot_energy_m1 = -self.g * (self.m1 + self.m2) * self.l1 * jnp.cos(x_vec[0])
        pot_energy_m2 = -self.g * self.m2 * self.l2 * jnp.cos(x_vec[1])
        return (pot_energy_m1 + pot_energy_m2).item()

    def calculate_kinetic_energy_2(self, x_vec: jnp.ndarray) -> float:
        return (
            (0.5 * (self.m1 + self.m2) * self.l1**2 * (x_vec[2]) ** 2)
            + (0.5 * (self.m2) * self.l2**2 * (x_vec[3]) ** 2)
            + (
                self.m2
                * self.l1
                * self.l2
                * x_vec[2]
                * x_vec[3]
                * jnp.cos(x_vec[0] - x_vec[1])
            )
        ).item()

    def calculate_kinetic_energy(self, x_vec: jnp.ndarray) -> float:
        mass_mat = self.calculate_mass_matrix(jnp.array(x_vec))
        x_vec_col = x_vec[2:].reshape(-1, 1)
        return ((0.5) * x_vec_col.T @ mass_mat @ x_vec_col).item()

    def calculate_total_energy(self, x_vec: jnp.ndarray) -> float:
        return self.calculate_kinetic_energy(x_vec) + self.calculate_potential_energy(
            x_vec
        )


class double_pend_abs_dyn:
    """
    Double pendulum object with absolute angle definitions
    - https://github.com/Tojomcmo/ctrl_sandbox.wiki.git
    - https://github.com/Tojomcmo/ctrl_sandbox/wiki/
        2024%E2%80%9005%E2%80%9001-double-pendulum-with-MoI-EoM-derivation
    """

    def __init__(
        self,
        g: float,
        m1: float,
        moi1: float,
        d1: float,
        l1: float,
        m2: float,
        moi2: float,
        d2: float,
        l2: float,
        b1: float,
        b2: float,
        shoulder_act: bool,
        elbow_act: bool,
    ):
        # populate gravity parameter
        self.g = g
        # populate inertial parameters
        self.m1 = m1
        self.moi1 = moi1
        self.m2 = m2
        self.moi2 = moi2
        # populate length parameters
        self.d1 = d1
        self.l1 = l1
        self.d2 = d2
        self.l2 = l2
        # populate damping parameters
        self.b1 = b1
        self.b2 = b2
        # set actuation boolean states
        self.shoulder_act = shoulder_act
        self.elbow_act = elbow_act
        # configure system matrices and static lumped terms
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def validate_inputs(self):
        # TODO fill in validation criteria
        pass

    def reconfigure_double_pend(self):
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def _configure_B_mat(self):
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, -1], [0, 1]], dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1], [0]], dtype=float)
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[-1], [1]], dtype=float)
        else:
            print("No control input. Control array must be len(1,)")
            self.B_mat = jnp.zeros([2, 1])

    def _calculate_lumped_terms(self):
        self.M_1 = (self.m1 * self.d1**2) + (self.m2 * self.l1**2) + self.moi1
        self.M_2 = (self.m2 * self.d2**2) + self.moi2
        self.M_3 = self.m2 * self.l1 * self.d2
        self.G_1 = (self.m1 * self.d1 + self.m2 * self.l1) * self.g
        self.G_2 = self.m2 * self.d2 * self.g

    def _calc_mass_matrix_cross_term(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        return self.M_3 * jnp.cos(x_vec[0] - x_vec[1])

    def calculate_mass_matrix(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        cross_term = self._calc_mass_matrix_cross_term(x_vec)
        return jnp.array([[self.M_1, cross_term], [cross_term, self.M_2]])

    def calculate_mass_matrix_inv(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        cross_term = self._calc_mass_matrix_cross_term(x_vec)
        return (1 / (self.M_1 * self.M_2 - cross_term**2)) * jnp.array(
            [[self.M_2, -cross_term], [-cross_term, self.M_1]]
        )

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        The state vector has form x = [th1, th2, thdot1, thdot2]'
        The control vector has form u = [tq1, tq2]'
        The output vector has form xdot = [thdot1, thdot2, thddot1, thddot2]
        helpful resource https://dassencio.org/33
        Derivation in github wiki https://github.com/Tojomcmo/ctrl_sandbox/wiki
        Thetas are both reference to inertial frame down position
        """
        s1m2 = jnp.sin(x_vec[0] - x_vec[1])

        mass_mat_inv = self.calculate_mass_matrix_inv(x_vec)

        gen_dyn_mat = jnp.array(
            [
                [
                    -self.M_3 * x_vec[3] ** 2 * s1m2
                    - (self.b1 + self.b2) * x_vec[2]
                    + (self.b2) * x_vec[3]
                ],
                [
                    self.M_3 * x_vec[2] ** 2 * s1m2
                    + (self.b2) * x_vec[2]
                    - (self.b2) * x_vec[3]
                ],
            ]
        )

        grav_mat = jnp.array(
            [[-self.G_1 * jnp.sin(x_vec[0])], [-self.G_2 * jnp.sin(x_vec[1])]]
        )

        theta_ddots = (
            mass_mat_inv @ (gen_dyn_mat + grav_mat + self.B_mat @ u_vec.reshape(-1, 1))
        ).reshape(-1)
        state_dot = jnp.array([x_vec[2], x_vec[3], theta_ddots[0], theta_ddots[1]])
        return state_dot

    def calculate_kinetic_energy(self, x_vec: jnp.ndarray) -> float:
        mass_mat = self.calculate_mass_matrix(jnp.array(x_vec))
        x_vec_col = x_vec[2:].reshape(-1, 1)
        return ((0.5) * x_vec_col.T @ mass_mat @ x_vec_col).item()

    def calculate_potential_energy(self, x_vec: jnp.ndarray) -> float:
        return (
            -self.g
            * (
                self.m1 * self.d1 * jnp.cos(x_vec[0])
                + self.m2 * (self.l1 * jnp.cos(x_vec[0]) + self.d2 * jnp.cos(x_vec[1]))
            ).item()
        )

    def calculate_total_energy(self, x_vec: jnp.ndarray) -> float:
        return self.calculate_kinetic_energy(x_vec) + self.calculate_potential_energy(
            x_vec
        )


class double_pend_rel_dyn:
    """
    Double pendulum object with relative angle definitions
    - https://github.com/Tojomcmo/ctrl_sandbox.wiki.git
    - https://github.com/Tojomcmo/ctrl_sandbox/wiki/
        2024%E2%80%9005%E2%80%9001-double-pendulum-with-MoI-EoM-derivation
    """

    def __init__(
        self,
        g: float,
        m1: float,
        moi1: float,
        d1: float,
        l1: float,
        m2: float,
        moi2: float,
        d2: float,
        l2: float,
        b1: float,
        b2: float,
        shoulder_act: bool,
        elbow_act: bool,
    ):
        # populate gravity parameter
        self.g = g
        # populate inertial parameters
        self.m1 = m1
        self.moi1 = moi1
        self.m2 = m2
        self.moi2 = moi2
        # populate length parameters
        self.d1 = d1
        self.l1 = l1
        self.d2 = d2
        self.l2 = l2
        # populate damping parameters
        self.b1 = b1
        self.b2 = b2
        # set actuation boolean states
        self.shoulder_act = shoulder_act
        self.elbow_act = elbow_act
        # configure system matrices and static lumped terms
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def validate_inputs(self):
        # TODO fill in validation criteria
        pass

    def reconfigure_double_pend(self):
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def _configure_B_mat(self):
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, 0], [0, 1]], dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1], [0]], dtype=float)
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[0], [1]], dtype=float)
        else:
            print("No control input. Control array must be len(1,)")
            self.B_mat = jnp.zeros([2, 1])

    def _calculate_lumped_terms(self):
        self.M_1 = (
            (self.m1 * self.d1**2)
            + (self.m2 * self.l1**2)
            + (self.m2 * self.d2**2)
            + self.moi1
            + self.moi2
        )
        self.M_2 = (self.m2 * self.d2**2) + self.moi2
        self.M_3 = self.m2 * self.l1 * self.d2
        self.G_1 = (self.m1 * self.d1 + self.m2 * self.l1) * self.g
        self.G_2 = self.m2 * self.d2 * self.g

    def _calc_state_dep_mass_term(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        return self.M_3 * jnp.cos(x_vec[1])

    def calculate_mass_matrix(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        M3c2 = self._calc_state_dep_mass_term(x_vec)
        return jnp.array(
            [[self.M_1 + 2 * M3c2, self.M_2 + M3c2], [self.M_2 + M3c2, self.M_2]]
        )

    def calculate_mass_matrix_inv(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        M3c2 = self._calc_state_dep_mass_term(x_vec)
        return (
            1 / ((self.M_1 + 2 * M3c2) * self.M_2 - (self.M_2 + M3c2) ** 2)
        ) * jnp.array(
            [[self.M_2, -(self.M_2 + M3c2)], [-(self.M_2 + M3c2), self.M_1 + 2 * M3c2]]
        )

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        The state vector has form x = [th1, th2, thdot1, thdot2]'
        The control vector has form u = [tq1, tq2]'
        The output vector has form xdot = [thdot1, thdot2, thddot1, thddot2]
        helpful resource https://dassencio.org/33
        Derivation in github wiki https://github.com/Tojomcmo/ctrl_sandbox/wiki
        """
        s2 = jnp.sin(x_vec[1])
        G2s12 = self.G_2 * jnp.sin(x_vec[0] + x_vec[1])
        mass_mat_inv = self.calculate_mass_matrix_inv(x_vec)
        C_mat = jnp.array(
            [
                [
                    self.M_3 * s2 * x_vec[3] * (2 * x_vec[2] + x_vec[3])
                    - (self.b1) * x_vec[2]
                ],
                [-self.M_3 * s2 * x_vec[2] ** 2 - (self.b2) * x_vec[3]],
            ]
        )
        grav_mat = jnp.array([[-self.G_1 * jnp.sin(x_vec[0]) - G2s12], [-G2s12]])
        theta_ddots = (
            mass_mat_inv @ (C_mat + grav_mat + self.B_mat @ u_vec.reshape(-1, 1))
        ).reshape(-1)
        state_dot = jnp.array([x_vec[2], x_vec[3], theta_ddots[0], theta_ddots[1]])
        return state_dot

    def calculate_kinetic_energy(self, x_vec: jnp.ndarray) -> float:
        mass_mat = self.calculate_mass_matrix(jnp.array(x_vec))
        x_vec_col = x_vec[2:].reshape(-1, 1)
        return ((0.5) * x_vec_col.T @ mass_mat @ x_vec_col).item()

    def calculate_potential_energy(self, x_vec: jnp.ndarray) -> float:
        return (
            -self.G_1 * jnp.cos(x_vec[0]) - self.G_2 * jnp.cos(x_vec[0] + x_vec[1])
        ).item()

    def calculate_total_energy(self, x_vec: jnp.ndarray) -> float:
        return self.calculate_kinetic_energy(x_vec) + self.calculate_potential_energy(
            x_vec
        )

    def get_animate_value_dict(self) -> dict:
        animate_vals = {"l1": self.l1, "l2": self.l2, "d1": self.d1, "d2": self.d2}
        return animate_vals


class ua_double_pend_rel_dyn:
    """
    Double pendulum object from underactuated lecture notes
    https://underactuated.mit.edu/multibody.html#Jain11
    """

    def __init__(
        self,
        g: float,
        m1: float,
        l1: float,
        m2: float,
        l2: float,
        shoulder_act: bool,
        elbow_act: bool,
    ):
        # populate gravity parameter
        self.g = g
        # populate inertial parameters
        self.m1 = m1
        self.m2 = m2
        # populate length parameters
        self.l1 = l1
        self.l2 = l2
        # set actuation boolean states
        self.shoulder_act = shoulder_act
        self.elbow_act = elbow_act
        # configure system matrices and static lumped terms
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def validate_inputs(self):
        # TODO fill in validation criteria
        pass

    def reconfigure_double_pend(self):
        self._configure_B_mat()
        self._calculate_lumped_terms()

    def _configure_B_mat(self):
        if self.shoulder_act is True and self.elbow_act is True:
            self.B_mat = jnp.array([[1, 0], [0, 1]], dtype=float)
        elif self.shoulder_act is True and self.elbow_act is False:
            self.B_mat = jnp.array([[1], [0]], dtype=float)
        elif self.shoulder_act is False and self.elbow_act is True:
            self.B_mat = jnp.array([[0], [1]], dtype=float)
        else:
            print("No control input. Control array must be len(1,)")
            self.B_mat = jnp.zeros([2, 1])

    def _calculate_lumped_terms(self):
        self.M_1 = (self.m1 + self.m2) * self.l1**2 + self.m2 * self.l1**2
        self.M_2 = self.m2 * self.l2**2
        self.M_3 = self.m2 * self.l1 * self.l2
        self.G_1 = (self.m1 + self.m2) * self.l1 * self.g
        self.G_2 = self.m2 * self.l2 * self.g

    def _calc_state_dep_mass_term(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        return self.M_3 * jnp.cos(x_vec[1])

    def calculate_mass_matrix(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        M3c2 = self._calc_state_dep_mass_term(x_vec)
        return jnp.array(
            [[self.M_1 + 2 * M3c2, self.M_2 + M3c2], [self.M_2 + M3c2, self.M_2]]
        )

    def calculate_mass_matrix_inv(self, x_vec: jnp.ndarray) -> jnp.ndarray:
        M3c2 = self._calc_state_dep_mass_term(x_vec)
        return (
            1 / ((self.M_1 + 2 * M3c2) * self.M_2 - (self.M_2 + M3c2) ** 2)
        ) * jnp.array(
            [[self.M_2, -(self.M_2 + M3c2)], [-(self.M_2 + M3c2), self.M_1 + 2 * M3c2]]
        )

    def cont_dyn_func(self, x_vec: jnp.ndarray, u_vec: jnp.ndarray) -> jnp.ndarray:
        """
        The state vector has form x = [th1, th2, thdot1, thdot2]'
        The control vector has form u = [tq1, tq2]'
        The output vector has form xdot = [thdot1, thdot2, thddot1, thddot2]
        """
        x_jax = jnp.array(x_vec)
        u_jax = jnp.array(u_vec)
        s2 = jnp.sin(x_jax[1])
        x_vel = x_jax[2:]
        mass_mat_inv = self.calculate_mass_matrix_inv(x_vec)
        C_mat = jnp.array(
            [
                [0, -self.M_3 * s2 * (2 * x_jax[2] + x_jax[3])],
                [
                    (0.5) * self.M_3 * s2 * (2 * x_jax[2] + x_jax[3]),
                    -(0.5) * self.M_3 * s2 * x_jax[2],
                ],
            ]
        )
        grav_mat = jnp.array(
            [
                [
                    -self.G_1 * jnp.sin(x_jax[0])
                    - self.G_2 * jnp.sin(x_jax[0] + x_jax[1])
                ],
                [-self.G_2 * jnp.sin(x_jax[0] + x_jax[1])],
            ]
        )
        theta_ddots = (
            mass_mat_inv
            @ (
                -(C_mat @ x_vel.reshape(-1, 1))
                + grav_mat
                + self.B_mat @ u_jax.reshape(-1, 1)
            )
        ).reshape(-1)
        state_dot = jnp.array([x_jax[2], x_jax[3], theta_ddots[0], theta_ddots[1]])
        return state_dot

    def calculate_kinetic_energy(self, x_vec: jnp.ndarray) -> float:
        mass_mat = self.calculate_mass_matrix(x_vec)
        x_vec_col = x_vec[2:].reshape(-1, 1)
        return ((0.5) * x_vec_col.T @ mass_mat @ x_vec_col).item()

    def calculate_potential_energy(self, x_vec: jnp.ndarray) -> float:
        return (
            -self.G_1 * jnp.cos(x_vec[0]) - self.G_2 * jnp.cos(x_vec[0] + x_vec[1])
        ).item()

    def calculate_total_energy(self, x_vec: jnp.ndarray) -> float:
        return self.calculate_kinetic_energy(x_vec) + self.calculate_potential_energy(
            x_vec
        )
