import numpy as np
import control as ct


if __name__ == '__main__':
    # Define the matrix A
    A = np.array([[5, 1], [3, 3]])
    B = np.array([[4],[6]])
    C = np.array([[1, 0], [0, 1]])
    D = np.array([[0],[0]])

    T1 = 0.1
    T2 = 0.2
    # Create a discrete-time state-space system
    sys_discrete_original = ct.ss(A, B, C, D, dt=T1)

    # Convert the discrete-time system to continuous-time
    sys_continuous = sys_discrete_original.to_continuous()

    # Convert the continuous-time system back to discrete-time with the new timestep
    sys_discrete_new = sys_continuous.to_discrete(T2, method='zoh')  # 'zoh' is zero-order hold. You can also use 'tustin'.

    # Extract the new A, B, C, D matrices
    A_new = sys_discrete_new.A
    B_new = sys_discrete_new.B
    C_new = sys_discrete_new.C
    D_new = sys_discrete_new.D
