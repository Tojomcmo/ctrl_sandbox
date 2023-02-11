#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from scipy.integrate import solve_ivp

class ilqrController(object):
    def __init__(self, state_dim, control_dim, state_transition_func, cost_func, state_init,
                 control_init, max_iter, time_step, final_time):
        self.state_dim     = state_dim
        self.control_dim   = control_dim
        self.fx            = state_transition_func
        self.cost_func     = cost_func
        self.state_init    = state_init
        self.control_init  = control_init
        self.max_iter      = max_iter
        self.time_step     = time_step
        self.final_time    = final_time

class stateSpace:
    def __init__(self, a, b, c, d, time_step = None):
        if a.shape[0] != a.shape[1]:
            raise ValueError('A matrix must be square')
        elif a.shape[0] != b.shape[0]:
            raise ValueError('A and B matrices must have same n(state) dimension')
        elif a.shape[0] != c.shape[1]:
            raise ValueError('A and C matrices must have same m(state) dimension')
        elif b.shape[1] != d.shape[1]:
            raise ValueError('B and D matrices must have the same m(control) dimension')
        elif c.shape[0] != d.shape[0]:
            raise ValueError('C and D matrices must have the same n(measurement) dimension')
        else:
            if time_step is None:
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.time_step = time_step
                self.type = 'continuous'
            elif isinstance(time_step, float) and time_step > 0:
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.time_step = time_step
                self.type = 'discrete'
            else:
                raise ValueError('invalid time step definition. time_step must be a positive float or None')

def calculate_forward_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step, **kwargs):
    # simulate forward dynamics
    # Linearize dynamics at each time step
    # calculate discretized linearized state space for each time step
    # return
    # **kwargs:
    # - sim_method('Euler', 'solve_ivp_zoh')
    # - c2d_method('Euler', 'zoh', 'zohCombined')
    state_seq, time_seq  = simulate_forward_dynamics(dyn_func_with_params, state_init, control_seq, time_step, **kwargs)
    total_cost, cost_seq = calculate_cost_seq_and_total_cost(cost_func, state_seq, control_seq)
    return state_seq, time_seq, cost_seq, total_cost

def calculate_initial_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step, **kwargs):
    state_seq, time_seq, cost_seq, total_cost = calculate_forward_rollout(dyn_func_with_params, cost_func, state_init, control_seq, time_step, **kwargs)
    ad_seq, bd_seq                            = calculate_linearized_state_space_seq(dyn_func_with_params, state_seq, control_seq, time_step, **kwargs)
    return ad_seq, bd_seq, time_seq

def calculate_backwards_pass():
#   du_optimal = argmin over delu of Q(delx,delu) = k + (K * delx)
#   k = - inv(q_uu) * q_u
#   K = - inv(q_uu) * q_ux
#   V_N = l(x_N)
#   Vxx = q_xx -       transpose(K) * q_uu * K
#   Vx  = q_x  -       transpose(K) * q_uu * k
#   DV  =      - 1/2 * transpose(k) * q_uu * k
    unew = 0
    v = 0
    print("this is the backward pass 2")
    return unew, v


def simulate_forward_dynamics(dyn_func, state_init, control_seq, time_step, sim_method = 'Euler', **kwargs):
    # This function integrates the nonlinear dynamics forward to formulate the corresponding trajectory
    # dyn_func[in]     - dynamics function (pre-populated with *params) for integration
    # control_seq[in]  - sequence of control inputs (scalar or vector depending on control dimension -- must match dyn_func control dim)
    # state_init[in]   - starting state for integration (vector, must match dyn_func state dim)
    # time_step[in]    - time interval between sequence points (scalar > 0)
    # t_final[in]      - final time projected for integration ((len(control_seq)+1) * time_step must equal final time)
    # state_seq[out]   - jax array shape[iter+1,state_dim] of sequences of state space
    int_iter       = len(control_seq)
    state_seq      = state_init
    time_seq       = jnp.arange(int_iter + 1) * time_step
    time_span      = (0, time_step)
    if sim_method == 'Euler':
        for idx in range(int_iter):
            state_dot  = dyn_func(time_seq[idx], state_seq[idx], control_seq[idx])
            state_next = jnp.array([state_seq[idx] + state_dot * time_step])
            state_seq  = jnp.append(state_seq, state_next, axis=0)
    elif sim_method == 'solve_ivp_zoh':
        for idx in range(int_iter):
            dyn_func_zoh = (lambda time_dyn, state: dyn_func(time_dyn, state, control_seq[idx]))
            result_ivp   = solve_ivp(dyn_func_zoh, time_span, state_seq[idx])
            state_next   = jnp.array([result_ivp.y[:,-1]])
            state_seq    = jnp.append(state_seq, state_next, axis=0)
    else:
        print('invalid simulation method')
    return state_seq, time_seq

def calculate_cost_seq_and_total_cost(cost_func, state_seq, control_seq):
    # check state_seq and control_seq are valid lengths
    # calculate each discrete cost
    # Calculate total cost

    total_cost = 0
    cost_seq = 0
    return total_cost, cost_seq

def taylor_expand_cost(cost_func, x_seq, u_seq):
# This function creates a quadratic approximation of the cost function Using taylor expansion
# Expansion is approximated about the rolled out trajectory
    l_x = 0
    l_u = 0
    l_xx = 0
    l_uu = 0
    l_ux = 0
    return l_x, l_u, l_xx, l_uu, l_ux

def taylor_expand_pseudo_hamiltonian(cost_func, dyn_func, x_seq, u_seq):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step
    q_x = 0
    q_u = 0
    q_xx = 0
    q_ux = 0
    q_uu = 0
    return q_x, q_u, q_xx, q_ux, q_uu

def linearize_dynamics(dyn_func, x_primal, u_primal, t_primal='None'):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u
    a_lin = jax.jacfwd(lambda x: dyn_func(t_primal, x       , u_primal))(x_primal)
    b_lin = jax.jacfwd(lambda u: dyn_func(t_primal, x_primal, u       ))(u_primal)
    return a_lin, b_lin

def discretize_state_space(input_state_space:stateSpace, time_step:float, c2d_method='Euler', **kwargs):
#   A - (nxn) - continuous state transition matrix
#   B - (nxm) - continuous control matrix
#   C - (pxn) - continuous state measurement matrix
#   D - (pxm) - continuous direct feedthrough matrix
#   Continuous state space:
#   xdot(t) = A * x(t) + B * u(t)
#   y(t)    = C * x(t) + D * u(t)
#   transformation for zohCombined:
#   e^([[A, B],[0,0]] * time_step) = [[Ad, Bd],[0,I]]
#   Discrete state space:
#   x[t+timestep] = Ad * x[t] + Bd * u[t]
#   y[t]          = Cd * x[t] + Dd * u[t]

    func_error = False

    if input_state_space.type == 'discrete':
        raise TypeError('input state space is already discrete')

    else:
        a_d = jnp.zeros(input_state_space.a.shape)
        b_d = jnp.zeros(input_state_space.b.shape)
        c_d = jnp.zeros(input_state_space.c.shape)
        d_d = jnp.zeros(input_state_space.d.shape)
        n  = input_state_space.a.shape[0]
        m  = input_state_space.b.shape[1]
        p  = input_state_space.c.shape[0]

        if c2d_method=='Euler' :
            a_d = jnp.eye(n) + (input_state_space.a * time_step)
            b_d = input_state_space.b * time_step
            c_d = input_state_space.c
            d_d = input_state_space.d

        elif c2d_method=='zoh' :
            if jscipy.linalg.det(input_state_space.a) > 10E-8:
                a_d = jscipy.linalg.expm(input_state_space.a * time_step)
                b_d = jnp.linalg.inv(input_state_space.a) @ (a_d - jnp.eye(n)) @ input_state_space.b
                c_d = input_state_space.c
                d_d = input_state_space.d
            else:
                raise ValueError('determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid')

        elif c2d_method=='zohCombined':
        #   create combined A B matrix e^([[A, B],[0,0]]
            a_b_c  = jnp.concatenate((jnp.concatenate((input_state_space.a, input_state_space.b),axis=1),jnp.zeros((m,n + m))), axis=0)
            a_b_d  = jscipy.linalg.expm(a_b_c * time_step)
            a_d   = a_b_d[:n,:n]
            b_d   = a_b_d[:n,n:]
            c_d   = input_state_space.c
            d_d   = input_state_space.d
        else:
            raise ValueError('invalid discretization method')
    d_state_space = stateSpace(a_d, b_d, c_d, d_d, time_step)
    return d_state_space

def calculate_linearized_state_space_seq(dyn_func_with_params, state_seq, control_seq, time_step, **kwargs):
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    if len(state_seq.shape) != 2 or len(control_seq.shape) != 2:
        raise ValueError('state or control sequence is incorrect array dimension. sequences must be 2d arrays')
    elif len(state_seq) != len(control_seq)+1:
        raise ValueError('state and control sequences are incompatible lengths. state seq must be control seq length +1')
    elif not isinstance(time_step, float) or not time_step > 0:
        raise ValueError('time_step is invalid. Must be positive float')
    else:
        state_dim   = len(state_seq[1,:])
        control_dim = len(control_seq[1,:])
        a_lin_array = jnp.zeros((state_dim, state_dim,1))
        b_lin_array = jnp.zeros((state_dim, control_dim,1))
        c_mat_dummy = jnp.zeros((1,state_dim))
        d_mat_dummy = jnp.zeros((1,control_dim))
        for idx,_ in enumerate(control_seq):
            a_lin, b_lin             = linearize_dynamics(dyn_func_with_params, state_seq[idx], control_seq[idx])
            ss_pend_lin_continuous   = stateSpace(a_lin, b_lin, c_mat_dummy, d_mat_dummy)
            ss_pend_lin_discrete     = discretize_state_space(ss_pend_lin_continuous, time_step, **kwargs)
            if idx == 0:
                a_lin_array = jnp.array([ss_pend_lin_discrete.a])
                b_lin_array = jnp.array([ss_pend_lin_discrete.b])
            else:
                a_lin_array = jnp.append(a_lin_array, jnp.array([ss_pend_lin_discrete.a]), axis=0)
                b_lin_array = jnp.append(b_lin_array, jnp.array([ss_pend_lin_discrete.b]), axis=0)
    return a_lin_array, b_lin_array