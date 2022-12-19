#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import time

class ilqr_controller(object):

    def __init__(self, state_dim, control_dim, state_transition_func, cost_func, state_init, control_init, max_iter, time_step, final_time):
        self.state_dim     = state_dim
        self.control_dim   = control_dim
        self.fx            = state_transition_func
        self.cost_func     = cost_func
        self.state_init    = state_init
        self.control_init  = control_init
        self.max_iter      = max_iter
        self.time_step     = time_step
        self.final_time    = final_time

        #health checks

    def _is_valid_control_init(self, control_init, time_step, final_time):
        # control vector - m x n matrix 
        # m - number of control dofs, 
        # n - number of time steps
        # control vector must be of same m dim as control_dim
        # control vector must contain N-1 timesteps (final timestep does not have control step)
        valid_bool = True
        return valid_bool
        
class state_space:
    def __init__(self, A, B, C, D, time_step = float(0)):
        if(A.shape[0] != A.shape[1]):
            print('A is not a square matrix')
            exit()
        elif(A.shape[0] != B.shape[0]):
            print('A and B have different state dimensions')
            exit()
        elif(A.shape[0] != C.shape[1]):
            print('A and C have different state dimensions')
            exit()
        elif(B.shape[1] != D.shape[1]):
            print('B and D have dirrenent control dimensions')
            exit()
        elif(C.shape[0] != D.shape[0]):
            print('C and D have different measurement dimensions')
            exit()
        else:
            if(isinstance(time_step,float)):
                if (time_step==0.0):
                    self.A = A
                    self.B = B
                    self.C = C
                    self.D = D
                    self.time_step = time_step
                    self.type = 'continuous'
                elif(time_step < 0):
                    print('negative timestep is invalid')    
                else:                    
                    self.A = A
                    self.B = B
                    self.C = C
                    self.D = D
                    self.time_step = time_step
                    self.type = 'discrete'
            else:
                print('Invalid time step input type')
                

def initialize_system(t_final, time_step):
    numt = 0
    return numt


def calculate_forward_pass(dyn_func, control_seq, state_init, time_step):
    x_seq = 0
    u_seq = 0
    cost_seq = 0
    cost_total = 0
    return x_seq, u_seq, cost_seq, cost_total


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


def calculate_backwards_pass():

#   du_optimal = argmin over delu of Q(delx,delu) = k + (K * delx)
#   k = - inv(q_uu) * q_u
#   K = - inv(q_uu) * q_ux
#   V_N = l(x_N)
#   Vxx = q_xx -       transpose(K) * q_uu * K
#   Vx  = q_x  -       transpose(K) * q_uu * k
#   DV  =      - 1/2 * transpose(k) * q_uu * k
    unew = 0
    V = 0
    print("this is the backward pass 2")
    return unew, V

def linearize_dynamics(dyn_func, x_primal, u_primal):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x_primal  - [in] primal state linearization point
# u         - [in] primal control linearization point
# A_lin     - [out] continuous time linearization of dyn_func wrt state eval at x,u
# B_lin     - [out] continuous time linearization of dyn_func wrt control eval at x,u

    A_lin = jax.jacfwd(lambda x: dyn_func(x, u_primal))(x_primal)  
    B_lin = jax.jacfwd(lambda u: dyn_func(x_primal, u))(u_primal) 

    return A_lin, B_lin

def discretize_state_space(input_state_space, time_step, method='Euler'):
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

    if(input_state_space.type == 'discrete'):
        print('state space is already discrete')
        Ad = input_state_space.A
        Bd = input_state_space.B
        Cd = input_state_space.C
        Dd = input_state_space.D
        time_step = input_state_space.time_step
        func_error = True

    else:   
        Ad = jnp.zeros(input_state_space.A.shape)
        Bd = jnp.zeros(input_state_space.B.shape)
        Cd = jnp.zeros(input_state_space.C.shape)
        Dd = jnp.zeros(input_state_space.D.shape)
        n  = input_state_space.A.shape[0]
        m  = input_state_space.B.shape[1]
        p  = input_state_space.C.shape[0]

        if(method=='Euler'):
            Ad = jnp.eye(n) + (input_state_space.A * time_step)
            Bd = input_state_space.B * time_step
            Cd = input_state_space.C
            Dd = input_state_space.D
            
        elif(method=='zoh'):
            if(jscipy.linalg.det(input_state_space.A) > 10E-8):
                Ad = jscipy.linalg.expm(input_state_space.A * time_step)
                Bd = jnp.linalg.inv(input_state_space.A) @ (Ad - jnp.eye(n)) @ input_state_space.B
                Cd = input_state_space.C
                Dd = input_state_space.D
            else:
                    print('determinant of A is excessively small (<10E-8), simple zoh method is potentially invalid')
                    func_error = True

        elif(method=='zohCombined'):
        #   create combined A B matrix e^([[A, B],[0,0]]
            ABc  = jnp.concatenate((jnp.concatenate((input_state_space.A, input_state_space.B),axis=1),jnp.zeros((m,n + m))), axis=0) 
            ABd  = jscipy.linalg.expm(ABc * time_step)
            Ad   = ABd[:n,:n]
            Bd   = ABd[:n,n:]
            Cd   = input_state_space.C
            Dd   = input_state_space.D

        else:
            print('invalid discretization method')
            func_error = True

    d_state_space = state_space(Ad, Bd, Cd, Dd, time_step)        

    return d_state_space, func_error

def calculate_linearized_state_vector(state_seq, control_seq, dyn_func, time_step):
    # walk through state and control sequences
    # for each element, linearize the dyn_func dynamics
    # calculate the discretized dynamics for each linearized element
    # return a 3d matrix of both state and control transition matrices for each time step
    A_lin_array = 0
    B_lin_array = 0
    
    return A_lin_array, B_lin_array