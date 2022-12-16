#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp
import jax.scipy as jscipy

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
        

def initialize_system(t_final, dt):
    numt = 0
    return numt


def calculate_forward_pass(dyn_func, control_seq, state_init, time_step):
    x_seq = 0
    u_seq = 0
    cost_seq = 0
    cost_total = 0
    return x_seq, u_seq, cost_seq, cost_total


def linearize_dynamics(dyn_func, x, u, time_step):
# Linearizes the dynamics function about the primals x and u
# dyn_func  - [in] continuous function of state transition
# x         - [in] primal state
# u         - [in] primal control
# time_step - [in] discrete time step
# A         - [out] discrete time linearization of dyn_func wrt state eval at x,u
# B         - [out] discrete time linearization of dyn_func wrt control eval at x,u
    A = 0
    B = 0
    return A, B


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


def discretize_state_space(Ac, Bc, Cc, Dc, time_step, method='Euler'):
#   Ac - (nxn) - continuous state transition matrix
#   Bc - (nxm) - continuous control matrix
#   Cc - (pxn) - continuous state measurement matrix
#   Dc - (pxm) - continuous direct feedthrough matrix
#   Continuous state space:
#   xdot(t) = Ac * x(t) + Bc * u(t)
#   y(t)    = Cc * x(t) + Dc * u(t)
#   transformation for zohCombined:
#   e^([[Ac, Bc],[0,0]] * time_step) = [[Ad, Bd],[0,I]]
#   Discrete state space:   
#   x[t+timestep] = Ad * x[t] + Bd * u[t]
#   y[t]          = Cd * x[t] + Dd * u[t]

    #  error catching
    if(Ac.shape[0] != Ac.shape[1]):
        print('Ac is not a square matrix')
        exit()
    elif(Ac.shape[0] != Bc.shape[0]):
        print('Ac and Bc have different state dimensions')
        exit()
    elif(Ac.shape[0] != Cc.shape[1]):
        print('Ac and Cc have different state dimensions')
        exit()
    elif(Bc.shape[1] != Dc.shape[1]):
        print('Bc and Dc have dirrenent control dimensions')
        exit()
    elif(Cc.shape[0] != Dc.shape[0]):
        print('Cc and Dc have different measurement dimesnions')
        exit()
    else:
        Ad = jnp.zeros(Ac.shape)
        Bd = jnp.zeros(Bc.shape)
        Cd = jnp.zeros(Cc.shape)
        Dd = jnp.zeros(Dc.shape)
        n  = Ac.shape[0]
        m  = Bc.shape[1]
        p  = Cc.shape[0]

        if(method=='Euler'):
            Ad = jnp.eye(n) + (Ac * time_step)
            Bd = Bc * time_step
            Cd = Cc
            Dd = Dc
        elif(method=='zoh'):
            if(jscipy.linalg.det(Ac) != 0):
                Ad = jscipy.linalg.expm(Ac * time_step)
                Bd = jnp.linalg.inv(Ac) @ (Ad - jnp.eye(n)) @ Bc
                Cd = Cc
                Dd = Dc
            else:
                print('Ac is non-invertible, simple zoh method is invalid')
        elif(method=='zohCombined'):
        #   create combined A B matrix e^([[Ac, Bc],[0,0]]
            ABc  = jnp.concatenate((jnp.concatenate((Ac, Bc),axis=1),jnp.zeros((m,n + m))), axis=0) 
            ABd  = jscipy.linalg.expm(ABc * time_step)
            Ad   = ABd[:n,:n]
            Bd   = ABd[:n,n:]
            Cd   = Cc
            Dd   = Dc

        else:
            print('invalid discretization method')

    return Ad, Bd, Cd, Dd