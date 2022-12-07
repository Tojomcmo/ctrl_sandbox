#https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

# This script provides functions for the ilqr algorithm outlined in the above link.
# iLQR is an algorithm for trajectory optimization.

import jax
import jax.numpy as jnp

class ilqr_controller():


def initialize_system(t_final, dt):

    return numt

#
def calculate_forward_pass(dyn_func, control_seq, state_init, dt):

    return x_seq, u_seq, cost_seq, cost_total

# Linearizes the dynamics function about the primals x and u
def linearize_dynamics(dyn_func, x, u):

    return A, B

#
def taylor_expand_cost(cost_func, x_seq, u_seq):

    return l_x, l_u, l_xx, l_uu, l_ux

#

def taylor_expand_pseudo_hamiltonian(cost_func, dyn_func, x_seq, u_seq):

#   Q(dx,du) = l(x+dx, u+du) + V'(f(x+dx,u+du))
#   where V' is value function at the next time step

    return q_x, q_u, q_xx, q_ux, q_uu


def calculate_backwards_pass():

#   du_optimal = argmin over delu of Q(delx,delu) = k + (K * delx)
#   k = - inv(q_uu) * q_u
#   K = - inv(q_uu) * q_ux

#   V_N = l(x_N)
#   Vxx = q_xx -       transpose(K) * q_uu * K
#   Vx  = q_x  -       transpose(K) * q_uu * k
#   DV  =      - 1/2 * transpose(k) * q_uu * k

    return unew, V

