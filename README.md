# ctrl_sandbox
This repository contains implementations of control algorithms, and serves as a sandbox for testing, developing, and comparing different algorithms.

Much of the project is jax-compatible, allowing for auto-differentiation and lax.scan functionality for fast calculation.


includes:
    - Iterative Linear Quadratic Regulation (iLQR)
    - Jax dynamic simulation pipeline with lax.scan
    - Jax compatible pendulum and double pendulum dynamic functions
    - pendulum and double pendulum animation 

In development
    - Mujoco-compatible iLQR
    - Direct collocation
    - LTI sysID tool - Welch's method
    - Time-varying trajectory-tracking LQR
    

resources:
    - https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf    
    - https://underactuated.csail.mit.edu/index.html
