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
    

How to install:

 - install poetry if not installed: brew install poetry
 - add poetry to path if not found (mac .zhrc): export PATH="$HOME/.local/bin:$PATH"
 - clone project: git clone https://github.com/Tojomcmo/ctrl_sandbox.git
 - navigate to project
 - create virtual environment in project: python -m venv .venv
 - install project with poetry: poetry install
 - run scripts!


resources:
    - https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf    
    - https://underactuated.csail.mit.edu/index.html
