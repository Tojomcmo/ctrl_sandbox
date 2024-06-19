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




**DOCKER INSTALL**
- (Repo configured for vs code)
- install docker
- clone repo
- install extensions Docker, Dev Containers
- shift_cmd_P: Dev containers: open folder with container...
- select project folder (ctrl_sandbox)
- in container:
    - python3 -m venv .venv
    - source .venv/bin/activate
    - poetry install
    - begin coding!











**UBUNTU INSTALL**

This is a set of bash commands that seemed to properly install ipopt and the project on a fresh linux distro.
Need to retest to verify

########## installing pipx and poetry on ubuntu ###########

sudo apt upgrade \
sudo apt install pipx \
pipx ensurepath \
pipx install poetry


########### installing ipopt on ubuntu ############

sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev \

cd \
mkdir ipopt_install_loc \
cd $HOME/ipopt_install_loc

git clone https://github.com/coin-or-tools/ThirdParty-ASL.git \
cd ThirdParty-ASL \
./get.ASL \
./configure \
make \
sudo make install

cd $HOME/ipopt_install_loc

git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git \
cd ThirdParty-Mumps \
./get.Mumps \
./configure \
make \
sudo make install

cd $HOME/ipopt_install_loc

git clone https://github.com/coin-or/Ipopt.git \
cd Ipopt \
mkdir build \
$HOME/ipopt_install/Ipopt/configure --with-mumps-cflags="-I$HOME/ipopt_install_loc/ThirdParty_Mumps/MUMPS/include"

make \
make test \
sudo make install


######### install and activate ctrl_sandbox ###########
cd desired/project/location/ \
git clone https://github.com/Tojomcmo/ctrl_sandbox.git \
cd ctrl_sandbox/ \
poetry install \
poetry shell