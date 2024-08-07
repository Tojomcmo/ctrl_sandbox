# Use an official Ubuntu as a parent image
FROM ubuntu:latest

# Set environment variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV PATH=$PATH:/root/.local/bin 
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    pipx \
    python3 \
    gcc \
    g++ \
    gfortran \
    git \
    patch \
    wget \
    pkg-config \
    liblapack-dev \
    libmetis-dev \
    make \
    python3-dev


# Install poetry using pipx
RUN pipx ensurepath
RUN pipx install poetry
RUN pipx ensurepath

# Clone and install ThirdParty-ASL
RUN git clone https://github.com/coin-or-tools/ThirdParty-ASL.git && \
    cd ThirdParty-ASL && \
    ./get.ASL && \
    ./configure && \
    make && \
    make install && \
    cd ..

# Clone and install ThirdParty-Mumps
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && \
    cd ThirdParty-Mumps && \
    ./get.Mumps && \
    ./configure && \
    make && \
    make install && \
    cd ..

# Clone and install Ipopt
RUN git clone https://github.com/coin-or/Ipopt.git && \
    cd Ipopt && \
    mkdir build && \
    cd build && \
    ../configure && \
    make && \
    make test && \
    make install

RUN apt install -y qtcreator qtbase5-dev qt5-qmake cmake && \
    apt install python3-pyqt5 -y
# Clean up
RUN rm -rf ThirdParty-ASL ThirdParty-Mumps Ipopt

# Copy the entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the script as the entrypoint
# ENTRYPOINT ["entrypoint.sh"]

# Set the default command to bash
CMD ["bash"]
