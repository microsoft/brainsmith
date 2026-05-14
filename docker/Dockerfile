# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM ubuntu:22.04
LABEL maintainer="Thomas Keller <thomaskeller@microsoft.com>, Mahdi Ghandi <Mahdi.Ghandi@microsoft.com>"

# ARG ENTRYPOINT - removed: no longer used for backward compatibility

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libc6-dev-i386 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    nano \
    zsh \
    rsync \
    git \
    openssh-client \
    sshpass \
    wget \
    sudo \
    unzip \
    zip \
    locales \
    lsb-core \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-setuptools-scm \
    python3-venv \
    pybind11-dev \
    libfmt-dev \
    libboost-dev \
    libjansson-dev \
    libgetdata-dev \
    libtinfo5 \
    libaio-dev \
    g++-10
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN locale-gen "en_US.UTF-8"

# Install CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-4 && \
    rm cuda-keyring_1.1-1_all.deb

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install pip dependencies for Brainsmith
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Install pip dependencies for FINN
COPY docker/requirements.finn.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Xilinx toolflow specific imports
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy all entrypoint scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/entrypoint_fast.sh /usr/local/bin/entrypoint_fast.sh
COPY docker/setup_env.sh /usr/local/bin/setup_env.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/entrypoint_fast.sh /usr/local/bin/setup_env.sh

# Set default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
