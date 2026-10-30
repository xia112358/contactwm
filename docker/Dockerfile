# All components are installed into the single managed main environment.
# PyTorch's CUDA 13 wheels require a compatible host NVIDIA driver.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Switch to root user for installation
USER root

# Environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/workspace
ENV CONDA_ROOT=/opt/miniforge3
ENV MINIFORGE_ROOT=/opt/miniforge3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    linux-libc-dev \
    python3 \
    ca-certificates \
    swig \
    curl \
    wget \
    unzip \
    git \
    sudo

# Install Miniforge as the package manager; main is the only project prefix.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /miniforge.sh && \
    bash /miniforge.sh -b -p $CONDA_ROOT

# Configure conda for non-interactive use
RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

# Create workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Copy repository contents to holosoma
COPY . ./holosoma

##### Run setup
RUN python3 /workspace/holosoma/scripts/bootstrap_runtime.py && \
    bash /workspace/holosoma/scripts/setup_all.sh

# Set up conda activation for runtime
RUN echo "source /workspace/holosoma/scripts/source_isaaclab3_newton_setup.sh" >> ~/.bashrc

# Set working directory
WORKDIR /workspace/holosoma

ENTRYPOINT []
CMD ["/bin/bash"]
