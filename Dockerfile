ARG CUDA_VERSION=12.1.0-devel
# ARG CUDA_VERSION=12.3.1-devel

FROM --platform=amd64 docker.io/nvidia/cuda:${CUDA_VERSION}-ubuntu22.04

# FROM ghcr.io/selkies-project/nvidia-egl-desktop:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libegl1-mesa-dev \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libgles2 \
        libgles2-mesa-dev \
        libglib2.0-0 \
        libglvnd-dev \
        libglvnd0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        wget \
        mesa-utils \
        libglew-dev \
        freeglut3-dev \
        libgl1-mesa-dri \
        && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python & \
    ln -s /usr/bin/python3.11 /usr/bin/python3 & \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN python -m pip install --upgrade pip

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH
ENV LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
ENV LIBGL_DEBUG=verbose

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

# install and initialize conda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     chmod +x Miniconda3-latest-Linux-x86_64.sh && \
#     ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
#     rm Miniconda3-latest-Linux-x86_64.sh
# ENV PATH="/workspace/miniconda3/bin:${PATH}"
# RUN conda init bash
# RUN conda create -n comfywr python=3.10 && echo "source activate comfywr" > ~/.bashrc
# ENV PATH /workspace/miniconda3/envs/comfywr/bin:$PATH
# RUN conda install Ninja
# RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y

# Main repo dependencies
COPY ComfyUI/requirements.txt .
RUN pip install -r requirements.txt

# 3D pack (complex dependencies)
COPY custom_nodes/ComfyUI-3D-Pack/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install ninja rembg[gpu] open_clip_torch
WORKDIR /install_dir/
COPY custom_nodes/ComfyUI-3D-Pack/ .
RUN python install.py

# RUN conda install -c conda-forge libstdcxx-ng libllvm15
# RUN conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive -y
# RUN conda install -c conda-forge libstdcxx-ng libllvm15 -y


ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux/:/workdir/blender_workdir/
WORKDIR /workdir/ComfyUI/

