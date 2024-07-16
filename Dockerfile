# ARG CUDA_VERSION=12.1.0-devel
ARG CUDA_VERSION=12.1.0-runtime
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
        # TODO not sure all of this is required, remove unnnecessary
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
        && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# lines copied from Unique3D dockerfile
RUN apt-get update && apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git-lfs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl mesa-utils-extra

RUN ln -s /usr/bin/python3.11 /usr/bin/python & \
    ln -s /usr/bin/python3.11 /usr/bin/python3 & \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN python -m pip install --upgrade pip


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl


# install and initialize conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"
RUN conda init bash
RUN conda create -n comfywr python=3.10 && echo "source activate comfywr" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/comfywr/bin:$PATH
RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y

# other packs (simple dependencies)
# COPY custom_nodes/comfyui_controlnet_aux/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/comfyui_marigold/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/ComfyUI_essentials/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/ComfyUI-Impact-Pack/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/ComfyUI-Inspire-Pack/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/comfy-image-saver/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/ComfyUI_Transparent-Background/requirements.txt .
# RUN pip install -r requirements.txt
# COPY custom_nodes/ComfyUI-KJNodes/requirements.txt .
# RUN pip install -r requirements.txt

# other packages (not listed in requirements.txt files in some submodules, but needed)
# RUN pip install ultralytics==8.2.1
# RUN pip install numba numexpr

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


# RUN pip install git+https://github.com/NVlabs/nvdiffrast
# RUN pip install git+https://github.com/ashawkey/kiuikit.git
# RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# RUN pip install --upgrade setuptools
# ENV FORCE_CUDA=1
# ENV TORCH_CUDA_ARCH_LIST="8.9"
# ENV CUDA_PATH=/usr/local/cuda-12.3
# RUN pip install ./diff-gaussian-rasterization
# RUN pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# RUN pip install git+https://github.com/rusty1s/pytorch_scatter.git

RUN apt-get install -y libglew-dev libgl-dev freeglut3 freeglut3-dev mesa-utils libgl1-mesa-dri

ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux/:/workdir/blender_workdir/
WORKDIR /workdir/ComfyUI/
