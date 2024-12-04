ARG CUDA_VERSION=12.1.0-devel

FROM --platform=amd64 docker.io/nvidia/cuda:${CUDA_VERSION}-ubuntu22.04

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
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{ \
    "file_format_version" : "1.0.0", \
    "ICD" : { \
        "library_path" : "libEGL_nvidia.so.0" \
    } \
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Main repo dependencies
COPY ComfyUI/requirements.txt .
RUN pip install -r requirements.txt

# other packs (simple dependencies)
COPY custom_nodes/comfyui_controlnet_aux/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/comfyui_marigold/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI_essentials/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI-Inspire-Pack/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/comfy-image-saver/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI_Transparent-Background/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI-KJNodes/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI_FizzNodes/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/ComfyUI-Impact-Pack/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/x_flux_comfyui/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/eden_comfy_pipelines/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/efficiency-nodes-comfyui/requirements.txt .
RUN pip install -r requirements.txt
COPY custom_nodes/rgthree-comfy/requirements.txt
RUN pip install -r requirements.txt

# other packages (not listed in requirements.txt files in some submodules, but needed)
RUN pip install ultralytics==8.2.1
RUN pip install numba numexpr

# other comfywr-specific requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# for PIL text rendering
RUN wget https://github.com/JotJunior/PHP-Boleto-ZF2/raw/master/public/assets/fonts/arial.ttf
RUN mv arial.ttf /usr/share/fonts/

# 3D pack (complex dependencies)
COPY custom_nodes/ComfyUI-3D-Pack/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install ninja rembg[gpu] open_clip_torch
WORKDIR /install_dir/
COPY custom_nodes/ComfyUI-3D-Pack/install.py .
COPY custom_nodes/ComfyUI-3D-Pack/_Pre_Builds/ ./_Pre_Builds
COPY custom_nodes/ComfyUI-3D-Pack/shared_utils/ ./shared_utils
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9+PTX"
# RUN python install.py

WORKDIR /workdir/ComfyUI/

ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux/:/workdir/blender_workdir/
