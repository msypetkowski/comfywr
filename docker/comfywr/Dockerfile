FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        freeglut3-dev \
        git \
        libegl1 \
        libegl1-mesa-dev \
        libgl1 \
        libgl1-mesa-dev \
        libgl1-mesa-dri \
        libgles2 \
        libgles2-mesa-dev \
        libglib2.0-0 \
        libglew-dev \
        libglvnd-dev \
        libglvnd0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mesa-utils \
        ninja-build \
        wget && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH=/workdir:/workdir/ComfyUI:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux:/workdir/blender_workdir

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libEGL_nvidia.so.0"' \
      '  }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN python -m pip install --no-cache-dir \
      torch==2.9.1 \
      torchvision==0.24.1 \
      torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu128

# Main repo requirements
COPY ComfyUI/requirements.txt /tmp/requirements/00-comfyui.txt

# Custom node requirements
COPY custom_nodes/comfyui_controlnet_aux/requirements.txt /tmp/requirements/10-comfyui_controlnet_aux.txt
COPY custom_nodes/comfyui_marigold/requirements.txt /tmp/requirements/11-comfyui_marigold.txt
COPY custom_nodes/ComfyUI_essentials/requirements.txt /tmp/requirements/12-comfyui_essentials.txt
COPY custom_nodes/ComfyUI-Inspire-Pack/requirements.txt /tmp/requirements/13-inspire_pack.txt
COPY custom_nodes/comfy-image-saver/requirements.txt /tmp/requirements/14-comfy_image_saver.txt
COPY custom_nodes/ComfyUI_Transparent-Background/requirements.txt /tmp/requirements/15-transparent_background.txt
COPY custom_nodes/ComfyUI-KJNodes/requirements.txt /tmp/requirements/16-kjnodes.txt
COPY custom_nodes/ComfyUI_FizzNodes/requirements.txt /tmp/requirements/17-fizznodes.txt
COPY custom_nodes/ComfyUI-Impact-Pack/requirements.txt /tmp/requirements/18-impact_pack.txt
COPY custom_nodes/x_flux_comfyui/requirements.txt /tmp/requirements/19-x_flux.txt
COPY custom_nodes/eden_comfy_pipelines/requirements.txt /tmp/requirements/20-eden_comfy_pipelines.txt
COPY custom_nodes/efficiency-nodes-comfyui/requirements.txt /tmp/requirements/21-efficiency_nodes.txt
COPY custom_nodes/rgthree-comfy/requirements.txt /tmp/requirements/22-rgthree.txt

# comfywr-specific requirements
COPY requirements.txt /tmp/requirements/99-comfywr.txt

RUN set -eux; \
    for f in /tmp/requirements/*.txt; do \
        python -m pip install -r "$f"; \
    done; \
    python -m pip install \
        ultralytics==8.2.1 \
        numba \
        numexpr \
        pyOpenSSL \
        watchdog; \
    \
    # Re-assert the torch stack last so no later dependency install can downgrade/replace it.
    python -m pip install --no-cache-dir --force-reinstall \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128; \
    \
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
PY

# For PIL text rendering
RUN wget -O /usr/share/fonts/arial.ttf \
    https://github.com/JotJunior/PHP-Boleto-ZF2/raw/master/public/assets/fonts/arial.ttf

WORKDIR /workdir/ComfyUI/
