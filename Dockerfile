# FROM jupyter/scipy-notebook:lab-4.0.7
FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  python3-pip -y

RUN apt-get update --fix-missing
RUN apt install -y nodejs npm

# RUN python3 -m ensurepip

RUN pip3 install jupyterlab-vim==4.1.0

COPY ComfyUI/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/comfyui_controlnet_aux/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/comfyui_marigold/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/ComfyUI-3D-Pack/requirements.txt .
RUN pip install -r requirements.txt

COPY requirements.txt .
RUN pip install -U -r requirements.txt

# RUN pip install pytorch-cuda


ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization

RUN pip install --upgrade setuptools

# RUN apt install nvcc -y
ENV CUDA_PATH=/usr/local/cuda-12.3
RUN pip install ./diff-gaussian-rasterization

RUN pip install git+https://github.com/NVlabs/nvdiffrast

RUN pip install numba numexpr

RUN pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git

# RUN pip install pytorch3d
# pip install --no-index --no-cache-dir pytorch3d
# RUN pip install pytorch3d

RUN pip install rembg

ENV FORCE_CUDA=1
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux/:/workdir/blender_workdir/
WORKDIR /workdir/ComfyUI/
