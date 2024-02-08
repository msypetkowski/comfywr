FROM jupyter/scipy-notebook:lab-4.0.7

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update --fix-missing
RUN apt install -y nodejs npm
RUN pip install jupyterlab-vim==4.1.0

COPY ComfyUI/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/comfyui_controlnet_aux/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/ComfyUI-Marigold/requirements.txt .
RUN pip install -r requirements.txt

COPY requirements.txt .
RUN pip install -U -r requirements.txt

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfyui_controlnet_aux/:/workdir/blender_workdir/
WORKDIR /workdir/ComfyUI/
