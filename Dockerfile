FROM jupyter/base-notebook:2023-03-13

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ComfyUI/requirements.txt .
RUN pip install -r requirements.txt

COPY custom_nodes/comfy_controlnet_preprocessors/requirements.txt .
RUN pip install -r requirements.txt

COPY requirements.txt .
RUN pip install -U -r requirements.txt

# uncomment for jupyterlab-vim plugin
RUN apt-get update --fix-missing
RUN apt install -y nodejs npm
RUN pip install jupyterlab jupyterlab-vim

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/workdir/:/workdir/ComfyUI/:/workdir/ComfyUI/custom_nodes/comfy_controlnet_preprocessors/:/workdir/blender_workdir/
WORKDIR /workdir/ComfyUI/
