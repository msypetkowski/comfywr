docker run --gpus all --rm --net=host \
    -v `pwd`/:/comfywr/ \
    -it diffusers-pytorch-cuda-wr:latest bash -c "cd /comfywr/ ; jupyter-lab --allow-root"
