dest_data_root=/workdir/ComfyUI/models/

docker run --gpus all --rm --ipc=host --net=host \
    -v `pwd`:/workdir/ \
    -v $MY_DATA_ROOT/models/Stable-diffusion/:$dest_data_root/checkpoints/ \
    -v $MY_DATA_ROOT/models/Lora/:/$dest_data_root/loras/ \
    -v $MY_DATA_ROOT/VAE/:/$dest_data_root/vae/ \
    -v $MY_DATA_ROOT/embeddings/:/$dest_data_root/embeddings/ \
    -v $MY_DATA_ROOT/RealESRGAN/:/$dest_data_root/upscale_models/ \
    -v $MY_DATA_ROOT/ControlNet/:/$dest_data_root/controlnet/ \
    -it comfywr:latest \
    bash -c "cd /workdir/; jupyter-lab --ip 0.0.0.0 --port 12345 --allow-root"
