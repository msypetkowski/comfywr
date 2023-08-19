data_root=/home/qwe/src/stable-diffusion-webui-docker/data/
dest_data_root=/workdir/ComfyUI/models/

docker run --gpus all --rm --ipc=host --net=host \
    -v `pwd`:/workdir/ \
    -v $data_root/models/Stable-diffusion/:$dest_data_root/checkpoints/ \
    -v $data_root/models/Lora/:/$dest_data_root/loras/ \
    -v $data_root/VAE/:/$dest_data_root/vae/ \
    -v $data_root/embeddings/:/$dest_data_root/embeddings/ \
    -v $data_root/RealESRGAN/:/$dest_data_root/upscale_models/ \
    -v $data_root/ControlNet/:/$dest_data_root/controlnet/ \
    -it comfywr:latest \
    bash -c "cd /workdir/; jupyter-lab --ip 0.0.0.0 --port 12345 --allow-root"
