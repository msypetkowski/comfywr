dest_data_root=/workdir/ComfyUI/models/
docker run --gpus all --rm --ipc=host --net=host \
    -v `pwd`:/workdir/ \
    -v $(pwd)/custom_nodes/ComfyUI_UltimateSDUpscale/:/workdir/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale/ \
    -v $(pwd)/custom_nodes/ComfyUI-DepthAnythingV2/:/workdir/ComfyUI/custom_nodes/ComfyUI-DepthAnythingV2/ \
    -v $(pwd)/downloaded_models/:/workdir/ComfyUI/models/ \
    -v $(pwd)/comfywr_cache/:/root/ \
    -it comfywr:latest \
    bash -c "cd /workdir/; jupyter-lab --ip 0.0.0.0 --port 12345 --allow-root"
    # -v $(pwd)/custom_nodes/:/workdir/ComfyUI/custom_nodes/ \
