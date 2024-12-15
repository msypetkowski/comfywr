DEST_MODELS_ROOT=/workdir/ComfyUI/models/
docker run --gpus all --rm --ipc=host --net=host \
    -v `pwd`:/workdir/ \
    -v $(pwd)/custom_nodes/ComfyUI_UltimateSDUpscale/:/workdir/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale/ \
    -v $(pwd)/custom_nodes/ComfyUI-DepthAnythingV2/:/workdir/ComfyUI/custom_nodes/ComfyUI-DepthAnythingV2/ \
    -v $(pwd)/downloaded_models/:$DEST_MODELS_ROOT \
    -v $(pwd)/comfywr_cache/:/root/ \
    -it comfywr:latest \
    bash -c "cd /workdir/; jupyter-lab --ip 0.0.0.0 --port 12345 --allow-root"
