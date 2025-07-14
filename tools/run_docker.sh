docker run --gpus=all --rm --net=host \
  --ipc=host \
  --platform linux/amd64 \
  -p 8188:8188 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd):/workdir/ \
  -v $(pwd)/downloaded_models/:/workdir/ComfyUI/models/ \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/comfywr_cache/:/root/ \
  -v $(pwd)/custom_nodes/:/workdir/ComfyUI/custom_nodes/ \
  -it comfywr:latest \
  bash -c "cd /workdir/ComfyUI/; python main.py"
  # -v $(pwd)/custom_nodes/ComfyUI-3D-Pack/:/workdir/ComfyUI/custom_nodes/ComfyUI-3D-Pack/ \
