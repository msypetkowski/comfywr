docker run --gpus=all --rm --net=host \
  --ipc=host \
  --platform linux/amd64 \
  -p 8188:8188 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd):/workdir/ \
  -v $(pwd)/downloaded_models/:/workdir/ComfyUI/models/ \
  -e DISPLAY=$DISPLAY \
  -v $(pwd)/custom_nodes/:/workdir/ComfyUI/custom_nodes/ \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /root/.u2net/:/root/.u2net/ \
  -it comfywr:latest \
  bash -c "cd /workdir/ComfyUI/; python main.py"
