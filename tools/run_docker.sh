docker run --gpus=all --rm --net=host \
  -v $(pwd):/workdir/ \
  -v $(pwd)/downloaded_models/:/workdir/ComfyUI/models/ \
  -v $(pwd)/custom_nodes/:/workdir/ComfyUI/custom_nodes/ \
  -it comfywr:latest \
  bash -c "cd /workdir/ComfyUI/; python main.py --force-fp16"
