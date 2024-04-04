ComfyUI wrappers and utils
==========================

This is purely experimental, WIP, convenience repo, with just some wrappers and relatively simple scripts for diffusion models inference.

ComfyUI is a very convinient UI for image generation, which attracts developers to make their own plugins.
While the diffusers library may seem to be a clean base for inference experiments,
using ComfyUI nodes as a python library backend seems a silly idea,
but due to the UI being thoroughly tested (by tons of users) and having many available plugins -- it has some advantages.

This library basically wraps ComfyUI nodes into very simple python functions so that it is easy to write
very complex inference scripts featuring multiple checkpoints, multiple controlnets, etc.


Setup environment
-----------------

Fetch submodules, apply patches, build image, run jupyterlab:
```bash
git submodule update --init --recursive
./tools/apply_patches.sh
./tools/build_docker.sh

# download some basic checkpoints
./tools/download_basic_checkpoints.sh


# run ComfyUI
./tools/run_docker.sh

# for jupyter notebooks server
# ./tools/run_docker_nb.sh
```

Examples
--------

See example notebooks in `./notebooks/`
