ComfyUI wrappers and utils
==========================

Just a simple convenience repo around ComfyUI, pinning some stuff together.


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
