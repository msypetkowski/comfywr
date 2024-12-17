ComfyUI wrappers and utils
==========================

Just a simple convenience repo around ComfyUI, pinning some stuff together.


Setup environment
-----------------

Fetch submodules, apply patches, build image, run jupyterlab:
```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 ./tools/apply_patches.sh
./tools/build_docker.sh

# download some basic checkpoints
./tools/download_basic_checkpoints.sh


# run ComfyUI
./tools/run_docker.sh

# for jupyter notebooks server
# ./tools/run_docker_nb.sh
```
