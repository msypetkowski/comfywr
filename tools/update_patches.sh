# pushd ComfyUI/
# git apply ../patches/cui.patch
# popd

pushd custom_nodes/ComfyUI-3D-Pack/
git diff > ../../patches/comfyui_3d_pack.diff
popd

pushd custom_nodes/ComfyUI-ModelUnloader//
git diff ../../patches/model_unloader.diff
popd
