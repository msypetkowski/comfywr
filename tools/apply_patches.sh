# pushd ComfyUI/
# git apply ../patches/cui.patch
# popd

pushd custom_nodes/ComfyUI-3D-Pack/
git stash
git apply ../../patches/comfyui_3d_pack.diff
popd

pushd custom_nodes/comfyui_marigold/
git stash
git apply ../../patches/marigold.diff
popd

pushd custom_nodes/ComfyUI-ModelUnloader//
git stash
git apply ../../patches/model_unloader.diff
popd
