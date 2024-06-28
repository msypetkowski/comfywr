pushd ComfyUI/
# git apply ../patches/cui.patch
popd

pushd custom_nodes/ComfyUI-3D-Pack/
git apply ../../patches/3d_pack.diff
popd

pushd custom_nodes/ComfyUI-ModelUnloader//
git apply ../../patches/model_unloader.diff
popd
