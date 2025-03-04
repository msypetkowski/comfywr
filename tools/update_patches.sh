pushd ComfyUI/
git diff > ../patches/comfyui.diff
popd

pushd custom_nodes/ComfyUI-3D-Pack/
git diff > ../../patches/comfyui_3d_pack.diff
popd

pushd custom_nodes/ComfyUI-ModelUnloader//
git diff > ../../patches/model_unloader.diff
popd

pushd custom_nodes/comfyui_marigold/
git diff > ../../patches/marigold.diff
popd
