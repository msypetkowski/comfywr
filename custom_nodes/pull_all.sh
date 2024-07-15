function do_one {
    git checkout $1
    git pull origin $1
}

pushd ComfyUI-Advanced-ControlNet
do_one main
popd
pushd ComfyUI-AnimateDiff-Evolved
do_one main
popd
pushd comfyui_controlnet_aux
do_one main
popd
pushd ComfyUI_FizzNodes
do_one main
popd
pushd ComfyUI-Frame-Interpolation
do_one main
popd
pushd ComfyUI-post-processing-nodes
do_one master
popd
pushd ComfyUI-VideoHelperSuite
do_one main
popd
pushd was-node-suite-comfyui
do_one main
popd



pushd ComfyUI-3D-Pack
do_one main
popd
pushd ComfyUI_essentials
do_one main
popd
pushd comfyui_marigold
do_one main
popd

pushd ComfyUI_IPAdapter_plus/
do_one main
popd

pushd ComfyUI-Impact-Pack//
do_one Main
popd
pushd ComfyUI-Impact-Pack/
do_one Main
popd

pushd ComfyUI_UltimateSDUpscale/
do_one main
popd

# pushd efficiency-nodes-comfyui//
# git pull origin main
# popd

pushd comfy-image-saver/
do_one main
popd

# pushd ComfyUI-InstantMesh/
# git pull origin master
# popd

pushd ComfyUI_UltimateSDUpscale/
do_one main
popd

# pushd ComfyUI-Unique3D/
# do_one master
# popd

pushd ComfyUI-DepthAnythingV2//
do_one main
popd

pushd ComfyUI_Transparent-Background/
do_one main
popd

pushd ComfyUI-BRIA_AI-RMBG/
do_one main
popd

pushd ComfyUI_Dave_CustomNode/
do_one main
popd
