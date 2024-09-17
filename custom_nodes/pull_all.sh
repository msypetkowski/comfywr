function do_one {
    pushd $2
    git checkout $1
    git pull origin $1
    popd
}

do_one main ComfyUI-Advanced-ControlNet
do_one main ComfyUI-AnimateDiff-Evolved
do_one main comfyui_controlnet_aux
do_one main ComfyUI_FizzNodes
do_one main ComfyUI-Frame-Interpolation
do_one master ComfyUI-post-processing-nodes
do_one main ComfyUI-VideoHelperSuite
do_one main was-node-suite-comfyui
do_one main ComfyUI-3D-Pack
do_one main ComfyUI_essentials
do_one main comfyui_marigold
do_one main ComfyUI_IPAdapter_plus/
do_one Main ComfyUI-Impact-Pack//
do_one Main ComfyUI-Impact-Pack/
do_one main ComfyUI_UltimateSDUpscale/
do_one main comfy-image-saver/
do_one main ComfyUI_UltimateSDUpscale/
do_one main ComfyUI-DepthAnythingV2//
do_one main ComfyUI_Transparent-Background/
do_one main ComfyUI-BRIA_AI-RMBG/
do_one main ComfyUI_Dave_CustomNode/
do_one main eden_comfy_pipelines/
do_one main efficiency-nodes-comfyui//
