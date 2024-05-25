mkdir -p downloaded_models/checkpoints/
mkdir -p downloaded_models/controlnet/
mkdir -p downloaded_models/embeddings/
mkdir -p downloaded_models/upscale_models/
mkdir -p downloaded_models/loras/
mkdir -p downloaded_models/vae/
mkdir -p downloaded_models/lgm/
mkdir -p downloaded_models/ipadapter/
mkdir -p downloaded_models/style_models/
mkdir -p downloaded_models/clip_vision/
cd downloaded_models

pushd checkpoints/
wget -nc https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors
wget -nc https://huggingface.co/Yntec/mistoonAnime2/resolve/main/mistoonAnime_v20_vae.safetensors
wget -nc https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaperXL_Turbo_dpmppSdeKarras_half_pruned_6.safetensors
wget -nc https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt
wget -nc https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors
popd

pushd controlnet/
wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
# wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors
# wget -nc https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.fp16.safetensors -O control_v11p_sd15_openpose_fp16.safetensors
popd

pushd embeddings/
wget -nc https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors
popd

pushd upscale_models/
wget -nc https://huggingface.co/spaces/Marne/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth
wget -nc "https://civitai.com/api/download/models/125843" -O 4xUltrasharp.pt

popd

pushd loras/
wget -nc https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
wget -nc https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt

wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors
popd

pushd vae/
wget -nc https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
popd

pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/lgm/
wget -nc https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors
popd

pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/flamehaze1115/
# TODO
popd

pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/ashawkey/imagedream-ipmv-diffusers/unet/
wget -nc https://huggingface.co/ashawkey/imagedream-ipmv-diffusers/resolve/main/unet/diffusion_pytorch_model.safetensors
popd
pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/ashawkey/imagedream-ipmv-diffusers/vae/
wget -nc https://huggingface.co/ashawkey/imagedream-ipmv-diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors
popd
pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/ashawkey/imagedream-ipmv-diffusers/image_encoder/
wget -nc https://huggingface.co/ashawkey/imagedream-ipmv-diffusers/resolve/main/image_encoder/model.safetensors
popd
pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/ashawkey/imagedream-ipmv-diffusers/text_encoder/
wget -nc https://huggingface.co/ashawkey/imagedream-ipmv-diffusers/resolve/main/text_encoder/model.safetensors
popd

pushd ../custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -nc https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt
popd

pushd ipadapter/
wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors

wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin
wget -nc https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin

wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
popd


pushd style_models/
wget -nc https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/coadapter-style-sd15v1.pth
popd

pushd clip_vision/
wget -nc https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin -O clip-vit-large-patch14.bin

wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors \
    -O CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
wget -nc https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors \
    -O CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors
popd

mkdir ../custom_nodes/ComfyUI-3D-Pack/checkpoints/crm/
pushd ../custom_nodes/ComfyUI-3D-Pack/checkpoints/crm/
wget -nc https://huggingface.co/Zhengyi/CRM/resolve/main/pixel-diffusion.pth
wget -nc https://huggingface.co/Zhengyi/CRM/resolve/main/ccm-diffusion.pth
wget -nc https://huggingface.co/Zhengyi/CRM/resolve/main/CRM.pth
popd
