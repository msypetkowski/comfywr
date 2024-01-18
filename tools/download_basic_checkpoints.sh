mkdir -p downloaded_models/checkpoints/
mkdir -p downloaded_models/controlnet/
mkdir -p downloaded_models/embeddings/
mkdir -p downloaded_models/upscale_models/
mkdir -p downloaded_models/loras/
cd downloaded_models

pushd checkpoints/
wget -nc https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors
wget -nc https://huggingface.co/Yntec/mistoonAnime2/resolve/main/mistoonAnime_v20_vae.safetensors
wget -nc https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaperXL_Turbo_dpmppSdeKarras_half_pruned_6.safetensors
popd

pushd controlnet/
wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
wget -nc https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
popd

pushd embeddings/
wget -nc https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors
popd

pushd upscale_models/
wget -nc https://huggingface.co/spaces/Marne/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth
popd

pushd loras/
wget -nc https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
popd
