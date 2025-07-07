import os
import folder_paths
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

# Import model components from the Orient Anything repo
# from vision_tower import DINOv2_MLP
# from inference import get_3angle

# import importlib
# DINOv2_MLP = importlib.import_module('custom_nodes.Orient-Anything.vision_tower').DINOv2_MLP
# get_3angle = importlib.import_module('custom_nodes.Orient-Anything.inference').get_3angle

import os, sys
# Make sure this path points to your submodule directory
repo_root = os.path.join(os.path.dirname(__file__), "..", "Orient-Anything")
repo_root = os.path.normpath(repo_root)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Now imports will resolve inside that directory
from vision_tower import DINOv2_MLP
from inference import get_3angle


class OrientAnythingNode:
    CATEGORY = "OrientAnything"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flip_elevation": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("azimuth", "elevation", "azimuth_string", "elevation_string")
    FUNCTION = "predict_orientation"

    def __init__(self):
        # Set up computation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        self.dino_mode = 'large'
        self.in_dim = 1024
        self.out_dim = 360 + 180 + 180 + 2  # azimuth + polar + roll + confidence

        orient_model_dir = os.path.join(folder_paths.models_dir, "orient_anything")

        # Download checkpoint from Hugging Face (auto-cached)
        ckpt_path = hf_hub_download(
            repo_id="Viglong/Orient-Anything",
            filename="croplargeEX2/dino_weight.pt",
            repo_type="model",
            cache_dir=orient_model_dir,
            resume_download=True
        )

        # Load model and weights
        self.model = DINOv2_MLP(
            dino_mode=self.dino_mode,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        )
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load preprocessing pipeline
        self.val_preprocess = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large",
            cache_dir="./models/orient_anything"
        )

    def predict_orientation(self, image, flip_elevation):
        # Convert image to PIL if necessary
        np_image = image.squeeze(0).mul(255).cpu().numpy().astype(np.uint8)
        image = Image.fromarray(np_image)

        # Inference
        angles = get_3angle(image, self.model, self.val_preprocess, self.device)
        
        elevation_sign = -1 if flip_elevation else 1

        # Return azimuth and elevation (polar angle)
        azimuth = float(angles[0])
        elevation = float(angles[1]) * elevation_sign

        print(f"(Orient Anything) Azimuth: {azimuth}, Elevation: {elevation}")

        return (azimuth, elevation, f"{azimuth:.4f}", f"{elevation:.4f}")
    
