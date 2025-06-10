import torch
from ..utils.compute_normals import compute_normal_map, normalize_depth_within_alpha

class DepthToNormalMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_rgba": ("IMAGE",),  # Shape (B, H, W, 4)
                "blur_kernel": ("INT", {"default": 3, "min": 1, "max": 15}),
                "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "normalize_depth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "transform"

    CATEGORY = "comfy3dgen/conversion"

    def transform(self, depth_rgba, blur_kernel, blur_sigma, normalize_depth):

        if depth_rgba.ndim != 4 or depth_rgba.shape[-1] != 4:
            raise ValueError("Expected RGBA image with shape (B, H, W, 4)")

        # Extract depth (R) and alpha (A) from last channel dim
        depth = depth_rgba[..., 0:1].permute(0, 3, 1, 2)  # → (B, 1, H, W)
        alpha = depth_rgba[..., 3:4].permute(0, 3, 1, 2)  # → (B, 1, H, W)

        if normalize_depth:
            depth = normalize_depth_within_alpha(depth, alpha)

        normal = compute_normal_map(depth, alpha,
                                    blur_kernel_size=blur_kernel,
                                    blur_sigma=blur_sigma)
        
        normal = normal.add(1).div_(2).mul_(alpha)
        
        normal = torch.cat([normal, alpha], dim=1)

        # Return normal in ComfyUI format (B, H, W, 4)
        return (normal.permute(0, 2, 3, 1),)