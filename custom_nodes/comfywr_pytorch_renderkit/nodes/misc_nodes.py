import torch
import comfy.model_management
from ..utils.compute_normals import compute_normal_map, normalize_depth_within_alpha
from ..utils.export_mesh import export_mesh_to_pytorch3d, mesh_copy
from ..utils.reproject_visible_faces import reproject_visible_faces
from ..cameras.parametric_camera import ParametricCamera

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


class ReprojectFromImageToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),  # Shape (B, H, W, 3)
                "mesh": ("MESH",),
                "camera_poses": ("ORBIT_CAMPOSES",),
                "fov": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 179.0}),
                "replace_albedo": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH","IMAGE",)
    RETURN_NAMES = ("textured_mesh","texture",)
    FUNCTION = "reproject"

    CATEGORY = "comfy3dgen/projections"

    def reproject(self,reference_image, mesh, camera_poses, fov, replace_albedo):
        
        radius, elev, azim, offset_x, offset_y, offset_z = camera_poses[0]
        offset = (offset_x, offset_y, offset_z)
        with torch.no_grad():
            device = comfy.model_management.get_torch_device()
            ref_image = reference_image.to(device)

            camera = ParametricCamera(
                    fov=fov,
                    dist=radius,
                    elev=-elev,
                    azim=azim,
                    offset=offset,).to(device)

            meshes = export_mesh_to_pytorch3d(mesh, device)


            new_texture = reproject_visible_faces(meshes, camera(device), reference_image = ref_image, replace_all = replace_albedo)
            new_mesh = mesh_copy(mesh)
            new_mesh.albedo = new_texture

        return (new_mesh, new_texture.unsqueeze(0),)
