import torch
import torch.nn as nn
import numpy as np
import comfy.model_management
import torch.nn.functional as F
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings, BlendParams, SoftSilhouetteShader
)
from ..cameras.parametric_camera import ParametricCamera
from ..shaders.normals_with_soft_silhouette import NormalsWithSoftSilhouetteShader
from ..optimizations.early_stop_optimization import optimize_mesh_to_image

class NormalMapGradientOptimization:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_normalmap": ("IMAGE",),
                "parametric_mesh": ("PARAMETRIC_MESH",),
                "iterations": ("INT", {"default": 6000, "min": 1, "max": 100000}),
                "initial_fov": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 179.0}),
                "lock_camera": ("BOOLEAN", {"default": False}),
                "downscale_factor": ("INT", {"default": 4, "min": 1, "max": 64 }),
            },
            "optional": {
                "camera_poses": ("ORBIT_CAMPOSES",),
            }
        }

    RETURN_TYPES = (
        "PARAMETRIC_MESH",
        "ORBIT_CAMPOSES",
        "FLOAT",
        "IMAGE",
    )
    RETURN_NAMES = (
        "optimized_mesh",
        "camera_poses",
        "camera_fov",
        "rendered_normals_rgba",
    )

    FUNCTION = "optimize"
    CATEGORY = "comfy3dgen/optimization"

    def prepare_renderer(self, image_size):
        rast = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-6,
            faces_per_pixel=5,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rast),
            shader=NormalsWithSoftSilhouetteShader(camera_relative=True),
        )
        return renderer

    def optimize(self, target_normalmap, parametric_mesh,
                 iterations, initial_fov, lock_camera, downscale_factor, camera_poses=None):
        
        with torch.inference_mode(False):

            learning_rate = 3e-4
        
            device = comfy.model_management.get_torch_device()
            target_normalmap = target_normalmap.to(device)
            
            # --- Camera Setup ---
            
            if camera_poses is not None:
                print(f"Camera poses: {camera_poses}, FOV: {initial_fov}")
                # Unpack the first camera pose: radius, elev, azim, center_x/y/z
                radius, elev, azim, offset_x, offset_y, offset_z = camera_poses[0]
                offset = (offset_x, offset_y, offset_z)
            else:
                radius, elev, azim = 2.5, 0.0, 0.0
                offset = (0.0, 0.0, 0.0)

            camera = ParametricCamera(
                fov=initial_fov,
                dist=radius,
                elev=-elev,
                azim=azim,
                offset=offset,
                lock_fov=lock_camera,
                lock_distance=lock_camera,
                lock_angles=lock_camera,
            ).to(device)

            # --- Mesh ---
            # --- Mesh Copy ---
            mesh = parametric_mesh.clone()

            # --- Target Downscaling ---
            if downscale_factor > 1:
                downscaled_target = F.interpolate(
                    target_normalmap.permute(0, 3, 1, 2),
                    scale_factor=(1./downscale_factor),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                downscaled_target = target_normalmap

            # --- Renderer Setup for Optimization ---
            renderer = self.prepare_renderer(image_size=downscaled_target.shape[1:3])

            # --- Stage 1 Optimization (mesh + camera) ---

            params = list(mesh.parameters())
            if not lock_camera:
                params += list(camera.parameters())
            optimizer = torch.optim.AdamW(params, lr=learning_rate)

            downscaled_target = downscaled_target.float().to(device)

            mesh, camera, _, best_render = optimize_mesh_to_image(
                mesh, camera, renderer, downscaled_target,
                optimizer=optimizer,
                iterations=iterations // 2,
                early_stop_window=150,
                early_stop_percent=1e-5,
                alpha_weight=4.0,
                device=device
            )

            # --- Stage 2 Optimization (mesh only, mask target) ---
            mask = best_render[..., -1:].gt(0).float()
            masked_target = downscaled_target * mask

            optimizer = torch.optim.AdamW(mesh.parameters(), lr=learning_rate)

            mesh, camera, _, _ = optimize_mesh_to_image(
                mesh, camera, renderer, masked_target,
                optimizer=optimizer,
                iterations=iterations // 2,
                early_stop_window=200,
                early_stop_percent=1e-5,
                alpha_weight=4.0,
                device=device
            )

        # --- Final full-res renderer for output ---
        final_renderer = self.prepare_renderer(image_size=target_normalmap.shape[1:3])

        with torch.no_grad():
            final_render = final_renderer(
                meshes_world=mesh(),
                cameras=camera(device)
            )

        final_pose = [
            camera.dist.item(),
            -camera.elev.item(),
            camera.azim.item(),
            *camera.offset[0].detach().tolist(),
        ]

        return mesh, [final_pose], camera.fov.item(), final_render


class SilhouetteGradientOptimization:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_mask": ("MASK",),            # ComfyUI mask input
                "parametric_mesh": ("PARAMETRIC_MESH",),
                "iterations": ("INT", {"default": 3000, "min": 1, "max": 100000}),
                "initial_fov": ("FLOAT", {"default": 45.0, "min": 1.0, "max": 179.0}),
                "lock_camera": ("BOOLEAN", {"default": False}),
                "downscale_factor": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {
                "camera_poses": ("ORBIT_CAMPOSES",),
            }
        }

    RETURN_TYPES = (
        "PARAMETRIC_MESH",
        "ORBIT_CAMPOSES",
        "FLOAT",
        "IMAGE",
    )
    RETURN_NAMES = (
        "optimized_mesh",
        "camera_poses",
        "camera_fov",
        "final_image"
    )

    FUNCTION = "optimize"
    CATEGORY = "comfy3dgen/optimization"

    def prepare_renderer(self, image_size):
        rast = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-6,
            faces_per_pixel=5,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rast),
            shader=SoftSilhouetteShader()
        )
        return renderer

    def optimize(self, target_mask, parametric_mesh,
                 iterations, initial_fov, lock_camera, downscale_factor, camera_poses=None):

        with torch.inference_mode(False):

            learning_rate = 1e-3
        
            device = comfy.model_management.get_torch_device()
            # Convert mask to single-channel float
            mask = target_mask.to(device).float()

            # Build 4-channel target: RGB=1, A=mask
            ones = torch.ones_like(mask)
            target_rgba = torch.stack([ones, ones, ones, mask], dim=-1)


            # Downscale if requested
            if downscale_factor > 1:
                target_ds = F.interpolate(
                    target_rgba.permute(0, 3, 1, 2),
                    scale_factor=(1. / downscale_factor), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                target_ds = target_rgba

            # Camera initialization
            if camera_poses:
                r, e, a, ox, oy, oz = camera_poses[0]
                offset = (ox, oy, oz)
            else:
                r, e, a = 2.5, 0.0, 0.0
                offset = (0.0, 0.0, 0.0)
            camera = ParametricCamera(
                fov=initial_fov, dist=r, elev=-e, azim=a, offset=offset,
                lock_fov=lock_camera, lock_distance=lock_camera, lock_angles=lock_camera
            ).to(device)

            # Prepare mesh copy
            mesh = parametric_mesh.clone()

            # Soft silhouette renderer for normalmap optimizer
            renderer = self.prepare_renderer(image_size=target_ds.shape[1:3])

            params = list(mesh.parameters())
            if not lock_camera:
                params += list(camera.parameters())
            optimizer = torch.optim.AdamW(params, lr=learning_rate)

            # Delegate to existing optimizer
            optimized_mesh, optimized_camera, _, _ = optimize_mesh_to_image(
                mesh=mesh, camera=camera, renderer=renderer, target_image=target_ds,
                optimizer=optimizer,
                iterations=iterations,
                early_stop_window=150,
                early_stop_percent=1e-5,
                device=device
            )

        final_renderer = self.prepare_renderer(image_size=target_rgba.shape[1:3])

        with torch.no_grad():
            final_render = final_renderer(
                meshes_world=mesh(),
                cameras=camera(device)
            )

        # Prepare outputs
        pose = [[
            optimized_camera.dist.item(),
            -optimized_camera.elev.item(),
            optimized_camera.azim.item(),
            *optimized_camera.offset[0].detach().tolist(),
        ]]
        return optimized_mesh, pose, optimized_camera.fov.item(), final_render
