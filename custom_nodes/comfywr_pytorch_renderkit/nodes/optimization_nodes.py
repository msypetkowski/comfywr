import torch
import torch.nn as nn
import numpy as np
import comfy.model_management
import torch.nn.functional as F
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader
)
from ..cameras.parametric_camera import ParametricCamera
from ..shaders.normals_with_soft_silhouette import NormalsWithSoftSilhouetteShader
from ..optimizations.early_stop_optimization import optimize_mesh_to_image
from abc import ABC, abstractmethod

class BaseGradientOptimization(ABC):
    """
    Abstract base for mesh optimization nodes. Implements shared optimize() logic.
    Subclasses override renderer and target preparation, and can override camera init or core loop.
    """
    # Default learning rate; subclasses override as needed
    DEFAULT_LR: float = 1e-3

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls):
        pass

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
        "output_image",
    )
    FUNCTION = "optimize"
    CATEGORY = "comfy3dgen/optimization"

    def optimize(
        self,
        target_image: torch.Tensor,
        parametric_mesh,
        iterations: int,
        initial_fov: float,
        lock_camera: bool,
        downscale_factor: int,
        camera_poses=None,
    ):
        # 1. enable gradients & get device
        device = comfy.model_management.get_torch_device()
        with torch.inference_mode(False):

            # 2. learning rate
            lr = self.DEFAULT_LR

            # 3. prepare target
            target = self._prepare_target(target_image).to(device)

            # 4. downscale for speed
            if downscale_factor > 1:
                target = F.interpolate(
                    target.permute(0, 3, 1, 2),
                    scale_factor=(1/downscale_factor),
                    mode="bilinear",
                    align_corners=False
                ).permute(0, 2, 3, 1)

            # 5. initialize camera
            camera = self._init_camera(
                initial_fov=initial_fov,
                lock_camera=lock_camera,
                camera_poses=camera_poses,
            ).to(device)

            # 6. clone mesh
            mesh = parametric_mesh.clone().to(device)

            # 7. build optimizer
            params = list(mesh.parameters())
            if not lock_camera:
                params += list(camera.parameters())
            optimizer = torch.optim.AdamW(params, lr=lr)

            # 8. core optimization
            mesh, camera = self._core_optimization(
                mesh, camera, self._get_renderer(target.shape[1:3]),
                target, optimizer, iterations, device
            )

        # 9. final full-res render
        with torch.no_grad():
            final_render = self._get_renderer(target_image.shape[1:3])(
                meshes_world=mesh(), cameras=camera(device)
            )

        # 10. extract pose and fov
        pose, fov = self._get_camera_output(camera)
        return mesh, [pose], fov, final_render

    @abstractmethod
    def _get_renderer(self, image_size: tuple) -> MeshRenderer:
        """Return a MeshRenderer instance configured for this subclass."""
        pass

    def _prepare_target(self, target_image: torch.Tensor) -> torch.Tensor:
        """Convert the raw input tensor into the desired optimization target."""
        return target_image

    def _init_camera(self, initial_fov: float, lock_camera: bool, camera_poses=None):
        if camera_poses:
            r, e, a, ox, oy, oz = camera_poses[0]
            offset = (ox, oy, oz)
        else:
            r, e, a = 2.5, 0.0, 0.0
            offset = (0.0, 0.0, 0.0)
        return ParametricCamera(
            fov=initial_fov,
            dist=r,
            elev=-e,
            azim=a,
            offset=offset,
            lock_fov=lock_camera,
            lock_distance=lock_camera,
            lock_angles=lock_camera,
        )

    def _core_optimization(
        self, mesh, camera, renderer, target, optimizer, iterations, device
    ):
        optimized_mesh, optimized_camera, _, _ = optimize_mesh_to_image(
            mesh=mesh,
            camera=camera,
            renderer=renderer,
            target_image=target,
            optimizer=optimizer,
            iterations=iterations,
            early_stop_window=150,
            early_stop_percent=1e-5,
            device=device,
        )
        return optimized_mesh, optimized_camera

    def _get_camera_output(self, camera):
        pose = [
            camera.dist.item(),
            -camera.elev.item(),
            camera.azim.item(),
            *camera.offset[0].tolist(),
        ]
        return pose, camera.fov.item()


class NormalMapGradientOptimization(BaseGradientOptimization):
    DEFAULT_LR = 3e-4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "parametric_mesh": ("PARAMETRIC_MESH",),
                "iterations": ("INT", {"default": 6000, "min": 1}),
                "initial_fov": ("FLOAT", {"default": 45.0, "min": 1.0}),
                "lock_camera": ("BOOLEAN", {"default": False}),
                "downscale_factor": ("INT", {"default": 4, "min": 1}),
            },
            "optional": {"camera_poses": ("ORBIT_CAMPOSES",)},
        }

    def _get_renderer(self, image_size):
        rast = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-6,
            faces_per_pixel=5,
        )
        shader = NormalsWithSoftSilhouetteShader(camera_relative=True)
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rast),
            shader=shader,
        )

    def _core_optimization(
        self, mesh, camera, renderer, target, optimizer, iterations, device
    ):
        mesh, camera, _, best_render = optimize_mesh_to_image(
            mesh=mesh,
            camera=camera,
            renderer=renderer,
            target_image=target,
            optimizer=optimizer,
            iterations=iterations // 2,
            early_stop_window=150,
            early_stop_percent=1e-5,
            alpha_weight=4.0,
            device=device,
        )
        mask = best_render[..., -1:].gt(0).float()
        masked_target = target * mask
        optimizer = torch.optim.AdamW(mesh.parameters(), lr=self.DEFAULT_LR)
        return optimize_mesh_to_image(
            mesh=mesh,
            camera=camera,
            renderer=renderer,
            target_image=masked_target,
            optimizer=optimizer,
            iterations=iterations // 2,
            early_stop_window=200,
            early_stop_percent=1e-5,
            alpha_weight=4.0,
            device=device,
        )[:2]


class SilhouetteGradientOptimization(BaseGradientOptimization):
    DEFAULT_LR = 1e-3

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("MASK",),
                "parametric_mesh": ("PARAMETRIC_MESH",),
                "iterations": ("INT", {"default": 3000, "min": 1}),
                "initial_fov": ("FLOAT", {"default": 45.0, "min": 1.0}),
                "lock_camera": ("BOOLEAN", {"default": False}),
                "downscale_factor": ("INT", {"default": 4, "min": 1}),
            },
            "optional": {"camera_poses": ("ORBIT_CAMPOSES",)},
        }

    def _get_renderer(self, image_size):
        rast = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-6,
            faces_per_pixel=5,
        )
        shader = SoftSilhouetteShader()
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rast),
            shader=shader,
        )

    def _prepare_target(self, target_image):
        ones = torch.ones_like(target_image)
        target_rgba = torch.stack([ones, ones, ones, target_image], dim=-1)

        return target_rgba
    