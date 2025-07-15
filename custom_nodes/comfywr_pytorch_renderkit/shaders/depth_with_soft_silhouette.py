import torch
from pytorch3d.renderer import SoftSilhouetteShader, BlendParams, softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes

class SoftSilhouetteWithDepthShader(SoftSilhouetteShader):
    def __init__(
        self,
        blend_params: BlendParams = None,
        normalize_depth: bool = False,
        eps: float = 1e-6,
        invalid_depth_weight: float = 1.0,
        ):
        # If no blend_params given, let parent pick its own default
        if blend_params is None:
            super().__init__()  
        else:
            super().__init__(blend_params=blend_params)

        self.normalize_depth = normalize_depth
        self.eps = eps
        self.invalid_depth_weight = invalid_depth_weight

    def forward(self, fragments, meshes, **kwargs):
        sil_rgba = super().forward(fragments, meshes, **kwargs)
        alpha = sil_rgba[..., 3:4]                # [N, H, W, 1]

        depth_per_face = fragments.zbuf.unsqueeze(-1)  # [N, H, W, K, 1]
        znear = depth_per_face.add(self.eps).detach().reciprocal_().amax().reciprocal().sub_(self.eps).item()
        zfar = depth_per_face.amax().item()

        # Blend normals using softmax_rgb_blend
        depth = softmax_rgb_blend(
            depth_per_face, fragments, self.blend_params, znear=znear, zfar=zfar
        ).mean(dim=-1, keepdim=True)

        if self.normalize_depth:
            znear = depth.add(self.eps).detach().reciprocal_().amax().reciprocal().sub_(self.eps).item()
            zfar = depth.amax().item()
            mask = depth.gt(0).float()
            min_mask = mask.mul(znear)
            norm_mask = mask.mul(zfar).add(1).sub(min_mask).sub(mask)
            depth = depth.sub(min_mask).div(norm_mask)

        depth = depth.clamp_min(-self.invalid_depth_weight)
        # 3) concatenate depth + alpha
        out = torch.cat([depth, alpha], dim=-1)    # [N, H, W, 2]
        return out