import torch
import torch.nn.functional as F
from pytorch3d.renderer import SoftSilhouetteShader, BlendParams, softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes

class NormalsWithSoftSilhouetteShader(SoftSilhouetteShader):
    def __init__(self, *args, camera_relative=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_relative = camera_relative

    def create_normalmap(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", getattr(self, "cameras", None))
        if cameras is None:
            raise ValueError("cameras must be passed to the shader forward method.")

        znear = getattr(cameras, "znear", 1.0)
        zfar = getattr(cameras, "zfar", 100.0)

        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        face_normals = vertex_normals[faces]  # (F, 3, 3)

        # Optionally transform normals to camera space
        if self.camera_relative:
            R = cameras.get_world_to_view_transform().get_matrix()[:, :3, :3]  # (N, 3, 3)
            mesh_to_camera_T = R[meshes.faces_packed_to_mesh_idx()]#.transpose(1, 2)  # (F, 3, 3)
            face_normals = torch.bmm(face_normals, mesh_to_camera_T)

        # Interpolate normals per fragment
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_normals
        )  # (N, H, W, K, 3)

        pixel_normals = F.normalize(pixel_normals, dim=-1)

        # Blend normals using softmax_rgb_blend
        normal_map = softmax_rgb_blend(
            pixel_normals, fragments, self.blend_params, znear=znear, zfar=zfar
        )  # (N, H, W, 4)

        return normal_map

    def forward(self, fragments, meshes, **kwargs):
        
        sil_rgba = super().forward(fragments, meshes, **kwargs)
        alpha = sil_rgba[..., 3:4]  # (N, H, W, 1)
        
        normal_map = self.create_normalmap(fragments, meshes, **kwargs)
        mask = fragments.zbuf.gt(0).any(dim=-1, keepdim=True)
        normal_map = normal_map.add(1).div(2).mul(mask)
        # Replace alpha with silhouette alpha
        normal_map[..., 3:4] = alpha
        return normal_map