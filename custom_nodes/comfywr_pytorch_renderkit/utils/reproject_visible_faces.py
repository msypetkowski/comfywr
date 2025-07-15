import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    FoVPerspectiveCameras,
    TexturesUV,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes

def reproject_visible_faces(
    meshes: Meshes,                   # Meshes with UV textures, batch size 1
    camera: FoVPerspectiveCameras,
    reference_image: torch.Tensor,    # (H, W, 3), float RGB [0..1]
    replace_all: bool = True          # if True, start from zeros; otherwise update existing albedo
) -> torch.Tensor:
    """
    Re-project visible faces from reference_image into UV-space and update the
    existing albedo texture in `meshes` in-place. Controlled by `replace_all` flag:
      - True: initialize reprojection map with zeros, updating only visible pixels
      - False: initialize with existing albedo, overwriting visible pixels

    Returns:
        new_albedo: updated albedo map tensor (H_tex, W_tex, 3)
    """
    device = reference_image.device

    # 1) Retrieve existing albedo and its size
    albedo = meshes.textures.maps_padded()[0]        # (H_tex, W_tex, 3)
    uv_H, uv_W = albedo.shape[:2]

    # 2) Rasterize reference image
    reference_image = reference_image.squeeze(0)
    H, W = reference_image.shape[:2]
    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1
    )
    fragments = MeshRasterizer(cameras=camera, raster_settings=raster_settings)(meshes)
    pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H, W)

    # 3) Interpolate per-pixel UVs
    verts_uvs = meshes.textures.verts_uvs_list()[0]  # (V_uv, 2)
    faces_uvs = meshes.textures.faces_uvs_list()[0]  # (F, 3)
    face_uvs = verts_uvs[faces_uvs]                  # (F, 3, 2)
    uv_map = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_uvs
    )                                                # (1, H, W, 2)
    uv = uv_map[0, ..., 0, :]                       # (H, W, 2)

    # z = (1 - uv.sum(-1, keepdim=True)).clip(0,1)
    # return torch.cat((uv,z), dim=-1).unsqueeze(0) # TODO just debug!!

    # 4) Initialize reprojection map based on flag
    uv_reproj = torch.zeros_like(albedo) if replace_all else albedo.clone()

    # 5) Scatter visible pixels
    valid = pix_to_face >= 0
    uvs = uv[valid]                                 # (P, 2)
    pix_colors = reference_image[valid]
    u_pix = (uvs[:, 0] * (uv_W - 1)).long()
    v_pix = (uvs[:, 1] * (uv_H - 1)).long()

    idx = v_pix * uv_W + u_pix

    

    flat_reproj = uv_reproj.view(-1, 3)

    # Hacky way of filling neighbouring pixels
    max_idx = len(flat_reproj)-1
    partial_colors = pix_colors/4
    neighbouring_idx = torch.cat((idx-1, idx-uv_W, idx-uv_W-1, idx+1, idx+uv_W, idx+uv_W+1)).clamp(0,max_idx)
    flat_reproj[neighbouring_idx] = 0
    flat_reproj[(idx-1).clamp(0,max_idx)] += partial_colors
    flat_reproj[(idx-uv_W).clamp(0,max_idx)] += partial_colors
    flat_reproj[(idx-uv_W-1).clamp(0,max_idx)] += partial_colors
    flat_reproj[(idx+1).clamp(0,max_idx)] += partial_colors
    flat_reproj[(idx+uv_W).clamp(0,max_idx)] += partial_colors
    flat_reproj[(idx+uv_W+1).clamp(0,max_idx)] += partial_colors

    #Filling proper pixels
    flat_reproj[idx] = pix_colors
    new_albedo = flat_reproj.view(uv_H, uv_W, 3)

    return new_albedo
