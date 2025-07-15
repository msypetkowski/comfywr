import copy
import torch
import numpy as np
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes


def mesh_copy(mesh):
    """
    Create a shallow copy of a mesh, but for each attribute:
      - If it's a torch.Tensor, clone it.
      - If it's a numpy.ndarray, copy it.
      - Otherwise, shallow-copy it via copy.copy().
    """
    # Create new instance without calling __init__
    new_mesh = mesh.__class__.__new__(mesh.__class__)
    
    for attr, value in mesh.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(new_mesh, attr, value.clone())
        elif isinstance(value, np.ndarray):
            setattr(new_mesh, attr, value.copy())
        else:
            # Shallow copy all other attribute values
            setattr(new_mesh, attr, copy.copy(value))
    
    return new_mesh

def export_mesh_to_pytorch3d(
    mesh,                  # custom Mesh with v, f, vt, ft, albedo
    device: torch.device   # target device for tensors
) -> Meshes:
    """
    Convert a custom mesh object into a PyTorch3D Meshes with UV textures.

    Args:
        mesh: custom Mesh with attributes:
            v: (V, 3) float tensor of vertices
            f: (F, 3) long tensor of face indices into v
            vt: (V_uv, 2) float tensor of UV coords per vertex
            ft: (F, 3) long tensor of face indices into vt
            albedo: (H_tex, W_tex, 3) float tensor of texture image [0..1]
        device: torch.device for output tensors

    Returns:
        mesh_p3d: a Meshes instance with TexturesUV
    """
    # Vertices and faces
    verts = mesh.v.unsqueeze(0).to(device)    # (1, V, 3)
    faces = mesh.f.unsqueeze(0).to(device)    # (1, F, 3)
    # UV coordinates and indices
    verts_uvs = mesh.vt.to(device)           # (V_uv, 2)
    faces_uvs = mesh.ft.to(device)           # (F, 3)
    # Texture map as (1, H_tex, W_tex, 3)
    tex = mesh.albedo.to(device)
    print(f"texture shape: {tex.shape}")
    tex_map = tex.permute(2, 0, 1).unsqueeze(0).permute(0, 2, 3, 1)
    print(f"map shape: {tex_map.shape}")
    # Create TexturesUV and wrap in Meshes
    textures = TexturesUV(
        maps=[tex],
        faces_uvs=[faces_uvs],
        verts_uvs=[verts_uvs]
    )
    mesh_p3d = Meshes(verts=verts, faces=faces, textures=textures)
    return mesh_p3d