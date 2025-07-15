import copy
import torch
import trimesh
import numpy as np
import cv2
from PIL import Image, ImageDraw


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

def mesh_to_trimesh(mesh_obj) -> trimesh.Trimesh:
    """
    WARNING: AI generated

    Convert a custom Mesh instance into a trimesh.Trimesh with geometry,
    normals, colors, UVs, and textures if present.

    WARNING: This function does NOT carry over any PBR material or metallic-roughness
    information. Only geometry, normals, vertex colors, UV coordinates, and a single
    albedo texture are supported.
    """
    # Geometry
    verts = mesh_obj.v.detach().cpu().numpy()
    faces = mesh_obj.f.detach().cpu().numpy()

    # Prepare kwargs
    kwargs = {}

    # Normals
    if hasattr(mesh_obj, 'vn') and mesh_obj.vn is not None:
        kwargs['vertex_normals'] = mesh_obj.vn.detach().cpu().numpy()

    # Vertex colors
    if hasattr(mesh_obj, 'vc') and mesh_obj.vc is not None:
        cols = (mesh_obj.vc.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)
        kwargs['vertex_colors'] = cols

    # UVs + texture
    if hasattr(mesh_obj, 'vt') and hasattr(mesh_obj, 'ft') and mesh_obj.vt is not None:
        uv = mesh_obj.vt.detach().cpu().numpy()
        face_uv = mesh_obj.ft.detach().cpu().numpy()
        tex_img = getattr(mesh_obj, 'albedo', None)
        if tex_img is not None:
            img = tex_img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (img.clip(0,1)*255).astype(np.uint8)
            # Create TextureVisuals and assign face_uv afterwards
            vis = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)
            vis.face_uv = face_uv
        else:
            vis = trimesh.visual.texture.TextureVisuals(uv=uv)
            vis.face_uv = face_uv
        kwargs['visual'] = vis

    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, **kwargs)
    return mesh


def mesh_silhouette_images(trimesh):
    """
    WARNING: AI generated

    Generates three silhouette images of a 3D mesh projected onto the XY, YZ, and XZ planes.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh, assumed to be within bounds [-1, 1] along each axis.

    Returns:
    tuple: Three 1024x1024 binary numpy arrays representing the silhouettes on the XY, YZ, and XZ planes.
    """
    trimesh = trimesh.copy()

    # Get vertices and faces from the mesh
    vertices = trimesh.vertices  # shape (n_vertices, 3)
    faces = trimesh.faces  # shape (n_faces, 3)
    triangles = vertices[faces]  # shape (n_faces, 3, 3)

    # Prepare images and drawing contexts for each projection
    img_xy = Image.new('1', (1024, 1024), 0)
    draw_xy = ImageDraw.Draw(img_xy)

    img_yz = Image.new('1', (1024, 1024), 0)
    draw_yz = ImageDraw.Draw(img_yz)

    # img_xz = Image.new('1', (1024, 1024), 0)
    # draw_xz = ImageDraw.Draw(img_xz)

    # Mapping functions from coordinate space [-1, 1] to pixel space [0, 1023]
    def coord_to_pixel(coord):
        # return ((coord + 1.0) * 511.5).round().astype(int)
        # assert ((-1.0 <= coord) & (coord <= 1.0)).all(), coord
        coord = (coord + 1) / 2
        return (coord * 1023).round().astype(int)

    def coord_to_pixel_flipped(coord):
        # return ((1.0 - coord) * 511.5).round().astype(int)
        # assert ((-1.0 <= coord) & (coord <= 1.0)).all(), coord
        coord = (coord + 1) / 2
        return ((1 - coord) * 1023).round().astype(int)

    # Loop over each triangle to project and draw on the images

    # assert ((-1.0 <= triangles) & (triangles <= 1.0)).all()
    # assert ((-0.5 >= triangles) | (triangles >= 0.5)).any()

    for tri in triangles:
        # XY projection (view along +Z direction)
        x = tri[:, 0]
        y = tri[:, 1]
        px = coord_to_pixel(x)
        py = coord_to_pixel_flipped(y)
        points = list(zip(px, py))
        draw_xy.polygon(points, fill=1)

        # YZ projection (view along +X direction)
        y = tri[:, 1]
        z = tri[:, 2]
        px = coord_to_pixel_flipped(z)
        py = coord_to_pixel_flipped(y)
        points = list(zip(px, py))
        draw_yz.polygon(points, fill=1)

        # XZ projection (view along +Y direction)
        # x = tri[:, 0]
        # z = tri[:, 2]
        # px = coord_to_pixel(x)
        # py = coord_to_pixel_flipped(z)
        # points = list(zip(px, py))
        # draw_xz.polygon(points, fill=1)

    # Convert images to numpy arrays and return
    img_xy_array = np.array(img_xy)
    img_yz_array = np.array(img_yz)
    # img_xz_array = np.array(img_xz)

    return img_xy_array, img_yz_array


def visualize_silhouettes(silhouettes):
    vis = np.zeros((silhouettes[0][0].shape[0], silhouettes[1][1].shape[1] * 2, 3), dtype=np.uint8)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 100, 100)]
    for s, col in zip(silhouettes, colors):
        assert 0 < np.mean(s) < 1
        mask = np.concatenate(s, 1).astype(np.uint8) * 255
        assert mask.shape == vis.shape[:2]
        edges = cv2.dilate(mask, np.ones((3, 3))) - cv2.erode(mask, np.ones((3, 3)))
        vis[edges > 0] = col
    return vis