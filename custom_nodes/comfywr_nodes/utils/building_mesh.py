import numpy as np
import trimesh

def ensure_normals_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Calculate mesh centroid
    mesh_center = mesh.centroid

    # For each face, compute the face center
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    # Vector from mesh center to face center
    to_face = face_centers - mesh_center

    # Normalize direction vectors
    to_face /= np.linalg.norm(to_face, axis=1)[:, np.newaxis]

    # Dot product between face normal and vector pointing outward
    dot_products = np.einsum('ij,ij->i', face_normals, to_face)

    # Find faces whose normals point inward
    inward_faces = dot_products < 0

    # Flip normals by swapping 2 vertices (this changes winding order)
    mesh.faces[inward_faces] = mesh.faces[inward_faces][:, [0, 2, 1]]

    return mesh


def simple_building_mesh(width, height, depth, roof_type, roof_height):
    w, h, d = width / 2, height / 2, depth / 2
    top_y = h

    vertices = []
    faces = []

    def add_face(v0, v1, v2, v3):
        start = len(vertices)
        vertices.extend([v0, v1, v2, v3])
        faces.append([start, start + 1, start + 2])
        faces.append([start, start + 2, start + 3])

    # Add side and bottom faces
    add_face([-w, -h,  d], [ w, -h,  d], [ w, -h, -d], [-w, -h, -d])  # bottom
    add_face([-w, -h, -d], [ w, -h, -d], [ w,  h, -d], [-w,  h, -d])  # back
    add_face([ w, -h, -d], [ w, -h,  d], [ w,  h,  d], [ w,  h, -d])  # right
    add_face([ w, -h,  d], [-w, -h,  d], [-w,  h,  d], [ w,  h,  d])  # front
    add_face([-w, -h,  -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d])  # left

    if roof_type == "none":
        # Flat top
        add_face([-w,  h, -d], [ w,  h, -d], [ w,  h,  d], [-w,  h,  d])

    elif roof_type == "pyramid":
        # Apex at center
        apex = [0, top_y + roof_height, 0]
        corners = [
            [-w, h, -d], [w, h, -d], [w, h, d], [-w, h, d]
        ]
        for i in range(4):
            v0 = corners[i]
            v1 = corners[(i + 1) % 4]
            start = len(vertices)
            vertices.extend([v0, v1, apex])
            faces.append([start, start + 1, start + 2])

    elif roof_type in ("slanted_x", "slanted_z"):
        axis = 0 if roof_type == "slanted_x" else 2  # x or z

        # Define ridge vertices
        ridge_y = top_y + roof_height
        if axis == 0:
            # Slanted along X: ridge along Z
            v0 = [-w, top_y, -d]
            v1 = [ w, top_y, -d]
            v2 = [ w, top_y,  d]
            v3 = [-w, top_y,  d]
            v4 = [w, ridge_y, 0]
            v5 = [-w, ridge_y, 0]

        else:
            # Slanted along Z: ridge along X
            v0 = [-w, top_y, d]
            v1 = [-w, top_y, -d]
            v2 = [ w, top_y, -d]
            v3 = [ w, top_y, d]
            v4 = [0, ridge_y, -d]
            v5 = [0, ridge_y,  d]
        add_face(v0, v5, v4, v1)
        add_face(v2, v3, v5, v4)
        # Two gables: left/right
        start = len(vertices)
        vertices.extend([v1, v4, v2])
        faces.append([start, start + 1, start + 2])
        start = len(vertices)
        vertices.extend([v3,v5,v0])
        faces.append([start, start + 1, start + 2])

    # Finalize geometry
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    building = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    building = ensure_normals_outward(building)

    return building
