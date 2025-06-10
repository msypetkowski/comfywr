import numpy as np
import torch
import trimesh
from enum import Enum


class Direction(Enum):
    """
    Cardinal directions for mesh orientation around the up (Z) axis.
    """
    N = 'N'
    E = 'E'
    S = 'S'
    W = 'W'


def euler_to_rotation_matrix(
    rotation: tuple[float, float, float],
    is_transformation: bool = False
) -> np.ndarray:
    """
    Create a 3×3 rotation matrix from Euler angles (in radians), optionally as a 4×4 homogeneous transform.
    Rotation order: Z (yaw), Y (pitch), X (roll).

    Args:
        rotation: (rx, ry, rz) Euler angles in radians.
        is_transformation: if True, return a 4×4 homogeneous matrix, else 3×3.
    Returns:
        Rotation matrix of shape 3×3 or 4×4.
    """
    rx, ry, rz = rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R3 = Rz @ Ry @ Rx
    if is_transformation:
        R4 = np.eye(4)
        R4[:3, :3] = R3
        return R4
    return R3


def direction_matrix(
    direction: Direction | str,
    is_transformation: bool = True
) -> np.ndarray:
    """
    Return an exact rotation matrix for a cardinal direction around Z-axis.

    Args:
        direction: Direction enum or name ('N', 'E', 'S', 'W').
        is_transformation: if True, return 4×4 homogeneous, else 3×3.
    Returns:
        Rotation matrix of shape 4×4 or 3×3.
    """
    # validate via Enum
    direction = Direction(direction)
    mats = {
        Direction.N: np.eye(3),
        Direction.E: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        Direction.S: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        Direction.W: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    }
    R3 = mats[direction]
    if is_transformation:
        R4 = np.eye(4)
        R4[:3, :3] = R3
        return R4
    return R3


def compute_transformation_matrix(
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Compute a 4×4 transformation matrix applying rotation, scaling about a pivot, then translation.
    """
    tx, ty, tz = translation
    sx, sy, sz = scale
    P = np.eye(4)
    P[:3, 3] = pivot_point
    P_inv = np.linalg.inv(P)

    R4 = euler_to_rotation_matrix(rotation, is_transformation=True)
    S = np.diag([sx, sy, sz, 1])
    T = np.eye(4)
    T[:3, 3] = (tx, ty, tz)

    return P @ R4 @ S @ P_inv @ T


def apply_transformation(
    mesh,
    matrix: np.ndarray
):
    """
    Apply a 4×4 transformation matrix to a mesh (trimesh.Trimesh or object with `.v`).
    """
    if isinstance(mesh, trimesh.Trimesh):
        mesh.apply_transform(matrix)
        return mesh
    if not hasattr(mesh, 'v'):
        raise TypeError("Mesh object must be trimesh.Trimesh or have a `v` attribute")
    verts = mesh.v.cpu().numpy()
    hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
    new_verts = (hom @ matrix.T)
    new_verts = new_verts[:, :3] / new_verts[:, -1:]
    mesh.v[:] = torch.from_numpy(new_verts)
    return mesh

