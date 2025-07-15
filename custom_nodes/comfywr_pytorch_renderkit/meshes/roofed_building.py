from .parametric_mesh import ParametricMesh
from.cuboid import Cuboid
from ..transforms.transform import Transform
import torch
import torch.nn as nn
import trimesh

from typing import Tuple


class RoofedBuilding(ParametricMesh):
    _signs = torch.tensor([
        # bottom (-y)
        [-1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [-1, -1,  1],
        # back (-z)
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        # front (+z)
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        # right (+x)
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1, -1],
        # left (-x)
        [-1, -1, -1], [-1, -1,  1], [-1,  1,  1], [-1,  1, -1]
    ], dtype=torch.float32)
    # _top_signs = torch.tensor([[-1,  1, -1], [ 1,  1, -1], [ 1,  1,  1], [-1,  1,  1]], dtype=torch.float32)
    _top_signs = torch.tensor([[-1,  1, -1], [-1,  1,  1], [ 1,  1,  1], [ 1,  1, -1]], dtype=torch.float32)

    _face_indices = torch.tensor([
            [2, 1, 0], [3, 2, 0],       # bottom
            [5, 6, 4], [6, 7, 4],       # back
            [10, 9,8], [11,10,8],       # front
            [13,14,12], [14,15,12],     # right
            [18,17,16], [19,18,16],     # left
    ], dtype=torch.int64)

    def __init__(self, width=1.0, height=1.0, depth=1.0,
                 roof_size=(0.2, 0.3, 0.2), export_without_roof=False, transform: Transform = None):
        self.sizes = torch.tensor([width, height, depth], dtype=torch.float32)
        self.roof_size = torch.tensor(roof_size, dtype=torch.float32)
        self.export_without_roof = export_without_roof
        super().__init__(transform or Transform(lock_scale=True, lock_rotation=True, lock_translation=True))

        # Dimensions
        self.sizes = nn.Parameter(self.sizes.clone(), requires_grad=True)
        # Roof insets (x_inset %, height %, z_inset %)
        self.roof_size = nn.Parameter(self.roof_size.clone(), requires_grad=True)

        self.recalculate()

    def calculate_verts(self, sizes, roof_size) -> torch.Tensor:
        device = self.sizes.device
        base_corners = self._signs.to(device) * (sizes / 2.0)

        top_verts = self._top_signs.to(device) * (sizes / 2.0)
        h_offset = (sizes / 2.0 + roof_size * sizes) * torch.tensor([0,1,0], dtype=torch.float32, device=device).unsqueeze(0)
        inset = roof_size.clamp(0,1) * torch.tensor([1,0,1], dtype=torch.float32, device=device).unsqueeze(0)
        roof_top_points = top_verts * inset + h_offset

        roof_sides = torch.stack([
            top_verts.select(-2,0), roof_top_points.select(-2,0), roof_top_points.select(-2,1), top_verts.select(-2,1),
            top_verts.select(-2,3), roof_top_points.select(-2,3), roof_top_points.select(-2,2), top_verts.select(-2,2),
            top_verts.select(-2,1), roof_top_points.select(-2,1), roof_top_points.select(-2,2), top_verts.select(-2,2),
            top_verts.select(-2,0), roof_top_points.select(-2,0), roof_top_points.select(-2,3), top_verts.select(-2,3),
        ], dim=-2).to(device)

        roof_top = roof_top_points
        return torch.cat([base_corners, roof_top, roof_sides], dim=-2)

    def _create_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        verts = self.calculate_verts(self.sizes, self.roof_size)
        roof_faces = self._face_indices + 20
        faces = torch.cat([self._face_indices, roof_faces], dim=0)
        return verts, faces

    def recalculate(self) -> torch.Tensor:
        verts = self.calculate_verts(self.sizes, self.roof_size)
        self._base_verts = verts
        return verts.unsqueeze(0)

    def cuboid(self) -> Cuboid:
        with torch.no_grad():
            params = [x.clone() for x in self.transform.parameters()]
            locks = [not x.requires_grad for x in self.transform.parameters()]
            transform = Transform(*params, *locks)
            return Cuboid(*self.sizes, transform=transform).to(device=self.sizes.device)
    
    def export_trimesh(self, apply_transform=True) -> trimesh.Trimesh:
        if self.export_without_roof:
            return self.cuboid().export_trimesh(apply_transform)
        else:
            return super().export_trimesh(apply_transform=True)

    def clone(self) -> "RoofedBuilding":
        new_mesh = super().clone()
        new_mesh.export_without_roof = self.export_without_roof
        return new_mesh