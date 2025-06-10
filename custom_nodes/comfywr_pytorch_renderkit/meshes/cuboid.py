from typing import Tuple
from .parametric_mesh import ParametricMesh
from ..transforms.transform import Transform
import torch
import torch.nn as nn


class Cuboid(ParametricMesh):
    """
    Parametric cuboid mesh with learnable dimensions (width, height, depth) packed
    into a single `sizes` parameter.
    """
    # Constant sign tensor for vertex definition, shape (8,3)
    _signs = torch.tensor([
        # back (-z)
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        # front (+z)
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        # bottom (-y)
        [-1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [-1, -1,  1],
        # top (+y)
        [-1,  1, -1], [ 1,  1, -1], [ 1,  1,  1], [-1,  1,  1],
        # right (+x)
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1, -1],
        # left (-x)
        [-1, -1, -1], [-1, -1,  1], [-1,  1,  1], [-1,  1, -1]
    ], dtype=torch.float32)

    def __init__(self, width=1.0, height=1.0, depth=1.0, transform: Transform = None):
        # store initial sizes for base init
        _init_sizes = torch.tensor([width, height, depth], dtype=torch.float32)
        self.sizes = _init_sizes.clone()
        super().__init__(transform or Transform(lock_scale=True, lock_rotation=True, lock_translation=True))
        # now define learnable sizes
        self.sizes = nn.Parameter(_init_sizes.clone(), requires_grad=True)
        # update base verts to use new sizes
        self.recalculate()

    def _create_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute base verts from sizes and constant signs
        verts = self._signs * (self.sizes / 2.0).unsqueeze(0)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],       # back
            [4, 6, 5], [4, 7, 6],       # front
            [8, 9,10], [8,10,11],       # bottom
            [12,14,13], [12,15,14],     # top
            [16,17,18], [16,18,19],     # right
            [20,22,21], [20,23,22]      # left
        ], dtype=torch.int64)
        return verts, faces

    def recalculate(self) -> None:
        """
        Override to update only base vertices based on `sizes`.
        """
        # Assign new base verts tensor so gradients flow through `sizes`
        self._base_verts = self._signs.to(self.sizes.device) * (self.sizes / 2.0).unsqueeze(0)


