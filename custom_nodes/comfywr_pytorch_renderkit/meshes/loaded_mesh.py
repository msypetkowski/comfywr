from typing import Tuple, Union
from custom_nodes.comfywr_nodes.utils.parametric.utils import ParametricMesh, Transform
import torch
import torch.nn as nn
import trimesh


class LoadedMesh(ParametricMesh):
    """
    Loads an arbitrary mesh from a Trimesh object or file path.

    Optionally applies per-vertex offsets as a learnable parameter.
    Offsets can be locked (frozen) via `lock_offsets`.
    """
    def __init__(self,
                 mesh: Union[trimesh.Trimesh, str],
                 lock_offsets: bool = False,
                 transform: Transform = None):
        # Load trimesh if path provided
        if isinstance(mesh, str):
            tm = trimesh.load(mesh, process=False)
        else:
            tm = mesh
        self._loaded_verts = torch.from_numpy(tm.vertices).float()
        self._faces = torch.from_numpy(tm.faces).long()
        super().__init__(transform)
        # now register offsets as parameter
        self.offsets = nn.Parameter(torch.zeros_like(self._loaded_verts), requires_grad=not lock_offsets)
        self.recalculate()

    def recalculate(self) -> None:
        """
        Update base vertices by adding offsets to original loaded vertices.
        """
        # Compute new verts with offsets
        self._base_verts = self._loaded_verts + self.offsets

    def _create_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Used by base init to set up base geometry
        return self._loaded_verts + self.offsets, self._faces