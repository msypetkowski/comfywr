from typing import Optional, Tuple
from ..transforms.transform import Transform
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pytorch3d.structures import Meshes
import trimesh


class ParametricMesh(nn.Module, ABC):
    """
    Abstract base class for parametric meshes.

    Subclasses implement `_create_mesh()` to return raw vertices and faces.
    The base geometry is cached in `_base_verts` and `_faces`. A separate
    `transform` module applies scale/rotation/translation at render time.
    """
    def __init__(self, transform: Transform = None):
        super().__init__()
        self.transform = transform or Transform()
        # Initialize base mesh
        verts, faces = self._create_mesh()
        self._base_verts = verts
        self._faces = faces

    @abstractmethod
    def _create_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw base mesh:
        - verts: FloatTensor of shape (V, 3)
        - faces: LongTensor of shape (F, 3)
        """
        pass

    def recalculate(self) -> torch.Tensor:
        """
        Recompute and cache only the base vertices. Faces remain constant.
        """
        verts, _ = self._create_mesh()
        self._base_verts = verts
        return verts.unsqueeze(0)

    def forward(self) -> Meshes:
        verts = self.recalculate()
        verts = self.transform(verts)
        faces = self._faces.unsqueeze(0).expand(len(verts), *self._faces.shape).to(verts.device)
        return Meshes(verts=verts, faces=faces)
    

    def export_trimesh(self, apply_transform=True) -> trimesh.Trimesh:
        with torch.no_grad():
            verts = self.recalculate()
            verts = self.transform(verts) if apply_transform else verts  # (1,V,3)
            faces = self._faces.to(verts.device).unsqueeze(0)  # (1,F,3)

            # Create a single‐mesh batch:
            meshes = Meshes(verts=verts, faces=faces)

            # PyTorch3D will give you:
            #  - face_normals: (F, 3) unit normals, one per face
            #  - verts_normals: (V, 3) unit normals, one per vertex (averaged over adjacent faces)
            face_norms = meshes.faces_normals_packed()     # returns a tensor of shape (F,3)
            vert_norms = meshes.verts_normals_packed()     # returns a tensor of shape (V,3)

            # Move everything to CPU+NumPy:
            verts_np      = verts.detach().cpu().numpy()
            faces_np      = faces.detach().cpu().flip(-1).numpy() # Flip faces for trimesh format
            face_norms_np = face_norms.detach().cpu().numpy()
            vert_norms_np = vert_norms.detach().cpu().numpy()

        return trimesh.Trimesh(
            vertices=verts_np,
            faces=faces_np,
            face_normals=face_norms_np,
            vertex_normals=vert_norms_np,
            process=False,    # disable auto‐cleanup (so Trimesh won’t merge duplicates or recompute normals)
            )
    
    def clone(self) -> "ParametricMesh":
        # 1) clone transform state
        cloned_transform = self.transform.clone()
        # 2) instantiate fresh via default init with cloned transform
        new_mesh = self.__class__(transform=cloned_transform)
        # 3) load tensor data for params and buffers
        new_mesh.load_state_dict(self.state_dict())
        # 4) restore requires_grad flags
        orig_params = dict(self.named_parameters())
        for name, new_param in new_mesh.named_parameters():
            new_param.requires_grad = orig_params[name].requires_grad
        return new_mesh
