import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix

class Transform(nn.Module):
    """
    Module for handling scale, rotation, and translation transforms with optional locking.
    """
    def __init__(self,
                 scale: torch.Tensor = torch.ones(3),
                 rotation: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 translation: torch.Tensor = torch.zeros(3),
                 lock_scale: bool = False,
                 lock_rotation: bool = False,
                 lock_translation: bool = False):
        super().__init__()
        self.scale = nn.Parameter(scale, requires_grad=not lock_scale)
        self.rotation = nn.Parameter(rotation, requires_grad=not lock_rotation)
        self.translation = nn.Parameter(translation, requires_grad=not lock_translation)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        v = verts * self.scale.unsqueeze(0)
        q = self.rotation / self.rotation.norm()
        R = quaternion_to_matrix(q.unsqueeze(0))[0]
        v = v @ R.transpose(0, 1)
        v = v + self.translation.unsqueeze(0)
        return v