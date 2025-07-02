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
        self.scale = nn.Parameter(scale.view(-1,3), requires_grad=not lock_scale)
        self.rotation = nn.Parameter(rotation.view(-1,4), requires_grad=not lock_rotation)
        self.translation = nn.Parameter(translation.view(-1, 3), requires_grad=not lock_translation)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        v = verts * self.scale
        q = self.rotation / self.rotation.norm()
        R = quaternion_to_matrix(q)[0]
        v = v @ R.transpose(1, 0)
        v = v + self.translation
        return v

    def clone(self) -> "Transform":
        # 1) instantiate fresh via default init
        new = self.__class__()
        # 2) load tensor data for params and buffers
        new.load_state_dict(self.state_dict())
        # 3) restore requires_grad flags
        for (orig_name, orig_p) in self.named_parameters():
            new_p = dict(new.named_parameters())[orig_name]
            new_p.requires_grad = orig_p.requires_grad
        return new
        