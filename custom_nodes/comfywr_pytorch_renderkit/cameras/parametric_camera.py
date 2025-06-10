from typing import Tuple, Union
import torch.nn as nn
import torch
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform

class ParametricCamera(nn.Module):
    """
    FoV camera with parameterized field of view (FOV), distance, elevation, and azimuth.
    Internally stores normalized parameters in [0,1] and denormalizes using ranges.
    """
    def __init__(
        self,
        fov: float = 60.0,
        dist: float = 2.7,
        elev: float = 0.0,
        azim: float = 0.0,
        offset: Union[torch.Tensor, Tuple] = (0.0,0.0,0.0),
        fov_range: Tuple = (1.0, 179.0),
        dist_range: Tuple = (0.1, 10.0),
        elev_range: Tuple = (-90.0, 90.0),
        azim_range: Tuple = (0.0, 360.0),
        lock_fov: bool = False,
        lock_distance: bool = False,
        lock_angles: bool = False,
        lock_offset: bool =False,
    ):
        super().__init__()
        # Store ranges
        self.fov_min, self.fov_max = fov_range
        self.dist_min, self.dist_max = dist_range
        self.elev_min, self.elev_max = elev_range
        self.azim_min, self.azim_max = azim_range
        # Normalize and register parameters
        norm = lambda v, mn, mx: (v - mn) / (mx - mn)
        self.fov_norm = nn.Parameter(torch.tensor(norm(fov, self.fov_min, self.fov_max)).view(-1,1).float(), requires_grad=not lock_fov)
        self.dist_norm = nn.Parameter(torch.tensor(norm(dist, self.dist_min, self.dist_max)).view(-1,1).float(), requires_grad=not lock_distance)
        self.elev_norm = nn.Parameter(torch.tensor(norm(elev, self.elev_min, self.elev_max)).view(-1,1).float(), requires_grad=not lock_angles)
        self.azim_norm = nn.Parameter(torch.tensor(norm(azim, self.azim_min, self.azim_max)).view(-1,1).float(), requires_grad=not lock_angles)
        self.offset = nn.Parameter(torch.tensor(offset).view(-1,3).float(), requires_grad=not lock_offset)

    @staticmethod
    def denorm(norm, nmin, nmax):
        return nmin + norm * (nmax - nmin)

    def forward(self, device: torch.device = None) -> FoVPerspectiveCameras:
        # Clamp
        fov_n = self.fov_norm.clamp(0, 1)
        dist_n = self.dist_norm.clamp(0, 1)
        elev_n = self.elev_norm.clamp(0, 1)
        azim_n = self.azim_norm.clamp(0, 1)
        # Denormalize
        fov = self.denorm(fov_n, self.fov_min, self.fov_max)
        dist = self.denorm(dist_n, self.dist_min, self.dist_max)
        elev = self.denorm(elev_n, self.elev_min, self.elev_max)
        azim = self.denorm(azim_n, self.azim_min, self.azim_max)
        # Build camera
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=self.offset, device=device)
        return FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    @property
    def fov(self):
        return self.denorm(self.fov_norm, self.fov_min, self.fov_max).detach()
    
    @property
    def dist(self):
        return self.denorm(self.dist_norm, self.dist_min, self.dist_max).detach()

    @property
    def elev(self):
        return self.denorm(self.elev_norm, self.elev_min, self.elev_max).detach()

    @property
    def azim(self):
        return self.denorm(self.azim_norm, self.azim_min, self.azim_max).detach()
