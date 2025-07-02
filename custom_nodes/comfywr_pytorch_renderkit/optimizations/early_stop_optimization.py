import torch
import torch.nn as nn
from ..meshes.parametric_mesh import ParametricMesh
from ..cameras.parametric_camera import ParametricCamera
from pytorch3d.renderer import MeshRenderer
from tqdm.auto import tqdm
import copy

def optimize_mesh_to_image(
    mesh: ParametricMesh,                         # ParametricMesh instance
    camera: ParametricCamera,                     # ParametricCamera instance
    renderer: MeshRenderer,                       # PyTorch3D renderer
    target_image: torch.Tensor,               # Shape (1, H, W, 4), already permuted
    optimizer: torch.optim.Optimizer,
    iterations: int = 3000,
    alpha_weight: float = 1.0,
    early_stop_window: int = 50,
    early_stop_percent: float = 0.001,        # Percent threshold for stopping
    device = torch.device('cpu'),
    verbose = True,
):  
    mesh.to(device)
    channel_weights = torch.tensor([1, 1, 1, alpha_weight], dtype=torch.float32, device=device)

    criterion = nn.HuberLoss(reduction='mean')
    weighted_target = target_image.float().to(device) * channel_weights

    losses = []
    eps = 1e-8

    best_loss = float('inf')
    best_mesh_state = None
    best_camera_state = None

    for _ in tqdm(range(iterations), disable= (not verbose)):
        optimizer.zero_grad()

        cams = camera(device)
        rendered = renderer(meshes_world=mesh(), cameras=cams)
        rendered = rendered * channel_weights

        loss = criterion(rendered, weighted_target)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_mesh_state = copy.deepcopy(mesh.state_dict())
            best_camera_state = copy.deepcopy(camera.state_dict())

        if len(losses) >= early_stop_window:
            window_mean = sum(losses[-early_stop_window:]) / early_stop_window
            percent_change = abs(current_loss - window_mean) / (window_mean + eps)
            if percent_change < early_stop_percent:
                break

    # Restore best-performing states
    if best_mesh_state is not None:
        mesh.load_state_dict(best_mesh_state)
    if best_camera_state is not None:
        camera.load_state_dict(best_camera_state)

    with torch.no_grad():
        cams = camera(device)
        best_render = renderer(meshes_world=mesh(), cameras=cams)

    return mesh, camera, losses, best_render