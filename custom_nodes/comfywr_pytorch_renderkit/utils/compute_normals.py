import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def ensure_batch(image_tensor):
    """Ensure tensor is 4D: (B, C, H, W)"""
    if image_tensor.ndim == 3:
        return image_tensor.unsqueeze(0)
    return image_tensor


def sobel_filter(image_tensor):
    """
    Computes Sobel gradients in x and y directions using a single convolution.
    Returns: grad_x, grad_y (B, C, H, W)
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # Define Sobel kernels for x and y (2, 1, 3, 3)
    sobel_kernel = torch.tensor([
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],  # dx
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]   # dy
    ], dtype=torch.float32, device=device).unsqueeze(1) / 8.0

    # Repeat for each channel
    kernel = sobel_kernel.repeat(C, 1, 1, 1)  # (2*C, 1, 3, 3)
    input_reshaped = image_tensor.view(B * C, 1, H, W)
    out = F.conv2d(input_reshaped, kernel, padding=1, groups=1)  # (B*C, 2, H, W)
    out = out.view(B, C, 2, H, W)

    grad_x = out[:, :, 0, :, :]
    grad_y = out[:, :, 1, :, :]
    return grad_x, grad_y


def compute_normal_map(depth: torch.Tensor, alpha: torch.Tensor,
                        blur_kernel_size=5, blur_sigma=1.0):
    # Ensure batch dimensions
    depth = ensure_batch(depth)
    alpha = ensure_batch(alpha)

    B, _, H, W = depth.shape

    # Blur depth and alpha using torchvision
    blurred_depth = gaussian_blur(depth, kernel_size=blur_kernel_size, sigma=blur_sigma)
    # blurred_alpha = gaussian_blur(alpha, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # Compute gradients
    dzdx, dzdy = sobel_filter(blurred_depth)

    # Construct normal components
    nx = -dzdx
    ny = -dzdy
    nz = torch.ones_like(dzdx) / max(W,H)

    # Stack normals and normalize
    normal = torch.cat([nx, ny, nz], dim=1)  # (B, 3, H, W)
    normal = F.normalize(normal, dim=1)

    # Apply alpha mask directly
    normal = normal * (alpha > 0.5).float()

    return normal

def normalize_depth_within_alpha(depth: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize depth values to [0, 1] based only on pixels where alpha > 0.
    """
    mask = (alpha > 0)
    if not mask.any():
        return depth

    masked_depth = depth[mask]
    dmin, dmax = masked_depth.amin(), masked_depth.amax()
    normalized = (depth - dmin) / (dmax - dmin + eps)
    return normalized * mask.float()