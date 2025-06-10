import torch
import numpy as np
import cv2
from scipy.optimize import differential_evolution

from .utils.mesh_utils import mesh_copy, mesh_to_trimesh, mesh_silhouette_images, visualize_silhouettes
from .utils.mesh_transformations import Direction, direction_matrix, compute_transformation_matrix, apply_transformation

def minimize_iou_fn(img, tgt):
    i_mask, t_mask = img > 0, tgt > 0
    intersection = (i_mask & t_mask).sum()
    union = (i_mask | t_mask).sum()
    return 1 - (intersection / union)

def mse_fn(img, tgt):
    return np.mean((img - tgt)**2)

def alignment_objective(params,
                        src: np.ndarray,
                        tgt: np.ndarray,
                        metric_fn,
                        keep_aspect_ratio = False,
                        padding_color=(0, 0, 0)) -> float:
    """
    Joint objective that applies a 3D-like affine transform to front and side silhouettes
    and computes a combined loss via a 4-input metric_fn.

    Parameters:
        params: [s_x, s_y, s_z, t_x, t_y, t_z]
            s_xy: uniform scale in x and y for front view
            s_z: depth scale affecting side view scale in x (mesh Z axis)
            t_x, t_y: pixel translation for both views (normalized to [0,1] fractions)
            t_z: depth translation mapping to horizontal translation in side view
        src_front: source front silhouette image (HxW)
        src_side: source side silhouette image (HxW)
        tgt_front: target front mask (HxW)
        tgt_side: target side mask (HxW)
        metric_fn: function(front_img, side_img, tgt_front, tgt_side) -> float
        padding_color: RGB tuple for border padding

    Returns:
        Combined loss from metric_fn(front_trans, side_trans, tgt_front, tgt_side).
    """
    # Unpack parameters
    if keep_aspect_ratio:
        s, t_x_frac, t_y_frac, t_z_frac = params
        s_x, s_y, s_z = s, s, s
    else:
        s_x, s_y, s_z, t_x_frac, t_y_frac, t_z_frac = params
    # Image dims
    h, w = tgt.shape[1:]

    # Compute pixel translations
    t_x = t_x_frac * w
    t_y = t_y_frac * h
    t_z = t_z_frac * w  # side horizontal shift
    
    # Calculate scale adjustments for centered scaling
    sa_x = (s_x - 1) * (w/2)
    sa_y = (s_y - 1) * (h/2)
    sa_z = (s_z - 1) * (w/2)

    trans = src.copy()
    # Affine for front view: scale s_xy, translate t_x, t_y
    M_front = np.array([[s_x, 0, t_x - sa_x],[0, s_y, t_y - sa_y]], dtype=np.float32)
    cv2.warpAffine(src[0], M_front, (w, h), dst=trans[0], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_color)

    # Affine for side view: scale by s_z on x-axis, and s_xy on y-axis (vertical preserved)
    M_side = np.array([[s_z, 0, t_z - sa_z],[0, s_y, t_y - sa_y]], dtype=np.float32)
    cv2.warpAffine(src[1], M_side, (w, h), dst=trans[1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_color)

    # Compute combined metric
    return metric_fn(trans, tgt)

def align_front_and_side_silhuettes(start_params,
                    src_silhuettes: np.ndarray,
                    tgt_silhuettes: np.ndarray,
                    metric_fn, downscale_for_optim=1.0,
                    keep_aspect_ratio = False,
                    **kwargs):

    h, w = tgt_silhuettes.shape[1:]
    h, w = int(h*downscale_for_optim), int(w*downscale_for_optim)
    src = np.stack([cv2.resize(silh, (w, h)) for silh in src_silhuettes])
    tgt = np.stack([cv2.resize(silh, (w, h)) for silh in tgt_silhuettes])

    n_scales = 1 if keep_aspect_ratio else 3
    bounds = [(0.5, 2.0)]*n_scales  + [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    result = differential_evolution(
                alignment_objective,
                bounds,
                x0=start_params,
                args=(src, tgt, metric_fn, keep_aspect_ratio),
                strategy='best1bin',
                **kwargs,
                mutation=(0.5, 1),
                recombination=0.7,
                polish=True,
                disp=True
            )

    return result.x, result.fun



LOSS_FN = {"MSE": mse_fn, "IoU": minimize_iou_fn}


class AlignMeshToMutliviewMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "masks": ("IMAGE",),
                "scale_guess": ("FLOAT", {"default": 1.0}),
                "x_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
                "y_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
                "z_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
                "loss_type": (list(LOSS_FN.keys()),),
                "keep_aspect_ratio": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH", "IMAGE", "TRANSFORM", )
    RETURN_NAMES = ("aligned_mesh", "vizualization", "transformation",)
    FUNCTION = "align_four"
    CATEGORY = "Mesh/Align"

    def align_four(self, mesh,
                   masks,
                   scale_guess,
                   x_offset_guess, y_offset_guess, z_offset_guess,
                   loss_type,
                   keep_aspect_ratio = False):

        # Load mesh & render silhouettes once
        trimesh = mesh_to_trimesh(mesh)
        silhouette_front, silhouette_side = mesh_silhouette_images(trimesh)
        silhouette_front = silhouette_front.astype(np.uint8)
        silhouette_side = silhouette_side.astype(np.uint8)

        # Prepare target masks
        direction_masks = {k: mask.cpu().numpy().astype(np.uint8)[:, :, 0] for k, mask in zip(Direction, masks.gt(0))}
        direction_masks = {k: cv2.resize(mask, (1024, 1024)) for k, mask in direction_masks.items()}

        # Choose metric function expecting 4 inputs
        metric_fn = LOSS_FN[loss_type]

        # Orientation pairs: front=mask_N, side variants
        orientation_pairs = ((Direction.N,Direction.E),
                             (Direction.E,Direction.S),
                             (Direction.S,Direction.W),
                             (Direction.W,Direction.N))
        orientation_pairs = tuple((f,s) for f,s in orientation_pairs if f in direction_masks and s in direction_masks)
        assert len(orientation_pairs) >= 1, "Not enough masks provided"
        best = {"loss": np.inf, "params": None, "orient": None}
        
        src = np.stack([cv2.resize(m, (1024, 1024)) for m in (silhouette_front, silhouette_side)])

        n_scales = 1 if keep_aspect_ratio else 3

        x0 = [scale_guess] * n_scales + [x_offset_guess, y_offset_guess, z_offset_guess]


        for front_key, side_key in orientation_pairs:
            # Perform preliminary optimization for each mesh orientation
            tgt_front = direction_masks[front_key]
            tgt_side  = direction_masks[side_key]
            tgt = np.stack((tgt_front,tgt_side))

            # Perform the optimization
            x, fun = align_front_and_side_silhuettes(x0, src, tgt,
                                                    metric_fn=metric_fn,
                                                    downscale_for_optim=0.0625,
                                                    keep_aspect_ratio=keep_aspect_ratio,
                                                    maxiter=20,
                                                    popsize=64,
                                                    tol=0.01)

            # Update best optimization performance
            if fun < best["loss"]:
                best.update({"loss": fun,
                             "params": x,
                             "orient": (front_key, side_key)})

        front_key, side_key = best["orient"]
        tgt_front = direction_masks[front_key]
        tgt_side  = direction_masks[side_key]
        tgt = np.stack((tgt_front,tgt_side))

        x0 = best["params"]

        # Perform the optimization on rotated mesh
        x, fun = align_front_and_side_silhuettes(x0, src, tgt,
                                                    metric_fn=metric_fn,
                                                    downscale_for_optim=0.25,
                                                    keep_aspect_ratio=keep_aspect_ratio,
                                                    maxiter=64,
                                                    popsize=32,
                                                    tol=0.01)
        x, fun = align_front_and_side_silhuettes(x, src, tgt,
                                                    metric_fn=metric_fn,
                                                    downscale_for_optim=1.0,
                                                    keep_aspect_ratio=keep_aspect_ratio,
                                                    maxiter=64,
                                                    popsize=8,
                                                    tol=0.004)
        
        best.update({"loss": fun, "params": x, "orient": (front_key, side_key)})

        # Apply best transform to mesh (uniform Z-scale into depth)
        if keep_aspect_ratio:
            scale, offset_x, offset_y, offset_z = best['params']
            scale_x, scale_y, scale_z = scale, scale, scale
        else:
            scale_x, scale_y, scale_z, offset_x, offset_y, offset_z = best['params']
        offset = (offset_x * 2, -offset_y * 2, -offset_z * 2)
        scale = (scale_x, scale_y, scale_z)

        direction = best["orient"][0]
        rotation_matrix = direction_matrix(direction)
        
        transformation_matrix = rotation_matrix @ compute_transformation_matrix(offset, scale)
        output_mesh = apply_transformation(mesh_copy(mesh), transformation_matrix)

        target_silh = (direction_masks[Direction.N], direction_masks[Direction.E])
        aligned_silh = mesh_silhouette_images(mesh_to_trimesh(output_mesh))


        vis = visualize_silhouettes([target_silh, aligned_silh])
        vis = torch.tensor(vis.astype(np.float32) / 255).unsqueeze(0)

        return (output_mesh, vis, torch.from_numpy(transformation_matrix), )
