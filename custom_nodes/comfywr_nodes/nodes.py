from copy import copy
import os

from .utils.mesh_utils import mesh_copy, mesh_silhouette_images, visualize_silhouettes
from .utils.mesh_transformations import apply_transformation, compute_transformation_matrix

from .utils.building_mesh import simple_building_mesh
import cv2
import numpy as np
import torch
import trimesh
import importlib
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import differential_evolution
from skimage.metrics import mean_squared_error

import folder_paths as comfy_paths


def put_text(img, text, coords, text_color, font_size):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("arial.ttf", font_size)

    text_x = coords[0]  # - text_width // 2
    text_y = coords[1] + font_size // 2
    draw.text((text_x, text_y), text, fill=tuple(list(text_color)), font=font)
    return np.array(img_pil)


def get_img_info_str(img):
    n_channels = img.shape[3]
    assert n_channels in (1, 3, 4)
    ret = []
    if n_channels in (3, 4):
        rgb = img[:, :, :, :3]
        ret.extend([
            f'{rgb.shape=}',
            f'{rgb.dtype=}',
            f'{list(np.min(rgb, axis=(0, 1, 2)))=}',
            f'{list(np.max(rgb, axis=(0, 1, 2)))=}',
            f'{list(np.mean(rgb, axis=(0, 1, 2)))=}',
            f'{list(np.min(rgb, axis=(1, 2, 3)))=}',
            f'{list(np.max(rgb, axis=(1, 2, 3)))=}',
            f'{list(np.mean(rgb, axis=(1, 2, 3)))=}',
        ])
    if n_channels in (1, 4):
        alpha = img[:, :, :, -1:]
        ret.extend([
            f'{alpha.shape=}',
            f'{alpha.dtype=}',
            f'{np.min(alpha)=}',
            f'{np.max(alpha)=}',
            f'{np.mean(alpha)=}',
            f'{np.min(alpha, axis=(1, 2, 3))=}',
            f'{np.max(alpha, axis=(1, 2, 3))=}',
            f'{np.mean(alpha, axis=(1, 2, 3))=}',
        ])
    return ret


class ImageBatchDiagnoser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",),
                         "font_size": ("INT", {"default": 24, "min": 1, "max": 256, "step": 1}),
                         },
        }

    # RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("Info Image", "Font Size")

    FUNCTION = "diagnose"

    OUTPUT_NODE = True

    CATEGORY = "comfywr_nodes"

    def diagnose(self, image, font_size):
        assert len(image.shape) == 4
        image = image.cpu().numpy()
        info_list = get_img_info_str(image)
        vis = np.concatenate(image, axis=1)
        vis -= np.min(vis)
        vis = (255 * (vis / np.max(vis))).astype(np.uint8)
        for i, info in enumerate(info_list):
            vis = put_text(vis, info, (0, (font_size + round(font_size * 0.15)) * i), (255, 0, 0), font_size)
        return (torch.tensor(vis.astype(np.float32) / 255).unsqueeze(0).cuda(),)
        # return (image,)


def alignment_objective_function(params, source_img, target_img, padding_color):
    """
    Objective function for optimization.
    Applies an affine transformation (scaling and translation only) to the source image
    and computes the MSE with the target image.
    Assumes padding color is always white (255, 255, 255).
    """
    # Extract affine parameters from params
    s, t_x, t_y = params
    h_target, w_target = target_img.shape[:2]
    t_x *= w_target
    t_y *= h_target

    # Build the affine transformation matrix (no shear or rotation)
    M = np.array([[s, 0, t_x],
                  [0, s, t_y]], dtype=np.float32)

    # Apply the affine transformation to the source image
    transformed_img = cv2.warpAffine(source_img, M, (w_target, h_target),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=padding_color.tolist())

    # Compute the Mean Squared Error between the transformed source image and the target image
    mse = mean_squared_error(target_img, transformed_img)

    return mse

def scale_image_by(img, ratio):
    return cv2.resize(img, (round(img.shape[1] * ratio), round(img.shape[0] * ratio)))

def align_images(source_img, target_img, x0, downscale_for_optim=0.3):
    """
    Uses differential evolution to find the affine transformation (scaling and translation)
    that minimizes the MSE between the transformed source image and the target image.
    Returns the transformed image and the optimization result.
    """

    result, _ = _align_images(source_img, target_img, x0, downscale_for_optim / 5, 64)
    result, transformed_img = _align_images(source_img, target_img, result.x, downscale_for_optim, 5)

    return transformed_img, result


def _align_images(source_img, target_img, x0, downscale_for_optim, popsize):
    # assume the first column of pixels defines padding/background color
    padding_color = np.median(target_img[:, 0], axis=0)

    source_img_orig = source_img
    h_target_orig, w_target_orig = target_img.shape[:2]

    # Initially rescale source image for optimization
    initial_downscale_ratio = target_img.shape[1] / source_img.shape[0]
    source_img = scale_image_by(source_img, initial_downscale_ratio)

    # Downscale source and target image for optimization
    source_img = scale_image_by(source_img, downscale_for_optim)
    target_img = scale_image_by(target_img, downscale_for_optim)
    # Set bounds for the affine transformation parameters
    bounds = [
        (0.4, 2.2),  # s (scaling in x and y)
        (-0.4, 0.6),  # t_x (translation in x)
        (-0.4, 0.6),  # t_y (translation in y)
    ]
    # Perform the optimization
    result = differential_evolution(
        alignment_objective_function,
        bounds,
        x0=x0,
        args=(source_img, target_img, padding_color),
        strategy='best1bin',
        maxiter=100,
        popsize=popsize,
        tol=0.004,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=True
    )
    # Extract the optimal parameters
    s, t_x, t_y = result.x
    s = s * initial_downscale_ratio
    print('Found transform:', s, t_x, t_y)
    t_x *= w_target_orig
    t_y *= h_target_orig
    # Build the affine transformation matrix with the optimal parameters
    M = np.array([[s, 0, t_x],
                  [0, s, t_y]], dtype=np.float32)
    # Apply the optimal affine transformation
    transformed_img = cv2.warpAffine(source_img_orig, M, (w_target_orig, h_target_orig),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=padding_color.tolist())
    return result, transformed_img


class ImageAligner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"source": ("IMAGE",),
                         "target": ("IMAGE",),
                         "scale_guess": ("FLOAT", {"default": 1.0}),
                         "x_offset_guess": ("FLOAT", {"default": 0.,"min": -1.5, "max": 1.5, "step": 0.01}),
                         "y_offset_guess": ("FLOAT", {"default": 0.,"min": -1.5, "max": 1.5, "step": 0.01}),
                         },
        }

    # RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Aligned",)

    FUNCTION = "align"

    OUTPUT_NODE = False

    CATEGORY = "comfywr_nodes"

    def align(self, source, target, scale_guess, x_offset_guess, y_offset_guess):
        assert len(source.shape) == 4
        assert len(target.shape) == 4
        x0 = [scale_guess, x_offset_guess, y_offset_guess]

        result = []
        for src, trgt in zip(source, target):
            src = src.cpu().numpy()
            trgt = trgt.cpu().numpy()
            result.append(align_images(src, trgt, x0)[0])
            assert result[-1].shape == trgt.shape

        return (torch.tensor(np.stack(result).astype(np.float32)).to(target.device),)


class AlignMeshToMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "output_mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "masks": ("IMAGE",),
                "scale_guess": ("FLOAT", {"default": 1.0}),
                "x_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
                "y_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
                "z_offset_guess": ("FLOAT", {"default": 0., "min": -1.5, "max": 1.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "IMAGE",
        "TRANSFORM",
    )
    RETURN_NAMES = (
        "output_mesh_file_path",
        "visualization",
        "transformation",
    )
    FUNCTION = "align_mesh"
    CATEGORY = "comfywr_nodes"

    def align_mesh(self, input_mesh_file_path, output_mesh_file_path, masks,
                   scale_guess, x_offset_guess, y_offset_guess, z_offset_guess):
        if not os.path.isabs(input_mesh_file_path):
            input_mesh_file_path = os.path.join(comfy_paths.input_directory, input_mesh_file_path)
        if not os.path.isabs(output_mesh_file_path):
            output_mesh_file_path = os.path.join(comfy_paths.output_directory, output_mesh_file_path)

        if os.path.exists(input_mesh_file_path):
            import importlib
            Mesh = importlib.import_module('custom_nodes.ComfyUI-3D-Pack.mesh_processer.mesh').Mesh
            mesh = Mesh.load(input_mesh_file_path, resize=False)
            trimesh_mesh = trimesh.load(input_mesh_file_path)
        else:
            print(f"[{self.__class__.__name__}] File {input_mesh_file_path} does not exist")

        original_mesh_silh = mesh_silhouette_images(trimesh_mesh)

        target_silh = [
            masks[0].cpu().numpy().astype(np.uint8)[:, :, 0] * 255,
            masks[1].cpu().numpy().astype(np.uint8)[:, :, 0] * 255,
        ]
        for i in range(2):
            target_silh[i] = cv2.resize(target_silh[i], (1024, 1024)) > 0

        x0 = [scale_guess, x_offset_guess, y_offset_guess]
        aligned1, params1 = align_images(original_mesh_silh[0].astype(np.uint8) * 255,
                                         target_silh[0].astype(np.uint8) * 255, x0)
        x0 = [scale_guess, z_offset_guess, y_offset_guess]
        aligned2, params2 = align_images(original_mesh_silh[1].astype(np.uint8) * 255,
                                         target_silh[1].astype(np.uint8) * 255, x0)
        print('Params:', params1.x, params2.x)

        # scale = (params1.x[0] + params2.x[0]) / 2
        scale_xy = params1.x[0]
        scale_z = params2.x[0]
        scale = [scale_xy] * 2 + [scale_z]
        offset_x = params1.x[1] * 2
        # offset_y = (params1.x[2] + params2.x[2]) / 2
        offset_y = -params1.x[2] * 2
        offset_z = -params2.x[1] * 2

        aligned_mesh = transform_mesh(mesh, offset_x, offset_y, offset_z, *scale)
        aligned_mesh.write(output_mesh_file_path)

        aligned_trimesh = transform_mesh(trimesh_mesh, offset_x, offset_y, offset_z, *scale)
        aligned_mesh_slih = mesh_silhouette_images(aligned_trimesh)
        aligned_silh = [aligned1 > 0, aligned2 > 0]
        vis = visualize_silhouettes([target_silh, original_mesh_silh, aligned_silh, aligned_mesh_slih])

        vis = torch.tensor(vis.astype(np.float32) / 255).unsqueeze(0).cuda()

        transformation_matrix = compute_transformation_matrix((offset_x, offset_y, offset_z), tuple(scale), pivot_point=(-1.0, 1.0, 1.0))
        transformation_matrix = torch.from_numpy(transformation_matrix)

        return (output_mesh_file_path, vis, transformation_matrix)


def transform_mesh(mesh, x_offset, y_offset, z_offset, x_scale, y_scale, z_scale):
    pivot_point = (-1.0, 1.0, 1.0)
    translation = (x_offset, y_offset, z_offset)
    scale = (x_scale, y_scale, z_scale)

    transformation_matrix = compute_transformation_matrix(translation, scale, pivot_point=pivot_point)
    transformed_mesh = apply_transformation(mesh, transformation_matrix)
    
    return transformed_mesh


class CreateSimpleBuildingMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "height": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "depth": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "roof_type": (
                    ["none", "pyramid", "slanted_x", "slanted_z"],
                ),
                "roof_height": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "texture_res": ("INT", {"default": 512, "min": 4, "max": 4096, "step": 4}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("output_mesh",)
    FUNCTION = "create"
    CATEGORY = "comfywr_nodes"

    def create(self, width, height, depth, roof_type, roof_height, texture_res):

        # White texture
        building = simple_building_mesh(width, height, depth, roof_type, roof_height)

        # Convert to 3D-pack Mesh
        Mesh = importlib.import_module('custom_nodes.ComfyUI-3D-Pack.mesh_processer.mesh').Mesh
        mesh = Mesh.load_trimesh(given_mesh=building)
        mesh.auto_normal()
        mesh.auto_uv()
        mesh.set_new_albedo(texture_res, texture_res)

        return (mesh,)



class NormalizeMeshBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh": ("MESH",),
                "margin": ("FLOAT", {"default": 0.05, "min": 0, "max": 5.0, "step": 0.01}),
                "x_min": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "x_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "y_min": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "y_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "z_min": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "z_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MESH", "TRANSFORM")
    RETURN_NAMES = ("output_mesh", "transform_matrix")
    FUNCTION = "normalize_mesh"
    CATEGORY = "comfywr_nodes"

    def normalize_mesh(self, input_mesh, margin, x_min, x_max, y_min, y_max, z_min, z_max, keep_aspect_ratio):
        """ Normalize input mesh to fit inside given bbox coordinates with a given absolute margin """
        verts = input_mesh.v.cpu().numpy()  # Nx3 tensor

        mesh_min = verts.min(axis=0)
        mesh_max = verts.max(axis=0)
        mesh_center = (mesh_min + mesh_max) / 2
        mesh_size = mesh_max - mesh_min

        target_size = np.array([
            (x_max - x_min) - 2 * margin,
            (y_max - y_min) - 2 * margin,
            (z_max - z_min) - 2 * margin
        ])

        if np.any(target_size <= 0):
            raise ValueError("Target size must be positive after accounting for margin")

        if keep_aspect_ratio:
            scale_factor = np.min(target_size / mesh_size)
            scale_factors = np.array([scale_factor, scale_factor, scale_factor])
        else:
            scale_factors = target_size / mesh_size

        # Compute target center
        target_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2,
        ])

        # Build the transformation: scale about mesh_center, then translate to target_center
        transform_matrix = compute_transformation_matrix(
            translation=target_center,
            scale=scale_factors,
        )

        transform_matrix = transform_matrix @ compute_transformation_matrix(translation=-mesh_center) 

        # verts = (verts - mesh_center) * scale_factors + target_center

        # Apply transform in-place
        output_mesh = apply_transformation(mesh_copy(input_mesh), transform_matrix)

        # input_mesh.v[...] = torch.from_numpy(verts)

        # sanity check
        assert (output_mesh.v.amin(axis=0).cpu().numpy() >= np.array([x_min, y_min, z_min])).all()
        assert (output_mesh.v.amax(axis=0).cpu().numpy() <= np.array([x_max, y_max, z_max])).all()

        return (output_mesh, torch.from_numpy(transform_matrix))

class ApplyMeshTransform:
    """
    ComfyUI node: apply a 4x4 transformation (TRANSFORM) to a mesh.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh": ("MESH",),
                "transform_matrix": ("TRANSFORM",),
                "inverse": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("output_mesh",)
    FUNCTION = "apply_transform_node"
    CATEGORY = "comfywr_nodes"

    def apply_transform_node(self, input_mesh, transform_matrix, inverse):
        """
        Apply the given 4x4 transform to the input mesh and return the mesh.

        Args:
            input_mesh: mesh with .v numpy/torch vertices or trimesh.Trimesh
            transform_matrix: 4x4 torch tensor or numpy array
        Returns:
            Transformed mesh
        """
        # Convert torch tensor to numpy
        if isinstance(transform_matrix, torch.Tensor):
            matrix = transform_matrix.cpu().numpy()
        else:
            matrix = np.array(transform_matrix)

        if inverse:
            matrix = np.linalg.inv(matrix)

        # Use utility to apply transform
        output_mesh = apply_transformation(mesh_copy(input_mesh), matrix)

        return (output_mesh, )

