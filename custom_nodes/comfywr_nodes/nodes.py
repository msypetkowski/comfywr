import os

import cv2
import numpy as np
import torch
import trimesh
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


def align_images(source_img, target_img, x0):
    """
    Uses differential evolution to find the affine transformation (scaling and translation)
    that minimizes the MSE between the transformed source image and the target image.
    Returns the transformed image and the optimization result.
    """

    # assume the first column of pixels defines padding/background color
    padding_color = np.median(target_img[:, 0], axis=0)

    # Initially rescale source image for optimization
    ratio = source_img.shape[0] / target_img.shape[1]
    source_img = cv2.resize(source_img, (round(source_img.shape[1] / ratio), round(source_img.shape[0] / ratio)))

    # Get dimensions of the images
    h_source, w_source = source_img.shape[:2]
    h_target, w_target = target_img.shape[:2]

    # Set bounds for the affine transformation parameters
    bounds = [
        (0.5, 2.0),  # s (scaling in x and y)
        (-0.5, 0.5),  # t_x (translation in x)
        (-0.5, 0.5),  # t_y (translation in y)
    ]

    # Perform the optimization
    result = differential_evolution(
        alignment_objective_function,
        bounds,
        x0=x0,
        args=(source_img, target_img, padding_color),
        strategy='best1bin',
        maxiter=100,
        popsize=5,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=True
    )

    # Extract the optimal parameters
    s, t_x, t_y = result.x
    t_x *= w_target
    t_y *= h_target

    # Build the affine transformation matrix with the optimal parameters
    M = np.array([[s, 0, t_x],
                  [0, s, t_y]], dtype=np.float32)

    # Apply the optimal affine transformation
    transformed_img = cv2.warpAffine(source_img, M, (w_target, h_target),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=padding_color.tolist())

    return transformed_img, result


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
    )
    RETURN_NAMES = (
        "output_mesh_file_path",
        "visualization",
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
        return (output_mesh_file_path, vis)


def mesh_silhouette_images(mesh):
    """
    WARNING: AI generated

    Generates three silhouette images of a 3D mesh projected onto the XY, YZ, and XZ planes.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh, assumed to be within bounds [-1, 1] along each axis.

    Returns:
    tuple: Three 1024x1024 binary numpy arrays representing the silhouettes on the XY, YZ, and XZ planes.
    """
    mesh = mesh.copy()

    # Get vertices and faces from the mesh
    vertices = mesh.vertices  # shape (n_vertices, 3)
    faces = mesh.faces  # shape (n_faces, 3)
    triangles = vertices[faces]  # shape (n_faces, 3, 3)

    # Prepare images and drawing contexts for each projection
    img_xy = Image.new('1', (1024, 1024), 0)
    draw_xy = ImageDraw.Draw(img_xy)

    img_yz = Image.new('1', (1024, 1024), 0)
    draw_yz = ImageDraw.Draw(img_yz)

    # img_xz = Image.new('1', (1024, 1024), 0)
    # draw_xz = ImageDraw.Draw(img_xz)

    # Mapping functions from coordinate space [-1, 1] to pixel space [0, 1023]
    def coord_to_pixel(coord):
        # return ((coord + 1.0) * 511.5).round().astype(int)
        # assert ((-1.0 <= coord) & (coord <= 1.0)).all(), coord
        coord = (coord + 1) / 2
        return (coord * 1023).round().astype(int)

    def coord_to_pixel_flipped(coord):
        # return ((1.0 - coord) * 511.5).round().astype(int)
        # assert ((-1.0 <= coord) & (coord <= 1.0)).all(), coord
        coord = (coord + 1) / 2
        return ((1 - coord) * 1023).round().astype(int)

    # Loop over each triangle to project and draw on the images

    # assert ((-1.0 <= triangles) & (triangles <= 1.0)).all()
    # assert ((-0.5 >= triangles) | (triangles >= 0.5)).any()

    for tri in triangles:
        # XY projection (view along +Z direction)
        x = tri[:, 0]
        y = tri[:, 1]
        px = coord_to_pixel(x)
        py = coord_to_pixel_flipped(y)
        points = list(zip(px, py))
        draw_xy.polygon(points, fill=1)

        # YZ projection (view along +X direction)
        y = tri[:, 1]
        z = tri[:, 2]
        px = coord_to_pixel_flipped(z)
        py = coord_to_pixel_flipped(y)
        points = list(zip(px, py))
        draw_yz.polygon(points, fill=1)

        # XZ projection (view along +Y direction)
        # x = tri[:, 0]
        # z = tri[:, 2]
        # px = coord_to_pixel(x)
        # py = coord_to_pixel_flipped(z)
        # points = list(zip(px, py))
        # draw_xz.polygon(points, fill=1)

    # Convert images to numpy arrays and return
    img_xy_array = np.array(img_xy)
    img_yz_array = np.array(img_yz)
    # img_xz_array = np.array(img_xz)

    return img_xy_array, img_yz_array


def transform_mesh(mesh, x_offset, y_offset, z_offset, x_scale, y_scale, z_scale):
    pivot_point = (-1.0, 1.0, 1.0)
    # pivot_point = (0, 0, 0)
    if isinstance(mesh, trimesh.Trimesh):
        pivot_matrix = np.eye(4)
        pivot_matrix[0, 3] = pivot_point[0]
        pivot_matrix[1, 3] = pivot_point[1]
        pivot_matrix[2, 3] = pivot_point[2]

        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = x_scale
        scale_matrix[1, 1] = y_scale
        scale_matrix[2, 2] = z_scale

        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = x_offset
        translation_matrix[1, 3] = y_offset
        translation_matrix[2, 3] = z_offset

        transformation_matrix = pivot_matrix @ translation_matrix @ scale_matrix @ np.linalg.inv(pivot_matrix)

        transformed_mesh = mesh
        transformed_mesh.apply_transform(transformation_matrix)
    else:
        transformed_mesh = mesh
        verts = transformed_mesh.v

        verts[:, 0] -= pivot_point[0]
        verts[:, 1] -= pivot_point[1]
        verts[:, 2] -= pivot_point[2]

        verts[:, 0] *= x_scale
        verts[:, 1] *= y_scale
        verts[:, 2] *= z_scale

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset
        verts[:, 2] += z_offset

        verts[:, 0] += pivot_point[0]
        verts[:, 1] += pivot_point[1]
        verts[:, 2] += pivot_point[2]

    return transformed_mesh


def visualize_silhouettes(silhouettes):
    vis = np.zeros((silhouettes[0][0].shape[0], silhouettes[1][1].shape[1] * 2, 3), dtype=np.uint8)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 100, 100)]
    for s, col in zip(silhouettes, colors):
        assert 0 < np.mean(s) < 1
        mask = np.concatenate(s, 1).astype(np.uint8) * 255
        assert mask.shape == vis.shape[:2]
        edges = cv2.dilate(mask, np.ones((3, 3))) - cv2.erode(mask, np.ones((3, 3)))
        vis[edges > 0] = col
    return vis
