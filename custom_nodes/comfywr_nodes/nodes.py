import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import differential_evolution
from skimage.metrics import mean_squared_error


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

def align_images(source_img, target_img):
    """
    Uses differential evolution to find the affine transformation (scaling and translation)
    that minimizes the MSE between the transformed source image and the target image.
    Returns the transformed image and the optimization result.
    """

    # assume the first column of pixels defines padding/background color
    padding_color = np.median(target_img[:, 0, :], axis=0)

    # Initially rescale source image for optimization
    ratio = source_img.shape[0] / target_img.shape[1]
    source_img = cv2.resize(source_img, (round(source_img.shape[1] / ratio), round(source_img.shape[0] / ratio)))

    # Get dimensions of the images
    h_source, w_source = source_img.shape[:2]
    h_target, w_target = target_img.shape[:2]

    # Set bounds for the affine transformation parameters
    bounds = [
        (0.5, 2.0),    # s (scaling in x and y)
        (-0.5, 0.5),   # t_x (translation in x)
        (-0.5, 0.5),   # t_y (translation in y)
    ]

    print([1.0, (w_target - w_source) / 2, 0])
    # Perform the optimization
    result = differential_evolution(
        alignment_objective_function,
        bounds,
        x0=[1.0, 0.25, 0],
        args=(source_img, target_img, padding_color),
        strategy='best1bin',
        maxiter=100,
        popsize=4,
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
                         "target": ("IMAGE",)},
        }

    # RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Aligned",)

    FUNCTION = "align"

    OUTPUT_NODE = False

    CATEGORY = "comfywr_nodes"

    def align(self, source, target):
        assert len(source.shape) == 4
        assert len(target.shape) == 4

        result = []
        for src, trgt in zip(source, target):
            src = src.cpu().numpy()
            trgt = trgt.cpu().numpy()
            result.append(align_images(src, trgt)[0])
            assert result[-1].shape == trgt.shape

        return (torch.tensor(np.stack(result).astype(np.float32)).to(target.device),)
