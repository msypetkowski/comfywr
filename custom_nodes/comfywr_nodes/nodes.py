import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


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

    CATEGORY = "comfywr_nodes/utils"

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
