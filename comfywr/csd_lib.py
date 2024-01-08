import numpy as np
import torch
from comfy_extras.nodes_model_merging import ModelMergeSimple, CLIPMergeSimple
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
from nodes import init_custom_nodes, ControlNetLoader, CheckpointLoaderSimple, EmptyLatentImage, \
    CLIPTextEncode, LatentUpscale, LatentUpscaleBy, VAEDecode, VAEEncode, LoadImage, ImageScale, ImageScaleBy, \
    common_ksampler, CLIPSetLastLayer, LoraLoader, StyleModelLoader, CLIPVisionLoader, CLIPVisionEncode, \
    StyleModelApply

from custom_nodes.comfyui_controlnet_aux.node_wrappers.canny import Canny_Edge_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.leres import LERES_Depth_Map_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.lineart import LineArt_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.mediapipe_face import Media_Pipe_Face_Mesh_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.midas import MIDAS_Normal_Map_Preprocessor, \
    MIDAS_Depth_Map_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.scribble import Scribble_Preprocessor


def load_lora(model, clip, lora_name, strength_model, strength_clip):
    return LoraLoader().load_lora(model, clip, lora_name, strength_model, strength_clip)


def load_style_model(model_name):
    return StyleModelLoader().load_style_model(model_name)[0]


def load_clip_vision(model_name):
    return CLIPVisionLoader().load_clip(model_name)[0]


def clip_vision_encode(clip_vision, img):
    return CLIPVisionEncode().encode(clip_vision, img)[0]


def apply_style_model(clip_vision_output, style_model, conditioning):
    return StyleModelApply().apply_stylemodel(clip_vision_output, style_model, conditioning)[0]


def control_net_set_create(checkpoint, initial_hint_image, strength, start_percent=0, end_percent=1):
    control_hint = initial_hint_image.movedim(-1, 1)
    return checkpoint.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))


def control_net_set_apply_hint(c_net, c_net_set, hint_image, strength, start_percent=0, end_percent=1):
    control_hint = hint_image.movedim(-1, 1)
    new_c_net = c_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
    new_c_net.set_previous_controlnet(c_net_set)
    return new_c_net


def cn_preprocess(imgs, preprocess_alg, **kwargs):
    # TODO: maybe consider selectin resolution differently
    res = min(imgs.shape[1], imgs.shape[2])
    if preprocess_alg == 'openpose':
        estimated, = OpenPose_Preprocessor().estimate_pose(imgs, *['enable'] * 3, 'v1.1', resolution=res)
    elif preprocess_alg == 'lineart':
        estimated, = LineArt_Preprocessor().execute(imgs, **kwargs, resolution=res)
    elif preprocess_alg == 'scribble':
        estimated, = Scribble_Preprocessor().execute(imgs, resolution=res)
    elif preprocess_alg == 'normal':
        estimated, = MIDAS_Normal_Map_Preprocessor().execute(imgs, np.pi * 2, 0.05, resolution=res)
    elif preprocess_alg == 'midas_depth':
        estimated, = MIDAS_Depth_Map_Preprocessor().execute(imgs, np.pi * 2, 0.05, resolution=res)
    elif preprocess_alg == 'leres_depth':
        estimated, = LERES_Depth_Map_Preprocessor().execute(imgs, 0, 0, resolution=res)
    elif preprocess_alg == 'canny':
        estimated, = Canny_Edge_Preprocessor().execute(imgs, 30, 50, 'disable', resolution=res)
    elif preprocess_alg == 'mediapipe_face':
        estimated, = Media_Pipe_Face_Mesh_Preprocessor().detect(imgs, 2, 100, resolution=res)
    else:
        raise NotImplementedError(preprocess_alg)
    # estimated, = OpenPose_Preprocessor().estimate_pose(imgs, True, True, True, 'v1')
    # from custom_nodes.comfy_controlnet_preprocessors.nodes.edge_line import Scribble_Preprocessor
    estimated = image_scale(estimated, width=imgs.shape[2], height=imgs.shape[1])
    assert estimated.shape == imgs.shape, (estimated.shape, imgs.shape)
    return estimated


def model_merge_simple(chkp1, chkp2, ratio):
    return ModelMergeSimple().merge(chkp1, chkp2, 1 - ratio)[0]


def clip_merge_simple(chkp1, chkp2, ratio):
    return CLIPMergeSimple().merge(chkp1, chkp2, 1 - ratio)[0]


def load_cn(path):
    return ControlNetLoader().load_controlnet(path)[0]


def _load_cn_check(paths, path_key):
    if path_key not in paths:
        return None
    try:
        return load_cn(paths[path_key])
    except AttributeError:
        print('WARNING: skipping non-existent checkpoint ' + str(paths[path_key]))
        return None


def load_sd_checkpoint(path):
    chkp = CheckpointLoaderSimple().load_checkpoint(path)
    return dict(sd=chkp[0], clip=chkp[1], vae=chkp[2])


def load_upscale_model(path):
    return UpscaleModelLoader().load_model(path)[0]


def load_checkpoints(paths):
    init_custom_nodes()
    with torch.no_grad():
        sd = load_sd_checkpoint(paths['sd'])
        ups = load_upscale_model(paths['upscale_model'])
        cn_depth = _load_cn_check(paths, 'cn_depth')
        return dict(
            **sd,
            cn_openpose=_load_cn_check(paths, 'cn_openpose'),
            cn_lineart=_load_cn_check(paths, 'cn_lineart'),
            cn_normal=_load_cn_check(paths, 'cn_normal'),
            cn_canny=_load_cn_check(paths, 'cn_canny'),
            cn_scribble=_load_cn_check(paths, 'cn_scribble'),
            cn_leres_depth=cn_depth,
            cn_midas_depth=cn_depth,
            cn_mediapipe_face=_load_cn_check(paths, 'cn_mediapipe_face'),
            cn_qr=_load_cn_check(paths, 'cn_qr'),
            cn_qr2=_load_cn_check(paths, 'cn_qr2'),
            upscale_model=ups,
        )


def create_empty_latent(width, height, batch_size):
    latent, = EmptyLatentImage().generate(width, height, batch_size)
    return latent
    # latent['samples'] = torch.concatenate([latent['samples']] * batch_size, axis=0)
    # latent['batch_index'] = 0


def clip_encode(clip, text):
    condition, = CLIPTextEncode().encode(clip, text)
    # TODO add asserts
    assert len(condition) == 1
    assert len(condition[0][1]) == 1
    # print('a')
    # print(condition[0][0].shape)
    # print(next(iter(condition[0][1].values())).shape)
    return [[condition[0][0], {}]]
    # assert len(condition[0]) == 2 and condition[0][1] == {}
    # return condition


def clip_set_last_layer(clip_chkp, last_layer=-1):
    ret, = CLIPSetLastLayer().set_last_layer(clip_chkp, last_layer)
    return ret


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
           batch_index=None, cn=None):
    if batch_index is None:
        batch_index = [0] * latent_image['samples'].shape[0]

    for cond in [positive, negative]:
        assert len(cond) >= 1
        for c in cond:
            assert len(c) == 2
            assert not c[1]
    if cn is not None:
        positive = [[positive[0][0], {'control': cn, 'control_apply_to_uncond': True}]]
    latents, = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                               dict(samples=latent_image['samples'], batch_index=batch_index),
                               denoise=denoise)
    return latents


def image_scale(image, width, height, crop='disabled'):
    image, = ImageScale().upscale(image, 'bicubic', width, height, crop)
    image -= image.min()
    image /= image.max()
    return image


def image_scale_by(image, by):
    image, = ImageScaleBy().upscale(image, 'bicubic', by)
    return image


def image_upscale_w_model(checkpoint, image):
    ret, = ImageUpscaleWithModel().upscale(checkpoint, image)
    return ret


def upscale_latent(latent, width, height):
    latent, = LatentUpscale().upscale(latent, 'bilinear', width, height, 'disabled')
    return latent


def upscale_latent_by(latent, by):
    latent, = LatentUpscaleBy().upscale(latent, 'bilinear', by)
    return latent


def vae_decode(checkpoint, latent):
    ret, = VAEDecode().decode(checkpoint, latent)
    return ret


def vae_encode(checkpoint, image):
    ret, = VAEEncode().encode(checkpoint, image)
    return ret


def load_image(path):
    img, mask = LoadImage().load_image(path)
    return img, mask
