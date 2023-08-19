import numpy as np
import torch

from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
from custom_nodes.comfy_controlnet_preprocessors.nodes.edge_line import Canny_Edge_Preprocessor, LineArt_Preprocessor, Scribble_Preprocessor
from custom_nodes.comfy_controlnet_preprocessors.nodes.normal_depth_map import MIDAS_Normal_Map_Preprocessor, \
    MIDAS_Depth_Map_Preprocessor, LERES_Depth_Map_Preprocessor
from custom_nodes.comfy_controlnet_preprocessors.nodes.pose import OpenPose_Preprocessor
from custom_nodes.comfy_controlnet_preprocessors.nodes.others import Media_Pipe_Face_Mesh_Preprocessor
from nodes import init_custom_nodes, ControlNetLoader, CheckpointLoaderSimple, EmptyLatentImage, \
    CLIPTextEncode, LatentUpscale, LatentUpscaleBy, VAEDecode, VAEEncode, LoadImage, ImageScale, ImageScaleBy, \
    VAELoader, common_ksampler, CLIPSetLastLayer


def control_net_set_create(checkpoint, initial_hint_image, strength):
    control_hint = initial_hint_image.movedim(-1, 1)
    return checkpoint.copy().set_cond_hint(control_hint, strength)


def control_net_set_apply_hint(c_net, c_net_set, hint_image, strength):
    control_hint = hint_image.movedim(-1, 1)
    new_c_net = c_net.copy().set_cond_hint(control_hint, strength)
    new_c_net.set_previous_controlnet(c_net_set)
    return new_c_net


def cn_preprocess(imgs, preprocess_alg):
    if preprocess_alg == 'openpose':
        estimated, = OpenPose_Preprocessor().estimate_pose(imgs, *['enable'] * 3, 'v1.1')
    elif preprocess_alg == 'lineart':
        estimated, = LineArt_Preprocessor().transform_lineart(imgs, coarse='enable')
    elif preprocess_alg == 'scribble':
        estimated, = Scribble_Preprocessor().transform_scribble(imgs)
    elif preprocess_alg == 'normal':
        estimated, = MIDAS_Normal_Map_Preprocessor().estimate_normal(imgs, np.pi * 2, 0.05)
    elif preprocess_alg == 'midas_depth':
        estimated, = MIDAS_Depth_Map_Preprocessor().estimate_depth(imgs, np.pi * 2, 0.05)
    elif preprocess_alg == 'leres_depth':
        estimated, = LERES_Depth_Map_Preprocessor().estimate_depth(imgs, 0, 0)
    elif preprocess_alg == 'canny':
        estimated, = Canny_Edge_Preprocessor().detect_edge(imgs, 30, 50, 'disable')
    elif preprocess_alg == 'mediapipe_face':
        estimated, = Media_Pipe_Face_Mesh_Preprocessor().detect(imgs, 2, 100)
    else:
        raise NotImplementedError(preprocess_alg)
    # estimated, = OpenPose_Preprocessor().estimate_pose(imgs, True, True, True, 'v1')
    # from custom_nodes.comfy_controlnet_preprocessors.nodes.edge_line import Scribble_Preprocessor
    return estimated


def load_cn(path):
    try:
        return ControlNetLoader().load_controlnet(path)[0]
    except AttributeError:
        print('WARNING: skipping non-existent checkpoint ' + str(path))
        return None

def load_checkpoints(paths):
    init_custom_nodes()
    with torch.no_grad():
        chkp = CheckpointLoaderSimple().load_checkpoint(paths['sd'])
        ups, = UpscaleModelLoader().load_model(paths['upscale_model'])
        cn_depth = ControlNetLoader().load_controlnet(paths['cn_depth'])[0]
        return dict(
            sd=chkp[0],
            clip=chkp[1],
            vae=chkp[2],
            cn_openpose=load_cn(paths['cn_openpose']),
            cn_lineart=load_cn(paths['cn_lineart']),
            cn_normal=load_cn(paths['cn_normal']),
            cn_canny=load_cn(paths['cn_canny']),
            cn_scribble=load_cn(paths['cn_scribble']),
            cn_leres_depth=cn_depth,
            cn_midas_depth=cn_depth,
            cn_mediapipe_face=load_cn(paths['cn_mediapipe_face']),
            cn_qr=load_cn(paths['cn_qr']),
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
        assert len(cond) == 1
        assert len(cond[0]) == 2
        assert not cond[0][1]
    if cn is not None:
        positive = [[positive[0][0], {'control': cn}]]
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
