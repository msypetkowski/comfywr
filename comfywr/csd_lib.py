import numpy as np
import torch

from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL
from comfy_extras.nodes_model_merging import ModelMergeSimple, CLIPMergeSimple
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterModelLoader, IPAdapterAdvanced
from custom_nodes.comfyui_controlnet_aux.node_wrappers.canny import Canny_Edge_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.leres import LERES_Depth_Map_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.lineart import LineArt_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.mediapipe_face import Media_Pipe_Face_Mesh_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.midas import MIDAS_Normal_Map_Preprocessor, \
    MIDAS_Depth_Map_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
from custom_nodes.comfyui_controlnet_aux.node_wrappers.scribble import Scribble_Preprocessor
from custom_nodes.comfyui_marigold.nodes import MarigoldDepthEstimation
from nodes import init_custom_nodes, ControlNetLoader, CheckpointLoaderSimple, EmptyLatentImage, \
    CLIPTextEncode, LatentUpscale, LatentUpscaleBy, VAEDecode, VAEEncode, LoadImage, ImageScale, ImageScaleBy, \
    common_ksampler, CLIPSetLastLayer, LoraLoader, StyleModelLoader, CLIPVisionLoader, CLIPVisionEncode, \
    StyleModelApply


@torch.no_grad()
def ultimate_sd_upscale(image, model, positive, negative, vae, upscale_by, seed,
                        steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                        mode_type='Linear', tile_width=512, tile_height=512, mask_blur=8, tile_padding=32,
                        seam_fix_mode='None', seam_fix_denoise=1.0, seam_fix_mask_blur=8,
                        seam_fix_width=64, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False):
    from nodes import NODE_CLASS_MAPPINGS
    cls = NODE_CLASS_MAPPINGS['UltimateSDUpscale']
    img, = cls().upscale(image, model, positive, negative, vae, upscale_by, seed,
                         steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                         mode_type, tile_width, tile_height, mask_blur, tile_padding,
                         seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                         seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode)
    return img


@torch.no_grad()
def load_ipadapter(chkp_path):
    ret, = IPAdapterModelLoader().load_ipadapter_model(chkp_path)
    return ret


@torch.no_grad()
def ip_adapter_apply(ipadapter, model, weight, clip_vision,
                     image, start_at=0.0, end_at=1.0, weight_type='strong_style_transfer'):
    ret = IPAdapterAdvanced().apply_ipadapter(ipadapter=ipadapter, model=model, weight=weight, clip_vision=clip_vision,
                                              image=image, start_at=start_at, end_at=end_at,
                                              weight_type=weight_type)
    assert isinstance(ret, tuple)
    ret, _ = ret
    return ret


@torch.no_grad()
def load_lora(model, clip, lora_name, strength_model, strength_clip):
    return LoraLoader().load_lora(model, clip, lora_name, strength_model, strength_clip)


@torch.no_grad()
def load_style_model(model_name):
    return StyleModelLoader().load_style_model(model_name)[0]


@torch.no_grad()
def load_clip_vision(model_name):
    return CLIPVisionLoader().load_clip(model_name)[0]


@torch.no_grad()
def clip_vision_encode(clip_vision, img):
    return CLIPVisionEncode().encode(clip_vision, img)[0]


@torch.no_grad()
def apply_style_model(clip_vision_output, style_model, conditioning):
    return StyleModelApply().apply_stylemodel(clip_vision_output, style_model, conditioning)[0]


@torch.no_grad()
def control_net_set_create(checkpoint, initial_hint_image, strength, start_percent=0, end_percent=1):
    control_hint = initial_hint_image.movedim(-1, 1)
    return checkpoint.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))


@torch.no_grad()
def control_net_set_apply_hint(c_net, c_net_set, hint_image, strength, start_percent=0, end_percent=1):
    control_hint = hint_image.movedim(-1, 1)
    new_c_net = c_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
    new_c_net.set_previous_controlnet(c_net_set)
    return new_c_net


@torch.no_grad()
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


@torch.no_grad()
def run_marigold_depth_estimation(image, **kwargs):
    inference_scale = kwargs['inference_scale']
    if inference_scale != 1.0:
        _, h, w, _ = image.shape
        image = image_scale_by(image, inference_scale)
    kwargs.pop('inference_scale')
    kw = dict(image=image, seed=42, denoise_steps=10, n_repeat=10,
              regularizer_strength=0.02,
              reduction_method='median', max_iter=5, tol=1e-3, invert=True,
              keep_model_loaded=True, n_repeat_batch_size=2, use_fp16=True,
              scheduler='DDIMScheduler', normalize=True)
    kw.update(**kwargs)
    ret, = MarigoldDepthEstimation().process(**kw)
    if inference_scale != 1.0:
        ret = image_scale(ret, w, h)
    return ret


@torch.no_grad()
def load_depth_anything_v2():
    from nodes import NODE_CLASS_MAPPINGS
    cls = NODE_CLASS_MAPPINGS['DownloadAndLoadDepthAnythingV2Model']
    ret, = cls().loadmodel('depth_anything_v2_vitl_fp32.safetensors')
    return ret


@torch.no_grad()
def run_depth_anything_v2(image, ckpt):
    from nodes import NODE_CLASS_MAPPINGS
    cls = NODE_CLASS_MAPPINGS['DepthAnything_V2']
    ret, = cls().process(ckpt, image)
    return ret


@torch.no_grad()
def model_merge_simple(chkp1, chkp2, ratio):
    return ModelMergeSimple().merge(chkp1, chkp2, 1 - ratio)[0]


@torch.no_grad()
def clip_merge_simple(chkp1, chkp2, ratio):
    return CLIPMergeSimple().merge(chkp1, chkp2, 1 - ratio)[0]


@torch.no_grad()
def load_cn(path):
    return ControlNetLoader().load_controlnet(path)[0]


@torch.no_grad()
def _load_cn_check(paths, path_key):
    if path_key not in paths:
        return None
    try:
        return load_cn(paths[path_key])
    except AttributeError:
        print('WARNING: skipping non-existent checkpoint ' + str(paths[path_key]))
        return None


@torch.no_grad()
def load_sd_checkpoint(path):
    chkp = CheckpointLoaderSimple().load_checkpoint(path)
    return dict(sd=chkp[0], clip=chkp[1], vae=chkp[2])


@torch.no_grad()
def load_upscale_model(path):
    return UpscaleModelLoader().load_model(path)[0]


@torch.no_grad()
def load_checkpoints(paths):
    init_custom_nodes()
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


@torch.no_grad()
def create_empty_latent(width, height, batch_size):
    latent, = EmptyLatentImage().generate(width, height, batch_size)
    return latent
    # latent['samples'] = torch.concatenate([latent['samples']] * batch_size, axis=0)
    # latent['batch_index'] = 0


@torch.no_grad()
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


@torch.no_grad()
def clip_encode_sdxl(clip, width, height, crop_w, crop_h,
                     target_width, target_height, text_g, text_l):
    ret, = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h,
                                       target_width, target_height, text_g, text_l)
    return ret


@torch.no_grad()
def clip_set_last_layer(clip_chkp, last_layer=-1):
    ret, = CLIPSetLastLayer().set_last_layer(clip_chkp, last_layer)
    return ret


@torch.no_grad()
def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
           batch_index=None, cn=None):
    if batch_index is None:
        batch_index = [0] * latent_image['samples'].shape[0]

    for cond in [positive, negative]:
        assert len(cond) >= 1
        for c in cond:
            assert len(c) == 2
            assert isinstance(c[1], dict)
            # assert not c[1]
    if cn is not None:
        condition_kwargs = positive[0][1]
        condition_kwargs.update({'control': cn, 'control_apply_to_uncond': True})
        positive = [[positive[0][0], condition_kwargs]]
    latents, = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                               dict(samples=latent_image['samples'], batch_index=batch_index),
                               denoise=denoise)
    return latents


@torch.no_grad()
def image_scale(image, width, height, crop='disabled'):
    image, = ImageScale().upscale(image, 'bicubic', width, height, crop)
    # image -= image.min()
    # image /= image.max()
    return image


@torch.no_grad()
def image_scale_by(image, by):
    image, = ImageScaleBy().upscale(image, 'bicubic', by)
    return image


@torch.no_grad()
def image_upscale_w_model(checkpoint, image):
    ret, = ImageUpscaleWithModel().upscale(checkpoint, image)
    return ret


@torch.no_grad()
def upscale_latent(latent, width, height):
    latent, = LatentUpscale().upscale(latent, 'bilinear', width, height, 'disabled')
    return latent


@torch.no_grad()
def upscale_latent_by(latent, by):
    latent, = LatentUpscaleBy().upscale(latent, 'bilinear', by)
    return latent


@torch.no_grad()
def vae_decode(checkpoint, latent):
    ret, = VAEDecode().decode(checkpoint, latent)
    return ret


@torch.no_grad()
def vae_encode(checkpoint, image):
    ret, = VAEEncode().encode(checkpoint, image)
    return ret


@torch.no_grad()
def load_image(path):
    img, mask = LoadImage().load_image(path)
    return img, mask
