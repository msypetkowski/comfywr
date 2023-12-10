import numpy as np
import torch

from .csd_lib import create_empty_latent, clip_encode, sample, upscale_latent, upscale_latent_by, vae_decode, \
    load_image, vae_encode, image_upscale_w_model, image_scale, clip_set_last_layer, cn_preprocess, \
    control_net_set_apply_hint, \
    control_net_set_create
from .my_lib import batch_conditions, interpolate_conditions, put_text


def interpolate_prompts_adaptive(checkpoints, p1, p2, start_weight=0, end_weight=1, batch_size=8,
                                 dist_thr=140, control_img=None, sampler_settings=None):
    with torch.no_grad():
        # imgs, _weights = interpolate_fun(weights[1], weights[-2], batch_size, control_img)
        def interpolate_fun(w1, w2, bs):
            return interpolate_prompts(checkpoints, p1, p2, w1, w2, bs, control_img, sampler_settings)

        ret, weights = _interpolate_adaptive(interpolate_fun, start_weight, end_weight, batch_size, dist_thr=dist_thr)
    assert len(ret) == len(weights)
    assert all(w1 < w2 for w1, w2 in zip(weights, weights[1:])), weights
    return torch.stack(ret, axis=0), weights


def image_distance(img1, img2):
    # assert np.min(img1) == np.min(img2) == 0
    # assert np.max(img1) == np.max(img2) == 255
    return torch.mean(torch.abs(img1 - img2) * 255).numpy()


def postprocess_interpolated(images, weights=None):
    assert len(images.shape) == 4
    proc_images = [(img.cpu().numpy() * 255).astype(np.uint8).copy() for img in images]
    if weights is not None:
        for proc_img2, img1, img2, w in zip(proc_images[1:], images, images[1:], weights):
            put_text(proc_img2, str(image_distance(img1, img2)), [20, 20], (255, 0, 0))
            put_text(proc_img2, str(w), [20, 40], (255, 0, 0))
    assert len(proc_images[0].shape) == 3, proc_images[0].shape
    return proc_images


def _interpolate_adaptive(interpolate_fun, start_weight, end_weight, batch_size, dist_thr=22, epsilon=4e-4,
                          first_image=None, last_image=None):
    assert end_weight > start_weight
    if end_weight - start_weight < epsilon:
        print('range is lower than EPSILON !!! skipping')
        assert first_image is not None
        return [first_image, last_image], [start_weight, end_weight]
    images = []
    if first_image is not None:
        weights = np.linspace(start_weight, end_weight, batch_size + 2)
        assert last_image is not None
        images.append(first_image)
        imgs, _weights = interpolate_fun(weights[1], weights[-2], batch_size)
        images.extend(imgs)
        assert (np.isclose(_weights, weights[1:-1])).all()
        images.append(last_image)
    else:
        weights = np.linspace(start_weight, end_weight, batch_size)
        if batch_size >= 2:
            images, _weights = interpolate_fun(start_weight, end_weight, batch_size)
            assert (weights == _weights).all()
        elif batch_size == 1:
            img1, w1 = interpolate_fun(start_weight, start_weight, 1)
            img2, w2 = interpolate_fun(end_weight, end_weight, 1)
            images = torch.concatenate([img1, img2])
            weights = np.concatenate([w1, w2])
        else:
            assert 0

    assert len(weights) == len(images)

    ret_images = []
    ret_weights = []
    for img1, img2, w1, w2 in zip(images[:-1], images[1:], weights[:-1], weights[1:]):
        ret_images.append(img1)
        ret_weights.append(w1)
        if image_distance(img1, img2) > dist_thr:
            imgs, _weights = _interpolate_adaptive(interpolate_fun, w1, w2, batch_size, dist_thr,
                                                   first_image=img1, last_image=img2)
            ret_images.extend(imgs[1:-1])
            ret_weights.extend(_weights[1:-1])
    ret_images.append(img2)
    ret_weights.append(w2)
    return ret_images, ret_weights


@torch.no_grad()
def interpolate_prompts(checkpoints, p1, p2, start_weight=0, end_weight=1, batch_size=8, sampler_settings=None):
    print(start_weight, end_weight)

    condition1 = clip_encode(checkpoints['clip'], p1)
    condition2 = clip_encode(checkpoints['clip'], p2)
    neg_condition1 = clip_encode(checkpoints['clip'], 'embedding:EasyNegative.safetensors')
    neg_condition2 = neg_condition1

    conditions_batched = []
    neg_conditions_batched = []

    coeffs = np.linspace(start_weight, end_weight, batch_size)
    for idx, i in enumerate(coeffs):
        conditions_batched.append(interpolate_conditions(condition1, condition2, i))
        neg_conditions_batched.append(interpolate_conditions(neg_condition1, neg_condition2, i))

    conditions_batched = batch_conditions(conditions_batched)
    neg_conditions_batched = batch_conditions(neg_conditions_batched)

    # "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc", "uni_pc_bh2"]
    initial_w = 512 - 128
    initial_h = 512 + 128
    imgs = hq_infer(checkpoints, neg_conditions_batched, initial_w, conditions_batched, initial_h, sampler_settings)

    return imgs, coeffs


@torch.no_grad()
def video_transform(checkpoints, vid, pos_txt, neg_txt, sampler_settings, denoise=0.5, clip_skip=2):
    controls = ['lineart', 'canny', 'openpose']
    weights = [0.7, 0.35, 1.0]
    # controls = ['lineart', 'canny']
    ret = []
    ret_control_vis = []
    # last_latent = None
    for frame in vid:
        frame = frame[None,]
        preprocessed = torch.stack([cn_preprocess(frame, control) for control in controls])
        ret_control_vis.append(preprocessed.mean(dim=0))
        cn = control_net_set_create(checkpoints[f'cn_{controls[0]}'], preprocessed[0], strength=weights[0])
        for control, prepr, w in zip(controls[1:], preprocessed[1:], weights[1:]):
            cn = control_net_set_apply_hint(checkpoints[f'cn_{control}'], cn, prepr, strength=w)
        # if last_latent is None:
        latent = vae_encode(checkpoints['vae'], frame)
        # else:
        #     latent = last_latent
        clip_chkp = checkpoints['clip']
        if clip_skip:
            clip_chkp = clip_set_last_layer(clip_chkp, -clip_skip)
        condition = clip_encode(clip_chkp, pos_txt)
        neg_condition = clip_encode(checkpoints['clip'], neg_txt)
        sampler_settings['cn'] = cn
        latent = sample(checkpoints['sd'], positive=condition, negative=neg_condition,
                        latent_image=latent, denoise=denoise, **sampler_settings)
        # last_latent = latent
        ret.append(vae_decode(checkpoints['vae'], latent))

    return torch.concatenate(ret), torch.concatenate(ret_control_vis)


@torch.no_grad()
def hq_infer_txt(checkpoints, initial_w=16 * 64, initial_h=9 * 64, batch_size=1,
                 pos_txt='high quality photo', neg_txt='embedding:EasyNegative.safetensors',
                 sampler_settings=None, upscale_by=1.5, initial_denoise=1.0, upscaled_denoise=0.75,
                 use_upscaler=False, return_first_stage=False, clip_skip=None,
                 no_upscale=False, initial_image=None, sampler_settings_stage2=None, style_image=None):
    chkp = checkpoints['clip']
    if clip_skip:
        chkp = clip_set_last_layer(chkp, -clip_skip)
    condition = clip_encode(chkp, pos_txt)
    neg_condition = clip_encode(checkpoints['clip'], neg_txt)
    return hq_infer(checkpoints, initial_w, initial_h, batch_size, condition, neg_condition,
                    sampler_settings, upscale_by, initial_denoise, upscaled_denoise, use_upscaler, return_first_stage,
                    no_upscale, initial_image, sampler_settings_stage2, style_image)


@torch.no_grad()
def hq_infer(checkpoints, initial_w, initial_h, batch_size, conditions, neg_conditions,
             sampler_settings, upscale_by=1.5, initial_denoise=1.0, upscaled_denoise=0.75,
             use_upscaler=False, return_first_stage=False, no_upscale=False,
             initial_image=None, sampler_settings_stage2=None, style_image=None):
    if sampler_settings_stage2 is None:
        sampler_settings_stage2 = sampler_settings
    if initial_image is None:
        latent = create_empty_latent(initial_w, initial_h, batch_size)
    else:
        if use_upscaler:
            upscaled = image_upscale_w_model(checkpoints['upscale_model'], initial_image)
            upscaled = image_scale(upscaled, initial_w, initial_h)
        else:
            upscaled = image_scale(initial_image, initial_w, initial_h)
        latent = vae_encode(checkpoints['vae'], upscaled)
    small_latents = sample(checkpoints['sd'], positive=conditions, negative=neg_conditions,
                           latent_image=latent, denoise=initial_denoise, **sampler_settings)
    if no_upscale:
        small_img = vae_decode(checkpoints['vae'], small_latents)
        large_img = small_img
    else:
        if use_upscaler:
            decoded = vae_decode(checkpoints['vae'], small_latents)
            upscaled = image_upscale_w_model(checkpoints['upscale_model'], decoded)
            upscaled = image_scale(upscaled, int(initial_w * upscale_by), int(initial_h * upscale_by))
            big_latents = vae_encode(checkpoints['vae'], upscaled)
        else:
            big_latents = upscale_latent_by(small_latents, upscale_by)
        big_latents = sample(checkpoints['sd'], positive=conditions, negative=neg_conditions,
                             latent_image=big_latents, denoise=upscaled_denoise, **sampler_settings_stage2)
        large_img = vae_decode(checkpoints['vae'], big_latents)
    if return_first_stage:
        return large_img, vae_decode(checkpoints['vae'], small_latents)
    else:
        return large_img


@torch.no_grad()
def interpolate_img2img_adaptive(checkpoints, input_img, prompt, start_weight=0.001, end_weight=1,
                                 dist_thr=140,
                                 batch_size=2, sampler_settings=None):
    # imgs, _weights = interpolate_fun(weights[1], weights[-2], batch_size, control_img)
    condition, latent, neg_condition = _prepare_img2img(checkpoints, input_img, prompt)

    def interpolate_fun(w1, w2, bs):
        return interpolate_img2img_latent(checkpoints, latent, condition, neg_condition,
                                          w1, w2, bs, sampler_settings=sampler_settings)

    ret, weights = _interpolate_adaptive(interpolate_fun, start_weight, end_weight, batch_size,
                                         dist_thr=dist_thr)
    assert len(ret) == len(weights)
    assert all(w1 < w2 for w1, w2 in zip(weights, weights[1:])), weights
    return torch.stack(ret, axis=0), weights


@torch.no_grad()
def interpolate_img2img(checkpoints, input_img, prompt, start_weight=0.001, end_weight=1, batch_size=2,
                        sampler_settings=None):
    condition, latent, neg_condition = _prepare_img2img(checkpoints, input_img, prompt)
    return interpolate_img2img_latent(checkpoints, latent, condition, neg_condition, start_weight,
                                      end_weight, batch_size, sampler_settings)


def _prepare_img2img(checkpoints, input_img, prompt):
    if isinstance(input_img, str):
        input_img, _ = load_image(input_img)
    latent = vae_encode(checkpoints['vae'], input_img)
    condition = clip_encode(checkpoints['clip'], prompt)
    neg_condition = clip_encode(checkpoints['clip'], 'embedding:EasyNegative.safetensors')
    return condition, latent, neg_condition


@torch.no_grad()
def interpolate_img2img_latent(checkpoints, input_latent, condition, neg_condition,
                               start_weight=0.001, end_weight=1, batch_size=2,
                               sampler_settings=None):
    images = []
    weights = np.linspace(start_weight, end_weight, batch_size)
    print(weights)
    for w in weights:
        l = sample(checkpoints['sd'], positive=condition, negative=neg_condition,
                   latent_image=input_latent, denoise=w, **sampler_settings)
        images.append(vae_decode(checkpoints['vae'], l)[0])
    return torch.stack(images), weights
