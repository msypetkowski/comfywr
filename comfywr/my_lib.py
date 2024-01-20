import math
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from . import csd_lib


def iter_subplots_axes(ncol, n_subplots, tile_size_col=5, tile_size_row=5, title=None, title_fontsize=14):
    """ Creates subplots figure, and iterates over axes in left-right/top-bottom order """
    nrow = math.ceil(n_subplots / ncol)
    fig, axes = plt.subplots(nrow, ncol)
    if title is not None:
        plt.suptitle(title, fontsize=title_fontsize)
    fig.set_size_inches(ncol * tile_size_col, nrow * tile_size_row)
    for i in range(n_subplots):
        if nrow > 1 and ncol > 1:
            ax = axes[i // ncol, i % ncol]
        else:
            if n_subplots > 1 or ncol > 1:
                ax = axes[i]
            else:
                ax = axes
        plt.sca(ax)
        yield ax


def put_text(img, text, pos, color, scale=1, thickness=1):
    pos[0] -= 10
    pos[1] += 10
    return cv2.putText(img, text, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def batch_conditions(conditions):
    c = torch.concatenate([c[0][0] for c in conditions], axis=0)
    # print(conditions[0][0][1])
    return [[c, conditions[0][0][1]]]
    # return [[c, {}]]


def interpolate_conditions(cond1, cond2, coeff):
    assert len(cond1) == len(cond2) == 1
    assert len(cond1[0]) == len(cond2[0]) == 2
    c1 = cond1[0][0]
    c2 = cond2[0][0]
    return [[c2 * coeff + c1 * (1 - coeff), cond1[0][1]]]


def animate_images(images, interval=10, save_path=None, fps=10, figsize_div=60):
    plt.style.use('dark_background')
    matplotlib.rcParams['animation.embed_limit'] = 10000
    # matplotlib.animation.embed_limit = dict(rc=10000)
    fig = plt.figure(figsize=(images[0].shape[1] // figsize_div, images[0].shape[0] // figsize_div))
    img = plt.imshow(images[0])

    def update(i):
        img.set_data(images[i % len(images)])  # Update the image data
        return img,

    ani = FuncAnimation(fig, update, frames=range(len(images)), blit=True, interval=interval)
    plt.tight_layout()

    if save_path is not None:
        ani.save(save_path, writer='ffmpeg', fps=fps)

    plt.close(fig)

    return HTML(ani.to_jshtml())


def images_to_video(image_list, video_name, fps):
    height, width, layers = image_list[0].shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in image_list:
        video.write(image[..., ::-1])
    video.release()


def read_video(video_path, start_frame, end_frame, scale_by=1):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print("Error: Could not open video file.")
        return None

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame + 1):

        success, frame = vidcap.read()
        if not success:
            print(f"Error: Failed to retrieve frame {i}.")
            break
        frames.append(frame[..., ::-1])

    frames = np.stack(frames)
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames)
    frames = csd_lib.image_scale_by(frames, scale_by)
    return frames


def make_unrolled_video(vid, n_unroll=4, step=3, nrows=1):
    assert n_unroll % nrows == 0
    ncols = n_unroll // nrows

    final_frames = []
    # for i in range(0, len(vid) + step - 1, step):
    i = 0
    while True:
        if i >= len(vid):
            break
        unrolled_frame_in_single_row = torch.zeros((n_unroll, *vid[0].shape), dtype=vid.dtype)
        subvid = vid[i:i + n_unroll]
        unrolled_frame_in_single_row[:subvid.shape[0], :subvid.shape[1], :subvid.shape[2]] = subvid

        unrolled_frame_rows = []
        for row_id in range(nrows):
            frames_for_row = unrolled_frame_in_single_row[row_id * ncols:(row_id + 1) * ncols]
            unrolled_frame_rows.append(torch.concatenate(list(frames_for_row), dim=1))
        unrolled_frame = torch.concatenate(unrolled_frame_rows, dim=0)

        final_frames.append(unrolled_frame)
        i += step
    return torch.stack(final_frames)


def revert_unrolled_video(vid, orig_n_frames, n_unroll=4, step=3, nrows=1):
    assert n_unroll % nrows == 0
    ncols = n_unroll // nrows

    frames = defaultdict(list)
    for unrolled_frame_id in range(vid.shape[0]):
        splitted_rows = np.split(vid[unrolled_frame_id], nrows, axis=0)
        for row_id, row in enumerate(splitted_rows):
            splitted = np.split(row, ncols, axis=1)
            for col_id, frame in enumerate(splitted):
                # j += row_id * ncols
                idx = unrolled_frame_id * step + row_id * nrows + col_id
                if idx >= orig_n_frames:
                    continue
                frames[idx].append(frame)

    assert list(frames.keys()) == np.arange(len(frames)).tolist()
    frames = [torch.stack(f).mean(axis=0) for f in frames.values()]
    ret = torch.stack(frames)
    return ret
