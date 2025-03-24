# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os

import imageio
import numpy as np
import torch as th
from einops import rearrange, repeat
from torch.utils.data import default_collate

# configure logger before importing anything else
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def load_samples():
    """load all samples"""
    local_dir = os.getcwd().split("scripts/")[0]
    samples_path = sorted(
        glob.glob(os.path.join(local_dir, "sample_data/ava_samples/*.npy"))
    )
    samples = [
        np.load(sample_path, allow_pickle=True).item() for sample_path in samples_path
    ]
    return samples


def load_batches(batch_size=2, num_views=2, resolution=1024, num_samples=10):
    """load sample batches for overfitting"""
    local_dir = os.getcwd().split("scripts/")[0]
    samples_path = sorted(
        glob.glob(os.path.join(local_dir, "sample_data/ava_samples/*.npy"))
    )
    samples = [
        np.load(sample_path, allow_pickle=True).item() for sample_path in samples_path
    ]
    samples = samples[:num_samples]

    if batch_size > num_samples:
        samples = samples * (batch_size // num_samples + 1)

    # remove extra views
    for sample in samples:
        for k in [
            "video",
            "cams",
            "cam_intrin",
            "cam_extrin",
            "cam_pose",
            "kpts",
            "frames",
            "plucker",
        ]:
            if k in sample:
                sample[k] = sample[k][:num_views]

    # resize images
    for sample in samples:
        for k in ["video", "ref_image", "plucker", "ref_plucker", "kpts", "ref_kpts"]:
            if k in sample:
                value = rearrange(sample[k], "nv t c h w -> (nv t) c h w")
                value = th.nn.functional.interpolate(
                    value, size=(resolution, resolution)
                )
                value = rearrange(
                    value,
                    "(nv t) c h w -> nv t c h w",
                    nv=1 if "ref" in k else num_views,
                )
                sample[k] = value

    batches = []
    for i in range(0, len(samples), batch_size):
        batch = default_collate(samples[i : i + batch_size])
        if "video" in batch:
            batch["image"] = batch.pop("video")
        batches.append(batch)

    return batches


def visualize_sample(sample, num_cams, vid_path):
    """Self-contained utility function to visualize a single sample from the dataset"""

    # visualize (video shape: [num_cams, num_frames, height, width, channels])
    if len(sample["video"].shape) == 5:
        sample["video"] = rearrange(sample["video"], "n t c h w -> (n t) c h w")
        sample["kpts"] = rearrange(sample["kpts"], "n t c h w -> (n t) c h w")
        sample["ref_image"] = rearrange(sample["ref_image"], "n t c h w -> (n t) c h w")

    nt, c, h, w = sample["video"].shape
    vid_list = []

    # store the video and ref image
    # the order of (n w) and (w n) decides which axis is split along first then concatenated. we want to split first into n frames then stack along the width axis.
    n = num_cams
    if sample["video"].shape[-1] == sample["video"].shape[-2] and (n != 4):
        t = nt // n
        vid = rearrange(sample["video"], "(n t) c h w -> t h (n w) c", n=n, t=t)
        vid_list.append(vid)

        if "kpts" in sample:
            kpts = rearrange(sample["kpts"], "(n t) c h w -> t h (n w) c", n=n, t=t)
            vid_list.append(kpts)
    else:
        t = nt
        vid = rearrange(sample["video"], "t c h w -> t h w c")
        vid_list.append(vid)

        if "kpts" in sample:
            kpts = rearrange(sample["kpts"], "t c h w -> t h w c")
            vid_list.append(kpts)

    # upsample ref image (if needed)
    ref_image = sample["ref_image"]
    if ref_image.shape[-2:] != (h, w):
        ref_image = TF.resize(ref_image, (h, w))

    ref_image = repeat(ref_image, "1 c h w-> t h w c", t=t)
    vid_list.append(ref_image)
    vid = th.concat(vid_list, dim=2)
    vid = vid.numpy()

    os.makedirs(os.path.dirname(vid_path), exist_ok=True)
    imageio.mimsave(vid_path, vid, fps=4)
    print(f"Saved: {vid_path}")


if __name__ == "__main__":
    samples = load_samples()
    save_dir = "/".join(
        os.path.abspath(__file__).split("/")[:-2] + ["outputs", "visuals"]
    )

    for idx, sample in enumerate(samples):
        save_path = os.path.join(save_dir, f"sample_{idx}.mp4")
        visualize_sample(sample, num_cams=4, vid_path=save_path)
        print(f"Saved: {save_path}")
