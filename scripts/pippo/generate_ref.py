# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Do not use this file for Pippo inference, it is only kept to be used during training. Use inference.py instead."""

import glob
import json
import logging
import os
import pprint
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from easydict import EasyDict as edict
from einops import rearrange, repeat
from more_itertools import chunked
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from common import global_registry
from latent_diffusion.samplers.ddim import DDIMSampler
from latent_diffusion.utils import load_checkpoint, load_from_config, to_device

th.set_float32_matmul_precision("highest")
SAVE_NPY = False
registry = edict(uncond_ramp=None)

# configure logger before importing anything else
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logging intialized.")


class InferenceSampler(Callable):
    """Wrapper around DDIM sampler to handle conditioning and unconditional sampling."""

    def __init__(
        self,
        sampler,
        model,
        num_channels,
        n_ddim_steps: int = None,
        cfg_rescale: float = None,
        uncond_scale: float = None,
        ddim_eta: float = None,
        ref_image_fn: Optional[Callable] = None,
        device=None,
        config=None,
        fixed_noise: bool = False,
        ar_config: Optional[dict] = None,
        dataset=None,
    ):
        model.eval()

        self.sampler = sampler
        self.model = model
        self.device = device
        self.num_channels = num_channels
        self.fixed_noise = fixed_noise
        self.noise = None
        self.ar_config = edict(ar_config)
        self.dataset = dataset
        self.n_ddim_steps = n_ddim_steps
        self.cfg_rescale = cfg_rescale
        self.uncond_scale = uncond_scale
        self.ddim_eta = ddim_eta
        self.num_channels = num_channels
        self.sampler.make_schedule(ddim_num_steps=n_ddim_steps, ddim_eta=ddim_eta)
        self.ref_image_fn = ref_image_fn
        self.flow_mode = False

    def cond_fn(self, cond, *args, **kwargs):
        # ensure that conditioning all conditionings are active with training=False
        return self.model.get_conds(*args, **cond, **kwargs, training=False)

    def uncond_fn(self, uncond, *args, **kwargs):
        return self.model.get_unconds(*args, **uncond, **kwargs)

    def _generate_image(self, batch: Dict[str, th.Tensor]) -> th.Tensor:
        # find image shapes
        if isinstance(self.model.img_size, (tuple, list)):
            H, W = self.model.img_size[:2]
        else:
            H = W = self.model.img_size
        B, NV, T = 1, self.model.num_views, self.model.num_frames
        zh, zw = H // 8, W // 8

        # keys to build conditional and unconditional dicts
        cond_keys = [
            "ref_image",
            "cam_pose",
            "ref_cam_pose",
            "kpts",
            "ref_kpts",
            "plucker",
            "ref_plucker",
        ]

        if getattr(self.model, "multiview_grid_stack", False):
            denoise_shape = (1, self.num_channels, zh, zw)
        else:
            denoise_shape = (1, NV, self.num_channels, zh, zw)

        cond = {k: v for k, v in batch.items() if k in cond_keys}
        uncond = {k: v for k, v in batch.items() if k in cond_keys}

        # prepare identifiers for each batch
        try:
            identifier = dict(
                capture=batch["capture"],
                frame=batch["frames"].int().tolist(),
                cams=batch["cams"].int().tolist(),
            )
        except:
            identifier = None
            if SAVE_NPY:
                logger.warning(f"Saving NPY is enabled, but no identifier found.")

        x_T = None
        if self.fixed_noise:
            if self.noise is None:
                self.noise = th.randn(
                    denoise_shape, dtype=th.float32, device=self.device
                )
            x_T = self.noise
            denoise_shape = None

        uncond_scale = self.uncond_scale

        if registry.uncond_ramp is not None:
            # half and half
            # uncond_ramp = np.linspace(9.0, self.uncond_scale, NV//2)

            # quarter, fixed, quarter
            uncond_ramp1 = np.linspace(registry.uncond_ramp, self.uncond_scale, NV // 4)
            uncond_ramp2 = np.linspace(self.uncond_scale, self.uncond_scale, NV // 4)
            uncond_ramp = np.concatenate([uncond_ramp1, uncond_ramp2])

            uncond_scale = th.from_numpy(
                np.concatenate([uncond_ramp, uncond_ramp[::-1]])
            )
            print(f"using uncond ramp: {uncond_scale}")

        # default is fp16
        inference_dtype = global_registry.get("inference_dtype", th.float32)
        with th.no_grad(), th.autocast(device_type="cuda", dtype=inference_dtype):
            x_sample = self.sampler.sample(
                n_ddim_steps=self.n_ddim_steps,
                shape=denoise_shape,
                uncond=uncond,
                cond=cond,
                cond_fn=self.cond_fn,
                uncond_fn=self.uncond_fn,
                cfg_scale=uncond_scale,
                cfg_rescale=self.cfg_rescale,
                x_T=x_T,
            )

            if len(x_sample.shape) == 5:
                B, NV = x_sample.shape[:2]
                x_sample = rearrange(x_sample, "B NV C H W -> (B NV) C H W")

                if not global_registry.get("chunk_vae", False):
                    try:
                        image_sample = (
                            self.model.decode_image(x_sample)
                            .clip(0, 255)
                            .to(th.uint8)
                            .cpu()
                        )
                    except:
                        global_registry.chunk_vae = True

                # chunk VAE to avoid OOM
                if global_registry.get("chunk_vae", False):
                    slices = th.split(x_sample, 2, dim=0)
                    slices_dec = [self.model.decode_image(x) for x in slices]
                    image_sample = (
                        th.cat(slices_dec, dim=0).clip(0, 255).to(th.uint8).cpu()
                    )

                image_sample = rearrange(
                    image_sample, "(B NV) C H W -> B NV C H W", B=B, NV=NV
                )
            else:
                image_sample = (
                    self.model.decode_image(x_sample).clip(0, 255).to(th.uint8).cpu()
                )

        output = {
            "image_sample": image_sample,
            "sample_id": identifier,
        }

        # add input conditions for visualization
        for k in ["ref_image", "ref_kpts", "kpts", "mask"]:
            if k in batch:
                output[k] = batch[k].cpu()
            else:
                output[k] = None
        return output

    def __call__(self, batch: Dict[str, Any]) -> Tuple[th.Tensor, th.Tensor]:
        return self._generate_image(batch)


def generate_and_save(
    sampler,
    dataloader_or_dataset,
    output_dir,
    n_max_samples=10,
    n_views_per_sample=3,
    save_gt=True,
    save_ref_image=True,
    device=None,
    sample_name=None,
    metrics=None,
    ref_images_dir=None,  # dir of reference images (passed externally to evaluate on)
    traj=None,  # if True, use custom cameras along a trajectory
    seq=False,  # if True, reference frames are a video sequence
    nerfstudio=False,  # if True, save outputs that can be used that can be splatted w/ nerfstudio
):
    outputs, metrics_list = [], []
    sample_name = "sample" if sample_name is None else sample_name

    for sid, batch in tqdm(enumerate(dataloader_or_dataset), desc="Number of Samples:"):
        from copy import deepcopy

        batch = deepcopy(batch)

        # keep only first sample
        for k, v in batch.items():
            if isinstance(v, (th.Tensor, np.ndarray, list)):
                batch[k] = v[:1]

        gen_samples = []
        for i in range(n_views_per_sample):
            success = False
            while not success:
                try:
                    preds = sampler(batch)
                    success = True
                except Exception as e:
                    print(f"Exception occured: {e}")
                    import traceback

                    logger.info(traceback.format_exc())
            gen_samples.append(preds)

        save_path = f"{sample_name}_{sid}"
        image_or_video_grid, sample_metrics = save_image(
            batch,
            gen_samples,
            output_dir,
            save_gt,
            save_ref_image,
            save_path,
            metrics,
            traj=traj,
            nerfstudio=nerfstudio,
        )

        outputs.append(image_or_video_grid)
        metrics_list.append(sample_metrics)

        if len(gen_samples) >= n_max_samples:
            break

    return outputs, metrics_list


def save_image(
    batch,
    gen_samples,
    output_dir,
    save_gt,
    save_ref_image,
    sample_name,
    metrics,
    traj,
    nerfstudio,
):
    # attempt squeezing only along NV,T axes (if they exist)
    images = [preds["image_sample"].squeeze(1, 2) for preds in gen_samples]
    new_axis_mode = len(images[0].shape) == 5

    if new_axis_mode:
        B, NV, C, H, W = gen_samples[0]["image_sample"].squeeze(1, 2).shape
        images = [
            (
                rearrange(_img, "B NV C H W -> B C H (NV W)")
                if len(_img.shape) == 5
                else _img
            )
            for _img in images
        ]
    else:
        B, C, H, W = gen_samples[0]["image_sample"].squeeze(1, 2).shape

    # we use same dataloader as video
    if "gt_image" in gen_samples[0]:
        batch["image"] = gen_samples[0]["gt_image"].squeeze(1, 2)
    else:
        batch["image"] = batch["image"].squeeze(1, 2)

    batch["image"] = (
        rearrange(batch["image"], "B NV C H W -> B C H (NV W)")
        if len(batch["image"].shape) == 5
        else batch["image"]
    )

    if "image" in batch:
        image_gt = batch["image"].cpu()
        images = [image_gt] + images

    if "ref_image" in batch:
        # interpolate to match video size
        ref_image = gen_samples[0]["ref_image"].squeeze(1, 2)
        ref_image = F.interpolate(ref_image, size=(H, W))
        # attach ref_image to each image and stack as column
        images = [th.cat([ref_image, _img], dim=-1) for _img in images]

    for k in ["mask", "kpts"]:
        if k in batch:
            if k == "kpts" and "ref_kpts" in batch:
                # concat ref_kpts and kpts together
                v = th.cat([batch["ref_kpts"], batch[k]], dim=1)
                v = v.squeeze(1, 2).cpu()
            elif k == "mask":
                # rescale mask
                v = batch[k].squeeze(1, 2).cpu() * 255.0
            else:
                raise NotImplementedError("only kpts and mask are supported")
            v = v[:, : NV + 1]  # one for reference
            v = rearrange(v, "B NV C H W -> B C H (NV W)")
            images = images + [v]
    try:
        image_grid = th.cat(images, dim=-2).permute(0, 2, 3, 1).cpu()
    except:
        breakpoint()

    save_image_grid = image_grid.numpy().astype(np.uint8).squeeze(0)

    ext = ".jpg"
    imageio.imsave(f"{output_dir}/{sample_name}{ext}", save_image_grid)

    metrics_dict = dict()
    if metrics is not None:
        if not new_axis_mode:
            raise NotImplementedError("only new_axis_mode is supported")

        # remove keypoints / masked rows from metric grid
        gen_image_grid = image_grid[:, : (H * (len(gen_samples) + 1))]
        metrics_dict = get_metrics(
            video_or_image_grid=gen_image_grid,
            metrics=metrics,
            output_dir=output_dir,
            batch=batch,
            sample_name=sample_name,
            H=H,
            W=W,
            NV=NV,
        )

    return image_grid, metrics_dict
