# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th
from torchvision.utils import make_grid

from latent_diffusion.samplers.ddim import DDIMSampler
from latent_diffusion.utils import load_from_config
from scripts.pippo.generate_ref import InferenceSampler, generate_and_save

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)


class MultiviewDenoisingSummary:
    def __init__(self, **kwargs) -> None:
        self.summary_kwargs = kwargs

    def __call__(
        self, preds: Dict[str, Any], **kwargs
    ) -> Dict[str, Tuple[th.Tensor, str]]:
        logger.info(f"Evaluating models and generating samples")
        model = kwargs["model"]

        # datasets
        val = kwargs["val"]
        train = kwargs["train"]

        iteration = kwargs["iteration"]
        run_dir = kwargs["run_dir"]
        config = kwargs.get("config", None)

        # sampler config
        uncond_scale = self.summary_kwargs.get("uncond_scale", 3.0)
        cfg_rescale = self.summary_kwargs.get("cfg_rescale", 0.0)
        n_ddim_steps = self.summary_kwargs.get("n_ddim_steps", 20)
        ddim_eta = self.summary_kwargs.get("ddim_eta", 0.1)
        n_views_per_sample = self.summary_kwargs.get("n_views_per_sample", 2)
        n_max_samples = self.summary_kwargs.get("n_max_samples", 2)

        # hacky way to get the device
        device = model.cam_mlp.in_proj.weight.device

        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # create save dir
        sample_name = f"iter{iteration}_cfg{uncond_scale}_rs{cfg_rescale}_ddim{n_ddim_steps}_eta{ddim_eta}__{ts}"
        output_dir = f"{run_dir}/generated/"
        os.makedirs(output_dir, exist_ok=True)

        # extract model from ddp container
        num_channels = config.consts.latent_channels

        # create sampler, generate and store results
        sampler = InferenceSampler(
            sampler=DDIMSampler(model=model),
            model=model,
            n_ddim_steps=n_ddim_steps,
            ddim_eta=ddim_eta,
            cfg_rescale=cfg_rescale,
            uncond_scale=uncond_scale,
            device=device,
            num_channels=num_channels,
        )

        # generate images and save to disk
        samples, metrics = generate_and_save(
            sampler=sampler,
            dataloader_or_dataset=val,
            output_dir=output_dir,
            n_max_samples=n_max_samples,
            n_views_per_sample=n_views_per_sample,
            device=device,
            sample_name=sample_name,
        )
        logger.info(f"Saved {len(samples)} generated val samples to {output_dir}")
