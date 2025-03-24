# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_into_tensor(a: th.Tensor, t: th.Tensor, x_shape: Tuple[int]) -> th.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def enforce_zero_terminal_snr(betas: th.Tensor) -> th.Tensor:
    """
    Forcing zero-terminal SNR.
    NOTE: this should be used with v-parameterization
    Args:
       betas: [T,] tensor
    Returns:
       rescael
    """
    alphas = 1.0 - betas
    alphas_cumprod = th.cumprod(alphas, dim=0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()

    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (
        alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
    )
    alphas_cumprod = alphas_cumprod_sqrt**2
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = th.cat([alphas_cumprod[0:1], alphas])
    betas = 1.0 - alphas
    return betas


def get_named_betas_schedule(
    schedule_type: str,
    num_timesteps: int,
    linear_start: float = 0.0085,
    linear_end: float = 0.0120,
    cosine_start: float = 0.008,
):
    if schedule_type == "linear":
        betas = th.linspace(linear_start, linear_end, steps=num_timesteps).to(
            th.float64
        )
    elif schedule_type == "linear_sd" or schedule_type == "scaled_linear":
        betas = (
            th.linspace(linear_start**0.5, linear_end**0.5, steps=num_timesteps)
            ** 2
        ).to(th.float64)
    elif schedule_type == "cosine":
        timesteps = (
            th.arange(num_timesteps + 1, dtype=th.float64) / num_timesteps
            + cosine_start
        )
        alphas = timesteps / (1 + cosine_start) * math.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise ValueError("unsupported schedule type")
    return betas


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        schedule_type: str = "linear_sd",
        num_timesteps: int = 1000,
        linear_start: float = 0.0085,
        linear_end: float = 0.0120,
        cosine_start: float = 0.008,
        zero_terminal_snr: bool = False,
        parameterization: str = "eps",
        noise_offset: Optional[float] = None,
        rescale_ratio: float = None,
    ):
        """
        Args:
            ...
            noise_offset: shifts noise globally
            rescale_ratio: rescales timesteps similar to SD3
        """
        super().__init__()

        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.parameterization = parameterization
        self.noise_offset = noise_offset
        self.rescale_ratio = rescale_ratio

        betas = get_named_betas_schedule(
            schedule_type=schedule_type,
            num_timesteps=num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_start=cosine_start,
        )

        if zero_terminal_snr:
            betas = enforce_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0).to(th.float32)

        # rescale alphas cumprod (noise scales) to adapt for varying image resolutions
        # equivalent of SD3 equations but for ddim instead of recitified flows
        # reference : https://arxiv.org/pdf/2403.03206#page=9.49
        if rescale_ratio is not None:
            # rescale_ratio = (# pixels in new resolution) / (# pixels in base: 256x256)
            assert rescale_ratio > 0.0 and rescale_ratio <= 64.0  # sane range
            rescaled_ac = th.zeros_like(alphas_cumprod)

            for i, _ac in enumerate(alphas_cumprod):
                rescaled_numerator = _ac
                rescaled_denominator = rescale_ratio + (1.0 - rescale_ratio) * _ac
                rescaled_ac[i] = rescaled_numerator / rescaled_denominator

            alphas_cumprod = rescaled_ac
            logger.info(f"rescaled denoising schedule with ratio: {rescale_ratio}")

        alphas_cumprod_prev = th.cat(
            [th.as_tensor([1.0], dtype=th.float32), alphas_cumprod[:-1]]
        )
        self.register_buffer("alphas", alphas.to(th.float32))
        self.register_buffer("betas", betas.to(th.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", th.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", th.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", th.log(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", th.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", th.sqrt(1.0 / alphas_cumprod - 1)
        )

    def q_sample(
        self, x_0: th.Tensor, t: th.Tensor, noise: Optional[th.Tensor] = None
    ) -> th.Tensor:
        if noise is None:
            noise = th.randn_like(x_0)

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            * noise
        )

    def predict_noise(self, x_t, t, model_preds):
        if self.parameterization == "eps":
            return model_preds
        elif self.parameterization == "v":
            return self.predict_noise_from_v(x_t, v_preds, t)
        else:
            raise NotImplementedError(
                f"parameterization `{self.parameterization}` is not supported"
            )

    def predict_x_0(
        self, x_t: th.Tensor, t: th.Tensor, model_preds: th.Tensor
    ) -> th.Tensor:
        """Get single-step x_0 from model predictions."""
        if self.parameterization == "eps":
            return self.predict_x_0_from_noise(x_t, t, model_preds.detach())
        elif self.parameterization == "v":
            return self.predict_x_0_from_v(x_t, model_preds, t)
        else:
            raise NotImplementedError(
                f"parameterization `{self.parameterization}` is not supported"
            )

    def predict_x_0_from_noise(
        self, x_t: th.Tensor, t: th.Tensor, noise: th.Tensor
    ) -> th.Tensor:
        """Get single-step x_0 from noise prediction."""
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def get_v(self, x: th.Tensor, t: th.Tensor, noise: th.Tensor) -> th.Tensor:
        """v from x and eps."""
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def predict_noise_from_v(self, x_t: th.Tensor, v: th.Tensor, t: th.Tensor):
        """eps from v."""
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def predict_x_0_from_v(self, x_t: th.Tensor, v: th.Tensor, t: th.Tensor):
        """x_0 from v and x_t."""
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def add_noise(self, x_0: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        device = x_0.device
        B, C = x_0.shape[:2]
        # TODO: move this to schedule?
        T = self.num_timesteps
        t = th.randint(0, T, size=(B,), device=device)
        noise = th.randn_like(x_0)
        if self.noise_offset is not None:
            noise = noise + self.noise_offset * th.randn(
                (B, C, 1, 1), dtype=x_0.dtype, device=device
            )
        x_t = self.q_sample(x_0, t, noise)
        return x_t, t, noise

    def _add_fixed_noise(
        self, x_0: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        device = x_0.device
        B, C = x_0.shape[:2]
        # TODO: move this to schedule?
        T = self.num_timesteps
        t = th.linspace(1.0, T - 1.0, B).to(device=device, dtype=th.int64)
        noise = th.randn_like(x_0)
        if self.noise_offset is not None:
            noise = noise + self.noise_offset * th.randn(
                (B, C, 1, 1), dtype=x_0.dtype, device=device
            )
        x_t = self.q_sample(x_0, t, noise)
        return x_t, t, noise

    def forward(
        self,
        model_preds: th.Tensor,
        x_t: th.Tensor,
        x_0: th.Tensor,
        noise: th.Tensor,
        t: th.Tensor,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        preds = {}

        preds["noise_preds"] = self.predict_noise(x_t=x_t, model_preds=model_preds, t=t)
        preds["x_0_pred"] = self.predict_x_0(x_t=x_t, model_preds=model_preds, t=t)

        # TODO: get_targets()?
        if self.parameterization == "v":
            v_preds = model_preds
            v_gt = self.get_v(x_0, t, noise)
            preds["v_gt"] = v_gt
            preds["v_preds"] = v_preds
        # TODO: should this also include a loss?

        return preds


def test_diffusion_schedule():
    schedule = DiffusionSchedule(
        schedule_type="linear",
        linear_start=0.0001,
        linear_end=0.02,
        zero_terminal_snr=False,
    )
    assert np.isclose(schedule.sqrt_alphas_cumprod[-1].item(), 0.006352818)

    schedule = DiffusionSchedule(
        schedule_type="linear_sd",
        linear_start=0.00085,
        linear_end=0.012,
        zero_terminal_snr=False,
    )
    assert np.isclose(schedule.sqrt_alphas_cumprod[-1].item(), 0.068265)

    schedule = DiffusionSchedule(
        schedule_type="linear_sd",
        linear_start=0.00085,
        linear_end=0.012,
        zero_terminal_snr=True,
    )
    assert schedule.sqrt_alphas_cumprod[-1].item() == 0.0
    print(schedule.sqrt_alphas_cumprod)


if __name__ == "__main__":
    test_schedule()
