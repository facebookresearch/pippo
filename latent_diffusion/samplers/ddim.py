# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from typing import Callable, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch as th
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def make_ddim_timesteps(
    ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, strength=1.0
):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        end_step = min(num_ddpm_timesteps, int(strength * num_ddpm_timesteps))
        ddim_timesteps = np.asarray(list(range(0, end_step - 1, c)))

    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    return steps_out


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    return sigmas, alphas, alphas_prev


def extract_into_tensor(a: th.Tensor, t: th.Tensor, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSampler:
    def __init__(
        self,
        model: nn.Module,
        silent: bool = False,
        normalize_std: bool = False,
    ):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.schedule.num_timesteps
        self.silent = silent
        self.normalize_std = normalize_std

    def register_buffer(self, name, attr):
        if type(attr) == th.Tensor:
            if attr.device != th.device("cuda"):
                attr = attr.to(th.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps: int,
        ddim_discretize="uniform",
        ddim_eta: float = 0.0,
        strength: float = 1.0,
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            strength=strength,
        )

        device = self.model.schedule.betas.device

        alphas_cumprod = self.model.schedule.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(th.float32).to(device)

        self.register_buffer("betas", to_torch(self.model.schedule.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.schedule.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * th.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @th.no_grad()
    def sample(
        self,
        cond: th.Tensor,
        x_T: Optional[th.Tensor] = None,
        cfg_scale: float = 1.0,
        shape: Optional[Tuple] = None,
        cfg_rescale: Optional[float] = None,
        uncond: Optional[th.Tensor] = None,
        cond_fn: Optional[Callable] = None,
        uncond_fn: Optional[Callable] = None,
        return_intermediate: bool = False,
        **kwargs,
    ) -> th.Tensor:
        device = self.model.schedule.betas.device

        if x_T is None:
            x_T = th.randn(shape, device=device)
        else:
            assert shape is None

        B = x_T.shape[0]
        x_t = x_T

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(
            time_range, desc="sample()", total=total_steps, disable=self.silent
        )

        interm_x_0 = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = th.full((B,), step, device=device, dtype=th.long)

            # take ddim step
            x_t, pred_x_0 = self.step(
                x_t,
                cond,
                t=ts,
                index=index,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                uncond=uncond,
                cond_fn=cond_fn,
                uncond_fn=uncond_fn,
                **kwargs,
            )

            if return_intermediate:
                interm_x_0.append(pred_x_0)

        if return_intermediate:
            return x_t, interm_x_0
        return x_t

    @th.no_grad()
    def step(
        self,
        x: th.Tensor,
        cond: th.Tensor,
        t: th.Tensor,
        index: int,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_rescale: Optional[float] = None,
        uncond: Optional[th.Tensor] = None,
        cond_fn: Optional[Callable] = None,
        uncond_fn: Optional[Callable] = None,
        fixed_randn_noise: Optional[th.Tensor] = None,
        **kwargs,
    ):
        B, *_, device = *x.shape, x.device
        n_dim = len(x.shape)

        if self.normalize_std:
            x = x / (x.std(axis=(1, 2, 3), keepdims=True) + 1.0e-8)

        if (uncond is None and uncond_fn is None) or cfg_scale == 1.0:
            if cond_fn is not None:
                cond = cond_fn(cond, t=t, x_t=x)
                x, t = cond.pop("x_t", x), cond.pop("t", t)
            o_t = self.model.predict(x, t, cond)
        else:
            if cond_fn is not None:
                cond = cond_fn(cond, t=t, x_t=x)
                # pippo returns x_t and t in cond dicts
                c_x, c_t = cond.pop("x_t", x), cond.pop("t", t)

            if uncond_fn is not None:
                uncond = uncond_fn(uncond, t=t, x_t=x)
                uc_x, uc_t = uncond.pop("x_t", x), uncond.pop("t", t)

            self.model.eval()
            with th.no_grad():
                # NOTE: this is less efficient than doing 1 pass but is much easier
                o_t_pos = self.model.predict(c_x, c_t, cond)
                o_t_neg = self.model.predict(uc_x, uc_t, uncond)
                o_t = o_t_neg + cfg_scale * (o_t_pos - o_t_neg)

            if cfg_rescale:
                # https://arxiv.org/pdf/2305.08891.pdf (14) - (16)
                phi = cfg_rescale
                pos_std = o_t_pos.std(dim=list(range(1, n_dim)), keepdim=True)
                cfg_std = o_t.std(dim=list(range(1, n_dim)), keepdim=True)
                o_t = phi * (pos_std / cfg_std) * o_t + (1.0 - phi) * o_t

        # TODO: this should be coming from DiffusionSchedule
        e_t = self.model.schedule.predict_noise(x, model_preds=o_t, t=t)
        x_0_pred = self.model.schedule.predict_x_0(x, model_preds=o_t, t=t)

        # misc
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        shape = (B,) + (1,) * len(x.shape[1:])
        a_t = th.full(shape, alphas[index], device=device)
        a_prev = th.full(shape, alphas_prev[index], device=device)
        sigma_t = th.full(shape, sigmas[index], device=device)
        sqrt_one_minus_at = th.full(shape, sqrt_one_minus_alphas[index], device=device)

        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * th.randn_like(x) * temperature
        x_prev = a_prev.sqrt() * x_0_pred + dir_xt + noise
        return x_prev, x_0_pred
