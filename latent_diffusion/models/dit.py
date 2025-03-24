# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from latent_diffusion.blocks.norm import LayerNorm, RMSNorm
from latent_diffusion.blocks.patch import Patchify, PatchifyPS, Unpatchify, UnpatchifyPS
from latent_diffusion.blocks.timestep import TimestepEmbedder

logger = logging.getLogger(__name__)

INTERPOLATE_MODE = "bilinear"
ANTIALIAS = INTERPOLATE_MODE == "bilinear"
BEFORE_FUSE = True


def modulate(x: th.Tensor, shift: th.Tensor, scale: th.Tensor) -> th.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT block modulated with ControlMLP and Attention Biasing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        # norm_layer: nn.Module = nn.LayerNorm,
        norm_layer: nn.Module = LayerNorm,
        qk_rms_norm: bool = False,
        elementwise_affine: bool = True,
        cond_attn_enabled: bool = True,
        #
        cp_attn_save_global_kv: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim, elementwise_affine=elementwise_affine)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3

        # self-attention
        self.q_norm = (
            norm_layer(self.head_dim) if not qk_rms_norm else RMSNorm(self.head_dim)
        )
        self.k_norm = (
            norm_layer(self.head_dim) if not qk_rms_norm else RMSNorm(self.head_dim)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        if cond_attn_enabled:
            # cross-attention conditioning
            self.cond_attn_norm = norm_layer(dim, elementwise_affine=elementwise_affine)
            self.cond_attn_proj = nn.Linear(dim, 2 * dim)
            # self.cond_attn_out_proj = nn.Linear(dim, dim)  # TODO: unused, remove?
            self.cond_attn_split = [dim] * 2
            self.k_cond_norm = (
                norm_layer(self.head_dim) if not qk_rms_norm else RMSNorm(self.head_dim)
            )

        # global conditioning
        self.cond_adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

    def forward(
        self,
        x: th.Tensor,
        cond: th.Tensor,
        cond_attn: Optional[th.Tensor] = None,
        control: Optional[th.Tensor] = None,
        cond_cat_fuse: bool = True,
        cond_cat_x_fuse: bool = False,
        NV: int = None,
        NVR: int = None,
        NVCC: int = None,
        before_fuse: bool = False,
        kpts_self_attn: bool = False,
        attn_bias: Optional[float] = None,
    ) -> th.Tensor:
        B, N, D = x.shape
        N_org = N
        N_kv = N
        device = x.device

        # TODO: can enable separate noise encoding for each view based on cond.shape here
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.cond_adaln(cond).chunk(6, dim=1)

        if NV is not None:
            x = rearrange(x, "(B NV) N D -> B (NV N) D", NV=NV)
            if cond_attn is not None:
                cond_attn = rearrange(cond_attn, "(B NVR) N D -> B (NVR N) D", NVR=NVR)

            if control is not None:
                control = rearrange(control, "(B NV) N D -> B (NV N) D", NV=NV)

            N = N_kv = NV * N
            B = B // NV

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        y = self.in_proj(y)

        x_mlp, q, k, v = th.split(y, self.in_split, dim=-1)

        # self-attention
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N_kv, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if cond_attn is not None:
            # cross-attention with conditioning
            B, N_cond, D_cond = cond_attn.shape
            assert D_cond == D
            y_cond = self.cond_attn_norm(cond_attn)
            y_cond = self.cond_attn_proj(y_cond)
            k_cond, v_cond = th.split(y_cond, [D] * 2, dim=-1)
            k_cond = self.k_cond_norm(
                k_cond.view(B, N_cond, self.num_heads, self.head_dim)
            ).transpose(1, 2)
            v_cond = v_cond.view(B, N_cond, self.num_heads, self.head_dim).transpose(
                1, 2
            )

            k = th.cat([k, k_cond], dim=2)
            v = th.cat([v, v_cond], dim=2)

        # attention biasing to adapt to more views (ref: pippo)
        scale = None
        if attn_bias is not None:
            num_tokens_per_view = N // (NV)

            # hardcoded number of training views (depending on resolution)
            if num_tokens_per_view == 1024:
                # 512x512
                train_views = 12
            elif num_tokens_per_view > 1024:
                # 1024x1024
                train_views = 2
            else:
                # 128x128
                train_views = 24

            curr_views = NV
            _D = q.shape[-1]
            curr_tokens = num_tokens_per_view * curr_views
            train_tokens = num_tokens_per_view * train_views
            # sqrt((log(N) / log(T)) / d)
            scale = math.sqrt(
                ((math.log(curr_tokens) / math.log(train_tokens)) * attn_bias) / _D
            )
            print(f"using rescale: {attn_bias}")

        x_attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=scale
        )

        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
        x_attn = self.attn_out_proj(x_attn)

        if x_attn.shape[0] == gate_attn.shape[0]:
            x_attn = gate_attn.unsqueeze(1) * modulate(x_attn, shift_attn, scale_attn)
        else:
            # handle different timesteps across each view
            gate_attn = repeat(gate_attn, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)
            shift_attn = repeat(shift_attn, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)
            scale_attn = repeat(scale_attn, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)

            # modulate and gate
            x_attn = x_attn * (1 + scale_attn) + shift_attn
            x_attn = gate_attn * x_attn

        # MLP activation, dropout, fc2, cond
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # NOTE: conditioning elsewhere does not seem to work
        if control is not None:
            # NOTE: with cond_cat, controls are defined only for generated tokens
            N_control = control.shape[1]
            x_mlp[:, :N_control] = x_mlp[:, :N_control] + control

        if x_mlp.shape[0] == gate_mlp.shape[0]:
            x_mlp = gate_mlp.unsqueeze(1) * modulate(x_mlp, shift_mlp, scale_mlp)
        else:
            # handle different timesteps across each view
            gate_mlp = repeat(gate_mlp, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)
            shift_mlp = repeat(shift_mlp, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)
            scale_mlp = repeat(scale_mlp, "(B NV) D -> B (NV L) D", NV=NV, L=N // NV)

            # modulate and gate
            x_mlp = x_mlp * (1 + scale_mlp) + shift_mlp
            x_mlp = gate_mlp * x_mlp

        # residual fuse
        x = x + x_attn + x_mlp

        if NV is not None:
            x = rearrange(x, "B (NV N) D -> (B NV) N D", NV=NV)
            if cond_attn is not None:
                cond_attn = rearrange(cond_attn, "B (NVR N) D -> (B NVR) N D", NVR=NVR)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        cond_dim: int = None,  # conditioning with diffusion timestep
        cond_spatial_dim: int = None,  # conditioning with 2D signals
        cond_attn_dim: int = None,  # conditioning with cross-attention
        cond_cat_dim: int = None,  # conditioning with cat-self-attenion
        depth: int = 28,
        dim: int = 1536,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        # kv_drop_size: Optional[int] = None,
        qk_rms_norm: bool = False,
        normalize_std: bool = False,
        elementwise_affine: bool = True,
        zero_init: bool = True,
        patchify_ps: bool = True,
        num_views: int = 1,  # multiplier for num_patches to use for positional encoding
        time_factor: float = 1.0,
        cond_attn_mlp: bool = True,
        learn_pe: bool = False,  # learnable positional encoding
        cond_self_attn: bool = False,  # merge cond_attn with self_attn layers
        in_proj_mul: int = 1,  # input projection layer that goes from mul * dim to dim
        in_proj_identity_init: bool = True,  # use identity for input projection
        cond_cat_x_fuse: bool = False,  # fuse x with cond_cat
        cond_cat_fuse: bool = True,  # fuse cond_cat with x
        kpts_self_attn: bool = False,  # fuse kpts with x
        cond_attn_enabled: bool = True,
        pos_embed_kind: str = "sincos",  # "sincos" or "learnable" or "old"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.normalize_std = normalize_std
        self.dim = dim

        self.cond_cat_x_fuse = cond_cat_x_fuse
        self.cond_cat_fuse = cond_cat_fuse
        logger.info(f"{self.cond_cat_x_fuse=} | {self.cond_cat_fuse=}")

        # replace cross-attention with self-attention
        self.cond_self_attn = cond_self_attn
        self.kpts_self_attn = kpts_self_attn

        self.patchify_ps = patchify_ps
        if patchify_ps:
            logger.info(f"using PatchifyPS")
            self.patchify = PatchifyPS(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                out_channels=dim,
                flatten=True,
            )

            self.unpatchify = UnpatchifyPS(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=dim,
                out_channels=self.out_channels,
                unflatten=True,
                zero_init=zero_init,
            )
        else:
            logger.info(f"using vanilla Patchify")
            self.patchify = Patchify(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                out_channels=dim,
                flatten=True,
            )
            self.unpatchify = Unpatchify(
                # img_size=img_size,
                patch_size=patch_size,
                in_channels=dim,
                out_channels=self.out_channels,
            )

        self.t_embedder = TimestepEmbedder(dim, time_factor=time_factor)

        if cond_dim is not None:
            # fused with timestep conditioning
            self.cond_embedder = nn.Linear(cond_dim, dim)
        else:
            self.cond_embedder = None

        if cond_spatial_dim is not None:
            # for semantic compressor conditioning
            self.cond_spatial_embedder = nn.Sequential(
                nn.Conv2d(cond_spatial_dim, dim * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                LayerNorm2d(dim, elementwise_affine=False, eps=1.0e-6),
            )
        else:
            self.cond_spatial_embedder = None

        self.cond_attn_mlp = cond_attn_mlp
        if cond_attn_dim is not None and cond_attn_mlp:
            # for cross-attention conditioning
            self.cond_attn_embedder = nn.Sequential(
                nn.Linear(cond_attn_dim, dim * 2, bias=True),
                nn.GELU(),
                nn.Linear(dim * 2, dim, bias=True),
                LayerNorm(dim, elementwise_affine=False, eps=1.0e-6),
            )
        elif cond_attn_dim is not None:
            self.cond_attn_embedder = nn.Linear(cond_attn_dim, dim)
        else:
            self.cond_attn_embedder = None

        self.num_views = num_views
        self.cond_cat_embedder = (
            nn.Linear(cond_cat_dim, dim) if cond_cat_dim is not None else None
        )

        # new PE
        height, width = self.patchify.grid_size
        self.pos_embed_kind = pos_embed_kind
        if pos_embed_kind == "sincos":
            self.pos_embed_block = SinCosPosEmbed2d(height=height, width=width, dim=dim)
        elif pos_embed_kind == "learnable":
            self.pos_embed_block = LearnablePosEmbed2d(
                height=height, width=width, dim=dim
            )
        # pippo uses old positional encoding
        elif pos_embed_kind == "old":
            num_patches = self.patchify.num_patches * num_views
            logger.info(f"{num_patches=}")
            logger.info(f"{patch_size=}")
            # will use fixed sin-cos embedding
            self.pos_embed = nn.Parameter(
                th.zeros(1, num_patches, dim), requires_grad=learn_pe
            )
            logger.info(f"learn positional embedding: {learn_pe}")

        else:
            raise NotImplementedError()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    # kv_drop_size=kv_drop_size,
                    elementwise_affine=elementwise_affine,
                    qk_rms_norm=qk_rms_norm,
                    cond_attn_enabled=cond_attn_enabled,
                )
                for _ in range(depth)
            ]
        )

        # intialize input projection layers
        self.in_proj_mul = in_proj_mul
        self.in_proj_identity_init = in_proj_identity_init
        self.in_proj = (
            nn.Linear(in_proj_mul * dim, dim, bias=True)
            if in_proj_mul > 1
            else nn.Identity()
        )

        self.initialize_weights()
        logger.info(f"block level precision: {th.get_float32_matmul_precision()}")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # TODO: this does not work for non-square images?
        if self.pos_embed_kind == "old":
            from latent_diffusion.blocks.pos_embed import get_2d_sincos_pos_embed

            if self.num_views > 1:
                h = w = int((self.pos_embed.shape[1] // self.num_views) ** 0.5)
                grid_size = (h, w * self.num_views)
            else:
                grid_size = int(self.pos_embed.shape[1] ** 0.5)
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=grid_size,
            )
            self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            self.pos_embed_block.initialize_weights()

        self.patchify.initialize_weights()
        self.unpatchify.initialize_weights()

        # cond
        if self.cond_embedder is not None:
            nn.init.normal_(self.cond_embedder.weight, std=0.02)

        if self.cond_attn_embedder is not None and self.cond_attn_mlp:
            nn.init.normal_(self.cond_attn_embedder[0].weight, std=0.02)
            nn.init.normal_(self.cond_attn_embedder[2].weight, std=0.02)
        elif self.cond_attn_embedder is not None:
            nn.init.normal_(self.cond_attn_embedder.weight, std=0.02)

        if self.cond_cat_embedder is not None:
            # nn.init.constant_(self.cond_cat_embedder.weight, 0)
            nn.init.constant_(self.cond_cat_embedder.bias, 0)
            nn.init.normal_(self.cond_cat_embedder.weight, std=0.02)

        if self.cond_spatial_embedder is not None:
            nn.init.normal_(self.cond_spatial_embedder[0].weight, std=0.02)
            nn.init.normal_(self.cond_spatial_embedder[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.cond_adaln[-1].weight, 0)
            nn.init.constant_(block.cond_adaln[-1].bias, 0)

        # intialize as identity for first dim and zero for rest
        logger.info(f"input proj identity init: {self.in_proj_identity_init}")
        if self.in_proj_mul > 1 and self.in_proj_identity_init:
            nn.init.eye_(self.in_proj.weight[:, : self.dim])
            nn.init.constant_(self.in_proj.weight[:, self.dim :], 0)
            nn.init.constant_(self.in_proj.bias, 0)

    def forward(
        self,
        x: th.Tensor,
        t: th.Tensor,
        cond: Optional[th.Tensor] = None,
        cond_spatial: Optional[th.Tensor] = None,
        cond_attn: Optional[th.Tensor] = None,
        cond_cat: Optional[th.Tensor] = None,
        cond_mask: Optional[Tuple[th.Tensor, th.Tensor]] = None,
        controls: Optional[List[th.Tensor]] = None,
        pos_enc_x: Optional[List[th.Tensor]] = None,
        patch_pos: bool = True,
        attn_bias: Optional[float] = None,
        **kwargs,
    ) -> th.Tensor:
        """
        Forward pass of DiT.
        Args:
            x: (B, C, H, W) tensor of spatial inputs
            t: (B,) tensor of diffusion timesteps
            cond: (B, D) tensor of conditioning codes (for modulation)
            cond_attn: (B, N_d, D) tensor of conditionings (for cross-attention)
            cond_cat: (B, N_d, D) tensor of conditionings (for cat-self-attention)
            cond_spatial: (B, C_ds, H_ds, W_ds) tensor of spatial conditionings (additive)
            controls: List[th.Tensor] of (B, N, D)
            pos_enc_x: List[th.Tensor] of  (B, N, D) for additional conditioning on x
        """
        NV, NVR = None, None
        if len(x.shape) == 4 and patch_pos:
            B, C, H, W = x.shape
        else:
            B, NV = x.shape[:2]

            # flatten out NV / NVR (operate on views independently)
            x = rearrange(x, "B NV ... -> (B NV) ...")

            if cond_attn is not None:
                _, NVR, _, _ = cond_attn.shape
                cond_attn = rearrange(cond_attn, "B NVR N D -> (B NVR) N D")

            if pos_enc_x is not None:
                pos_enc_x = rearrange(pos_enc_x, "B NV N D -> (B NV) N D")

            if cond_cat is not None:
                cond_cat = rearrange(cond_cat, "B NV N D -> (B NV) N D")

            if len(t.shape) > 1:
                t = rearrange(t, "B NV -> (B NV)")

            if cond is not None and len(cond.shape) > 2:
                cond = rearrange(cond, "B NV ... -> (B NV) ...")

            # print(f"B: {B} | NV: {NV} | NVR: {NVR}")

        if self.normalize_std:
            x = x / (x.std(axis=(1, 2, 3), keepdims=True) + 1.0e-8)

        # patchify and add positional encoding (if not already)
        if patch_pos:
            # patchify all views independently
            x = self.patchify(x)

            if self.pos_embed.shape[1] > x.shape[1]:
                # positional encoding different for different views
                pos_embed = repeat(
                    self.pos_embed, "1 (NV N) D -> (B NV) N D", NV=NV, B=B
                )
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed

        # num tokens
        N = x.shape[1]

        if self.pos_embed_kind != "old":
            x = self.pos_embed_block(x)

        # camera positional encoding different for different views
        if pos_enc_x is not None:
            x = x + pos_enc_x

        # add concatenated conditioning
        if cond_cat is not None:
            x = th.cat([x, cond_cat], dim=-1)
            x = self.in_proj(x)

        if cond_spatial is not None:
            # spatial conditioning
            cond_spatial = self.cond_spatial_embedder(cond_spatial)
            cond_spatial = F.interpolate(
                cond_spatial,
                size=self.patchify.grid_size,
                mode="bilinear",
                align_corners=True,
            )
            cond_spatial = self.patchify.apply_flatten(cond_spatial)  # (B, N, D)
            x = x + cond_spatial

        if cond_mask is not None:
            cond_mask_img, cond_mask_emb = cond_mask
            cond_mask_attn = self.patchify(cond_mask_img)
            cond_mask_attn = cond_mask_attn * self.pos_embed + cond_mask_emb

        # cond_attn is None when in multiview setup
        if cond_attn is not None and self.cond_attn_embedder is not None:
            cond_attn = self.cond_attn_embedder(cond_attn)

        t = self.t_embedder(t)  # (B, D)

        if self.cond_embedder is not None and cond is not None:
            cond = self.cond_embedder(cond)  # (B, D)
            c = t + cond  # (B, D)
        elif cond is not None:
            c = t + cond
        else:
            c = t

        # merge cond_attn within self_attn (Pippo)
        if self.cond_self_attn:
            # preserve original shapes
            assert NV is not None and NVR is not None
            _NV, _NVR = NV, NVR
            # fuse all views
            x = rearrange(x, "(B NV) N D -> B NV N D", NV=NV)
            cond_attn = rearrange(cond_attn, "(B NVR) N D -> B NVR N D", NVR=NVR)

            # concatenated conditioning
            if cond_attn.shape[-1] != x.shape[-1]:
                cond_attn = self.in_proj(cond_attn)

            # attach for self-attention
            # NOTE: important to note the order (denoise views, then reference view)
            x = th.cat([x, cond_attn], dim=1)

            x = rearrange(x, "B NV N D -> (B NV) N D")
            # update shapes
            NV = NV + NVR
            NVR, cond_attn = None, None
            NVCC = None

        # adding `cond_cat` tokens
        if cond_cat is not None:
            # NOTE: only images supported for cond_cat currently
            assert len(x.shape) == 3
            assert self.cond_cat_embedder is not None
            # TODO: test whether this makes a difference?
            cond_cat = self.cond_cat_embedder(cond_cat)
            # cond_cat = self.cond_attn_embedder(cond_cat)
            x = th.cat([x, cond_cat], dim=1)
            assert x.shape[1] == N + cond_cat.shape[1]

        if controls is not None:
            for b, (block, control) in enumerate(zip(self.blocks, controls)):
                x = block(
                    x,
                    cond=c,
                    cond_attn=cond_attn,
                    control=control,
                    NV=NV,
                    NVR=NVR,
                    NVCC=NVCC,
                    attn_bias=attn_bias,
                )
        else:
            for block in self.blocks:
                x = block(x, cond=c, cond_attn=cond_attn, NV=NV, NVR=NVR)

        # remove cond_attn within self_attn
        # NOTE: important to note the order (denoise views, then reference view)
        if self.cond_self_attn:
            x = rearrange(x, "(B NV) N D -> B NV N D", NV=NV)
            x, cond_attn = x[:, :_NV], x[:, _NV:]
            x = rearrange(x, "B NV N D -> (B NV) N D")
            NV = _NV

        if cond_cat is not None:
            assert x.shape[1] == cond_cat.shape[1] + N
            x = x[:, :N]

        # unpatchify all views independently
        if self.patchify_ps:
            x = self.unpatchify(x)  # (B, C, H, W)
        else:
            if NV is not None and c.shape[0] == B:
                c = repeat(c, "B D -> (B NV) D", NV=NV)
            x = self.unpatchify(x, c)  # (B, C, H, W)

        if NV is not None:
            x = rearrange(x, "(B NV) C H W -> B NV C H W", NV=NV)

        return x
