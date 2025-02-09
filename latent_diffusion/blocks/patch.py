# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union
import collections.abc

import torch as th
import torch.nn as nn
from einops import rearrange


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: th.Tensor):
        """
        Args:
           x: [B, C, H, W] 4D tensor
        """
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def modulate(x: th.Tensor, shift: th.Tensor, scale: th.Tensor) -> th.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def to_2tuple(x):
    return tuple(x) if isinstance(x, collections.abc.Iterable) else (x, x)


def to_3tuple(x):
    return tuple(x) if isinstance(x, collections.abc.Iterable) else (x, x, x)


class Patchify(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()

    def initialize_weights(self):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchifyPS(nn.Module):
    """
    Maps image to a collection of patch-tokens.

    Using PixelShuffle and LayerNorm2d.

    Args:
      flatten: whether to map to (B D H W) -> (B (H W) D)
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
        flatten: bool = False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_channels = in_channels

        self.blocks = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(in_channels * (patch_size**2), out_channels, kernel_size=1),
            LayerNorm2d(out_channels, elementwise_affine=False, eps=1.0e-6),
        )
        self.flatten = flatten

        self.initialize_weights()

    def initialize_weights(self):
        # small init for inputs
        nn.init.xavier_uniform_(self.blocks[1].weight, 0.02)

    def apply_flatten(self, x: th.Tensor):
        return rearrange(x, "B D H W -> B (H W) D")

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.blocks(x)
        if self.flatten:
            x = rearrange(x, "B D H W -> B (H W) D")
        return x


class Unpatchify(nn.Module):

    def __init__(self, in_channels: int, patch_size: int, out_channels: int):
        super().__init__()
        # self.img_size = to_2tuple(img_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.norm_final = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            in_channels, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(in_channels, 2 * in_channels, bias=True)
        )

    def initialize_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        Returns:
           imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = th.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return self.unpatchify(x)


class UnpatchifyPS(nn.Module):
    """
    Maps a bunch of patch-tokens to images.

    Args:
       unflatten: whether to map to (B N D) -> (B D H W)
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
        unflatten: bool = False,
        zero_init: bool = True,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unflatten = unflatten
        self.zero_init = zero_init

        self.blocks = nn.Sequential(
            LayerNorm2d(in_channels, elementwise_affine=False, eps=1.0e-6),
            nn.Conv2d(in_channels, out_channels * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # zero-init for outputs
        if self.zero_init:
            nn.init.constant_(self.blocks[1].weight, 0.0)
        else:
            nn.init.xavier_uniform_(self.blocks[1].weight, 0.1)

    def apply_unflatten(self, x: th.Tensor):
        B, N, D = x.shape
        assert N == self.num_patches
        h, w = self.grid_size
        x = x.reshape(B, h, w, D).permute(0, 3, 1, 2)
        return x

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.unflatten:
            B, N, D = x.shape
            assert N == self.num_patches
            h, w = self.grid_size
            x = x.reshape(B, h, w, D).permute(0, 3, 1, 2)
        x = self.blocks(x)
        return x


def test_patchify_ps():

    device = th.device("cuda:0")

    B, D, H, W = 32, 4, 256, 256
    P = 4

    patchify = PatchifyPS(
        img_size=(H, W),
        patch_size=P,
        in_channels=4,
        out_channels=1024,
        flatten=True,
    ).to(device)
    unpatchify = UnpatchifyPS(
        img_size=(H, W),
        patch_size=P,
        in_channels=1024,
        out_channels=4,
        unflatten=True,
    ).to(device)

    x = th.randn((B, D, H, W), device=device, dtype=th.float32)

    with th.no_grad():
        x = patchify(x)
        print(x.shape)
        x = unpatchify(x)
        print(x.shape)
