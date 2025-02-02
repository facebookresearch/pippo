from typing import List, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from latent_diffusion.blocks.pos_embed import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)
from latent_diffusion.blocks.patch import PatchifyPS
from latent_diffusion.blocks.siren import SIREN

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: th.Tensor):
        """
        Args:
           x: [B, C, H, W] 4D tensor
        """
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class RefBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult: int = 1,
        glu: bool = True,
        init_last_zero: bool = False,
        use_skip: bool = True,
    ):
        super().__init__()
        self.dim = dim
        dim_out = dim if dim_out is None else dim_out
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, dim_out=dim_out, mult=mult, glu=glu)
        self.use_skip = use_skip

        if init_last_zero:
            self.ff.net[-1].weight.data.fill_(0)
            self.ff.net[-1].bias.data.fill_(0)
            self.use_skip = False

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.ff(self.norm1(x)) if self.use_skip else self.ff(self.norm1(x))
        return x


class RefMLP(nn.Module):
    """Lightweight alternative to RefUNet compatible with DiT architecture."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_patches_or_tokens: int,  # reference condition can be 1D / 2D
        out_channels: int = None,
        cond_type: str = "2D",  # 2D for images, 1D for tokens
        mult: int = 1,
        init_last_zero: bool = False,  # initialize last layer to zeros

    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches_or_tokens = num_patches_or_tokens

        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        out_channels = hidden_size if out_channels is None else out_channels

        # NOTE: IS THIS IN THE LIST OF PARAMS?
        blocks = [RefBlock(dim=hidden_size, mult=mult) for _ in range(depth - 1)]
        # last block has out_channels dim
        blocks += [
            RefBlock(
                dim=hidden_size,
                mult=mult,
                dim_out=out_channels,
                init_last_zero=init_last_zero,
            )
        ]
        self.blocks = nn.ModuleList(blocks)

        # make pos_embed learnable (zero intialized)
        self.pos_embed = nn.Parameter(th.zeros(1, num_patches_or_tokens, hidden_size), requires_grad=True)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize pos_embed by sin-cos embedding:
        if self.cond_type == "2D":
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.pos_embed.shape[-2]**0.5)
            )
        elif self.cond_type == "1D":
            pos_embed = get_1d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.pos_embed.shape[-2]
            )
        else:
            raise NotImplementedError(f"Unknown cond_type: {cond_type}")
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, ref_x: th.Tensor) -> List[th.Tensor]:
        """
        Args:
           ref_x: (B N D) tensor, tokenized reference (image)
        Returns:
           ref_cond: a list of reference encodings
        """
        ref_cond = []

        x = self.in_norm(ref_x)
        x = self.in_proj(x)

        if self.num_patches_or_tokens > 1:
            pos_embed = self.pos_embed[:, : x.shape[1], ...]
            x = x + pos_embed
        else:
            # preserve shape in case input is (B, D) and not (B, N, D)
            pos_embed = self.pos_embed[0] if len(x.shape) == 2 else self.pos_embed
            x = x + pos_embed

        for block in self.blocks:
            x = block(x)
            ref_cond.append(x)
        return ref_cond


class Ref2DMLP(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        out_channels: int = None,
        mult: int = 1,
        init_last_zero: bool = False,  # initialize last layer to zeros
        siren: bool = False,
        siren_config: Optional[dict] = None,
        use_pos_embed: bool = True,
        conv_block: bool = False,
    ):
        super().__init__()

        self.siren = siren
        if self.siren:
            self.siren_encoder = SIREN(**siren_config)
            in_channels = siren_config["out_features"]
            use_pos_embed = False
        self.use_pos_embed = use_pos_embed

        self.conv_block = conv_block
        if self.conv_block:
            conv_in_channels = siren_config["out_features"]
            conv_out_channels = siren_config["out_features"]
            patch_size = 2
            self.conv_encoder = nn.Sequential(
                nn.Conv2d(conv_in_channels, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(96, 96, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(96, 128, 3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, conv_out_channels, 3, padding=1),
                LayerNorm2d(conv_out_channels, elementwise_affine=True, eps=1.0e-6),
            )

        # downsample plucker to latent space
        self.patchify = PatchifyPS(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=hidden_size,
            flatten=True,
        )

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches_or_tokens = self.patchify.num_patches

        self.in_proj = nn.Linear(hidden_size, hidden_size)
        out_channels = hidden_size if out_channels is None else out_channels

        # NOTE: IS THIS IN THE LIST OF PARAMS?
        blocks = [RefBlock(dim=hidden_size, mult=mult) for _ in range(depth - 1)]
        # last block has out_channels dim
        blocks += [
            RefBlock(
                dim=hidden_size,
                mult=mult,
                dim_out=out_channels,
                init_last_zero=init_last_zero,
            )
        ]
        self.blocks = nn.ModuleList(blocks)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(th.zeros(1, self.num_patches_or_tokens, hidden_size), requires_grad=True)
            # initialize weights
            self.initialize_weights()

    def initialize_weights(self):
        # initialize weights similar to a controlnet
        if self.conv_block:
            def _basic_init(module):
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.conv_encoder.apply(_basic_init)

        # initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.pos_embed.shape[-2]**0.5)
        )
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: th.Tensor) -> List[th.Tensor]:
        """
        x: [B, C, H, W] tensor, plucker image
        """
        if self.siren:
            x = rearrange(x, "b c h w -> b h w c")
            x = self.siren_encoder(x)
            x = rearrange(x, "b h w c -> b c h w")

        # for better (and learnable) downsampling of the control signal
        if self.conv_block:
            x = self.conv_encoder(x)

        # patchify and tokenize
        x = self.patchify(x)

        # projection and add positional embedding
        x = self.in_proj(x)
        x = x + self.pos_embed if self.use_pos_embed else x

        # pass through mlp blocks (last zero initialized if needed)
        for block in self.blocks:
            x = block(x)

        return x


def test_ref_mlp():

    device = th.device("cuda:0")

    B, N, D = 16, 256, 1024

    x = th.randn((B, N, D), device=device, dtype=th.float32)

    ref_mlp = RefMLP(
        in_channels=D,
        hidden_size=D,
        depth=16,
        num_patches=N,
    ).to(device)

    x_out = ref_mlp(x)

    print(x_out[0].shape)
