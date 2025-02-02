from typing import List, Optional
import logging
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from latent_diffusion.blocks.timestep import TimestepEmbedder
from latent_diffusion.blocks.siren import SIREN

logger = logging.getLogger(__name__)


def modulate(x: th.Tensor, shift: th.Tensor, scale: th.Tensor) -> th.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: th.Tensor):
        """
        Args:
           x: [B, C, H, W] 4D tensor
        """
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ControlEncoder(nn.Module):
    """Encoder for dense control signal. Downsamples input 16x."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # TODO: should this be more expressive/initialized from somewhere?
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
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
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, out_channels, 3, padding=1),
        )
        # TODO: add initialization?
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv_block(x)


class ControlEncoderLatent(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_size: int = 256):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # prevent any loss of information in the encoder (no hidden_size bottleneck)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(out_channels * 2, out_channels, 1, padding=0),
            LayerNorm2d(out_channels, elementwise_affine=True, eps=1.0e-6),
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv_block(x)


class ZeroControlBlock(nn.Module):
    """
    Produces a dense control map
    """

    def __init__(
        self,
        channels: int,
        cond_size: int,
        zero_init: bool = True,
        init_std: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.cond_size = cond_size

        self.zero_ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_size, 2 * channels, bias=True)
        )
        self.zero_linear = nn.Linear(channels, channels, bias=True)

        if zero_init:
            # zero-adaln
            nn.init.constant_(self.zero_ada_ln[-1].weight, 0)
            nn.init.constant_(self.zero_ada_ln[-1].bias, 0)

            # zero-linear
            nn.init.constant_(self.zero_linear.weight, 0)
            nn.init.constant_(self.zero_linear.bias, 0)
        else:
            # non-zero weights (does it help?)
            nn.init.normal_(self.zero_ada_ln[-1].weight, std=init_std)
            nn.init.constant_(self.zero_ada_ln[-1].bias, 0)

            nn.init.normal_(self.zero_linear.weight, std=init_std)
            nn.init.constant_(self.zero_linear.bias, 0)

    def forward(self, h: th.Tensor, cond: th.Tensor) -> th.Tensor:
        shift, scale = self.zero_ada_ln(cond).chunk(2, dim=-1)
        # print(f"{shift.shape=}, {scale.shape=}, {h.shape=}")
        h = modulate(h, shift, scale)
        h = self.zero_linear(h)
        return h


class ControlMLP(nn.Module):
    """Lightweight ControlNet-style module compatible with DiT architecture."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        depth: int,
        zero_init: bool = True,
        debug: bool = False,
        control_drop_prob: Optional[float] = None,
        control_encoder_latent: bool = False,
        plucker_siren_config: Optional[dict] = None,
        noise_std: float = 0.0,
        normalize_inputs: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.control_encoder = (
            ControlEncoder(in_channels, hidden_size)
            if not control_encoder_latent
            else ControlEncoderLatent(in_channels, hidden_size)
        )

        logger.info(f"{zero_init=}")

        # NOTE: IS THIS IN THE LIST OF PARAMS?
        self.zero_blocks = nn.ModuleList(
            [
                ZeroControlBlock(hidden_size, hidden_size, zero_init)
                for _ in range(depth)
            ]
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.control_drop_prob = control_drop_prob
        self.control_zero_token = nn.Parameter(th.zeros(1, 1, self.hidden_size))
        self.noise_std = noise_std
        self.normalize_inputs = normalize_inputs

        # encode plucker with siren before concatentation
        self.plucker_encoder = None
        if plucker_siren_config is not None:
            self.plucker_encoder = SIREN(**plucker_siren_config)

    def forward(
        self, dense_cond, t, x_t = None, p_drop = 0.0, plucker=None, drop=False
    ) -> List[th.Tensor]:
        """
        Args:
          dense_cond: [B C H W] tensor, dense conditioning signal (such as keypoints)
          t: [B]
        """
        t = self.t_embedder(t)

        # normalize control input (dense_cond)
        if self.normalize_inputs:
            dense_cond = (dense_cond - 127.5) / 127.5

        # encode plucker (used in pippo)
        if plucker is not None:
            # B, NV, T, C, H, W = plucker.shape
            plucker = plucker.squeeze(2)

            if self.plucker_encoder is not None:
                plucker = rearrange(plucker, "B C H W -> B H W C")
                plucker = self.plucker_encoder(plucker)
                plucker = rearrange(plucker, "B H W C -> B C H W")

            # downsample to dense_cond and concat
            if dense_cond is not None:
                downsample_shape = dense_cond.shape[-2:]
            else:
                downsample_shape = (plucker.shape[-2] // 8, plucker.shape[-1] // 8)
            plucker = F.interpolate(plucker, size=downsample_shape, mode="bilinear", antialias=False, align_corners=True)

            # if we only provide plucker
            if dense_cond is not None:
                dense_cond = th.cat([dense_cond, plucker], dim=1)
            else:
                dense_cond = plucker

            # attach x_t to dense_cond
            if x_t is not None:
                try:
                    dense_cond = th.cat([dense_cond, x_t], dim=1)
                except:
                    breakpoint()

        # when not encoding with separate encoder
        if len(dense_cond.shape) == 4:
            h = self.control_encoder(dense_cond)
            h = rearrange(h, "B C H W -> B (H W) C")
        else:
            h = dense_cond

        B, N, C = h.shape
        device = h.device
        drop_mask = None

        # CFG logic but handled at different levels (in pippo vs rest, TODO: unify)
        if p_drop > 0:
            drop_mask = th.rand(size=(B,), device=device) < p_drop
        else:
            if self.training and self.control_drop_prob is not None:
                drop_mask = th.rand(size=(B,), device=device) < self.control_drop_prob
                zero_embs = self.control_zero_token.repeat(B, N, 1).to(h.dtype)

        controls = []
        for block in self.zero_blocks:
            h_out = block(h, t)
            if drop_mask is not None:
                h_out[drop_mask] = zero_embs[drop_mask].clone().to(h_out.dtype)
            elif drop:
                h_out[:] = self.control_zero_token.to(h.dtype)

            if self.training and self.noise_std:
                h_out = h_out + self.noise_std * th.randn_like(h_out)
            controls.append(h_out)

        return controls
