# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch as th
import torch.nn as nn
import math


def timestep_embedding(
    t,
    dim: int,
    max_period: float = 10000,
    time_factor: Optional[float] = 1.0,
) -> th.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: a 1-D Tensor of N indices or timesteps, one per batch element.
        dim: the dimension of the output
        max_period: controls the minimum frequency of the embeddings
    Returns:
        (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t.float()
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=t.device)
    args = t[:, None] * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        time_factor: Optional[float] = 1.0,
    ):
        """
        Args:

        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.time_factor = time_factor

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t: th.Tensor) -> th.Tensor:
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, time_factor=self.time_factor
        )
        t_emb = self.mlp(t_freq)
        return t_emb
