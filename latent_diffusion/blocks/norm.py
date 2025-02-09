# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

from torch.nn import RMSNorm as BaseRMSNorm

class LayerNorm(nn.LayerNorm):
    """LayerNorm.

    By default, casts input to float32.
    """

    def __init__(self, *args, cast_dtype=th.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.cast_dtype = cast_dtype

    def forward(self, x):
        in_type = x.dtype
        # TODO: figure out a minimal version of this
        with th.autocast(device_type="cuda", enabled=False):
            x = super().forward(x.type(self.cast_dtype))
        return x.type(in_type)


class RMSNorm(BaseRMSNorm):
    """RMSNorm with a scaling parameter.

    By default, casts input to float32.

    # TODO: check if casting is necessary for TE IMPL
    """

    def __init__(self, *args, cast_dtype=th.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.cast_dtype = cast_dtype

    @th.compiler.disable()
    def forward(self, x):
        in_type = x.dtype
        # TODO: figure out a minimal version of this
        with th.autocast(device_type="cuda", enabled=False):
            x = super().forward(x.type(self.cast_dtype).contiguous())
        return x.type(in_type)
