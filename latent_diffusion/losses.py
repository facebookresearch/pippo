# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DenoisingLoss(nn.Module):
    def forward(self, preds):
        loss_l2 = (preds["noise_preds"] - preds["noise"]).pow(2.0).mean()
        loss_dict = {
            "loss_total": loss_l2,
            "loss_l2": loss_l2,
        }
        return loss_l2, loss_dict
