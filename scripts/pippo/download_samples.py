# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from huggingface_hub import HfApi, snapshot_download

# download models and cache (from hub)
local_dir = os.getcwd().split("scripts/")[0]

snapshot_download(
    repo_id="yashkant/pippo",
    local_dir=local_dir,
    repo_type="dataset",
    local_dir_use_symlinks=False,
)
