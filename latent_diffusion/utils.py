# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
import copy
import importlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch as th
import numpy as np

from torch import nn
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

logger: logging.Logger = logging.getLogger(__name__)


def load_module(
    module_name: str, class_name: Optional[str] = None, silent: bool = False
):
    """
    Load a module or class given the module/class name.

    Args:
        module_name: str
        The full path of the module relative to the root directory. Ex: ``utils.module_loader``

        class_name: str
        The name of the class within the module to load.

        silent: bool
        If set to True, return None instead of raising an exception if module/class is missing

    Returns:
        object:
        The loaded module or class object.
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        else:
            return module
    except ModuleNotFoundError as e:
        if silent:
            return None
        logger.error(f"Module not found: {module_name}", exc_info=True)
        raise
    except AttributeError as e:
        if silent:
            return None
        logger.error(
            f"Can not locate class: {class_name} in {module_name}.", exc_info=True
        )
        raise


def load_class(class_name: str):
    """
    Load a class given the full class name.

    Args:
        class_name: txt
        The full class name including the full path of the module relative to the root directory.
    Returns:
        A class
    """
    # This is a false-positive, pyre doesn't understand rsplit(..., 1) can only have 1-2 elements
    # pyre-fixme[6]: In call `load_module`, for 1st positional only parameter expected `bool` but got `str`.
    return load_module(*class_name.rsplit(".", 1))


def load_from_config(
    config: Any,
    load_ckpt: bool = True,
    map_location: str = "cpu",
    **kwargs,
):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    ckpt = None if "ckpt" not in config else config.pop("ckpt")
    pretrained_path = (
        None if "pretrained_path" not in config else config.pop("pretrained_path")
    )
    assert not (ckpt is not None and pretrained_path is not None)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    instance = object_class(**config, **kwargs)
    if ckpt is not None and load_ckpt:
        load_checkpoint(
            ckpt_path=ckpt.path,
            modules={ckpt.get("module_name", "model"): instance},
            ignore_names=ckpt.get("ignore_names", None),
            strict=ckpt.get("strict", False),
            ignore_shape_mismatch=ckpt.get("ignore_shape_mismatch", False),
            rename_compiled_ckpt=ckpt.get("rename_compiled_ckpt", False),
            map_location=ckpt.get("map_location", None),
        )
    elif pretrained_path is not None:
        logger.info(f"loading pretrained module from `{pretrained_path}`")
        params = th.load(
            pretrained_path,
            map_location=map_location,
            weights_only=True,
        )
        instance.load_state_dict(params, strict=False)

    return instance


def load_checkpoint(
    ckpt_path,
    modules: Dict[str, Any],
    iteration=None,
    strict=False,
    map_location=None,
    ignore_names=None,
    ignore_shape_mismatch=False,
    rename_compiled_ckpt=False,
    weights_only: bool = False,
):
    """Load a checkpoint.
    Args:
        ckpt_path: directory or the full path to the checkpoint
    """
    if map_location is None:
        map_location = "cpu"
    # adding
    if os.path.isdir(ckpt_path):
        if iteration is None:
            # lookup latest iteration
            iteration = max(
                [
                    int(os.path.splitext(os.path.basename(p))[0])
                    for p in glob.glob(os.path.join(ckpt_path, "*.pt"))
                ]
            )
        ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
    logger.info(f"loading checkpoint `{ckpt_path}`")
    ckpt_dict = th.load(ckpt_path, map_location=map_location, weights_only=weights_only)

    for name, mod in modules.items():
        if name not in ckpt_dict and not strict:
            logger.info(f"`{name}` not in checkpoint, skipping")
            continue
        # pre-processing for compiled checkpoints
        ckpt_dict[name] = {
            k.replace("_orig_mod.", ""): v for k, v in ckpt_dict[name].items()
        }

        params = ckpt_dict[name]
        if ignore_names is not None:
            if OmegaConf.is_dict(ignore_names) and name in ignore_names:
                logger.info(f"skipping: {ignore_names[name]}")
                params = filter_params(params, ignore_names[name])
            elif OmegaConf.is_list(ignore_names):
                params = filter_params_list(params, ignore_names)
            else:
                raise ValueError(f"Unknown ignore_names type: {type(ignore_names)}")

        if isinstance(mod, th.optim.Optimizer):
            # load parameters into optimizer
            try:
                mod.load_state_dict(params)
            except:
                print(
                    f"Optimizer loaded from this checkpoint has different set of trainable parameters than current training run! Skipping loading optimizer state dict."
                )
        else:
            # rewrite compiled checkpoint
            if rename_compiled_ckpt:
                logger.info("rewriting keys in a compiled checkpoint")
                param_names = list(params.keys())
                for name in param_names:
                    prefix = "_orig_mod."
                    if name.startswith(prefix):
                        params[name.replace(prefix,"")] = params.pop(name)

            # prune params with shape mismatch
            if ignore_shape_mismatch:
                for name, param in mod.named_parameters():
                    if name in params and param.shape != params[name].shape:
                        logger.warning(
                            f"skipping {name} with shape mismatch: {param.shape} vs {params[name].shape}"
                        )
                        del params[name]

            # load parameters into model
            logger.warning("FIXME: lr_scheduler vs nn.Module (!)")
            try:
                mod.load_state_dict(params, strict=strict)
            except:
                mod.load_state_dict(params)

    return ckpt_dict


def build_optimizer(config, model, use_zero: bool = False):
    """Build an optimizer given optimizer config and a model.

    Args:
        config: DictConfig
        model: nn.Module|Dict[str,nn.Module]

    """
    config = copy.deepcopy(config)

    if isinstance(model, nn.Module):
        if "per_module" in config:
            params = []
            for name, value in config.per_module.items():
                if not hasattr(model, name) or getattr(model, name) is None:
                    logger.warning(
                        f"model {model.__class__} does not have an initialized submodule {name}, skipping"
                    )
                    continue

                try:
                    params.append(
                        dict(
                            params=getattr(model, name).parameters(),
                            **value,
                        )
                    )
                except Exception as e:
                    params.append(
                        dict(
                            params=getattr(model, name),
                            **value,
                        )
                    )

            defined_names = set(config.per_module.keys())
            for name, module in model.named_children():
                n_params = len(list(module.named_parameters()))
                if name not in defined_names and n_params:
                    logger.warning(
                        f"not going to optimize module {name} which has {n_params} parameters"
                    )
            config.pop("per_module")
        else:
            params = model.parameters()
    else:
        # NOTE: can we do
        assert "per_module" in config
        assert isinstance(model, dict)
        for name, value in config.per_module.items():
            params = []
            for name, value in config.per_module.items():
                if name not in model:
                    logger.warning(f"not aware of {name}, skipping")
                    continue
                params.append(
                    dict(
                        params=model[name].parameters(),
                        **value,
                    )
                )

    return load_from_config(config, params=params)


def save_checkpoint(
    ckpt_path,
    modules: Dict[str, Any],
    iteration: Optional[int] = None,
    fsdp: bool = False,
    rank: Optional[int] = None,
    **kwargs,
):
    if not fsdp or not rank:
        ckpt_dict = {}

        if os.path.isdir(ckpt_path):
            assert iteration is not None
            ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")

        if iteration is not None:
            ckpt_dict["iteration"] = iteration
        ckpt_dict.update(kwargs)

        for name, mod in modules.items():
            if hasattr(mod, "module"):
                mod = mod.module
            if hasattr(mod, "state_dict"):
                ckpt_dict[name] = mod.state_dict()
            else:
                ckpt_dict[name] = mod

        th.save(ckpt_dict, ckpt_path)
    elif fsdp:
        raise NotImplementedError()


def filter_params(params, ignore_names):
    return OrderedDict(
        [
            (k, v)
            for k, v in params.items()
            if not any([re.match(n, k) is not None for n in ignore_names])
        ]
    )


def filter_params_list(params, ignore_names):
    filtered_params = OrderedDict()
    for name, param in params.items():
        skip = False

        # check if this parameter should be ignored
        for ignore_name in ignore_names:
            if ignore_name in name:
                logger.info(f"skipping pretrained param: {name}")
                skip = True
                break

        if not skip:
            filtered_params[name] = param

    return filtered_params


def count_parameters(model, trainable: bool = True):
    return np.sum(
        [p.numel() for p in model.parameters() if p.requires_grad == trainable]
    )


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, th.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = to_device(v, device)
    return batch


def process_losses(loss_dict, reduce=True, detach=True):
    """Preprocess the dict of losses outputs."""
    result = {
        k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")
    }
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result
