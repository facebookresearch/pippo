import glob
import logging
import os
import sys
import time
from copy import deepcopy
from itertools import cycle

import torch as th
from torch import nn

import torch.multiprocessing as mp
import torchvision


from latent_diffusion.utils import build_optimizer, load_from_config, load_checkpoint, save_checkpoint, count_parameters, process_losses

from latent_diffusion.data import load_batches

from omegaconf import DictConfig, OmegaConf

# configure logger before importing anything else
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

mp.set_start_method("spawn", force=True)
torchvision.disable_beta_transforms_warning()


def train(config):
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.autograd.set_detect_anomaly(config.train.get("detect_anomaly", False))

    # load and initialize the model
    device = th.device(f"cuda")
    model = load_from_config(config.model)
    model = model.to(device)
    if hasattr(model, "post_init"):
        model.post_init()

    # load optimizer and loss function
    optimizer = build_optimizer(config.optimizer, model)
    loss_fn = load_from_config(config.loss).to(device)
    iteration, epoch = 0, 0

    # resume from checkpoint
    ckpt_path = config.train.get("ckpt_path", None)
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            ckpt_dict = load_checkpoint(
                ckpt_path,
                modules={
                    "model": model,
                    "optimizer": optimizer,
                },
            )
            iteration = ckpt_dict.get("iteration", None)
            epoch = ckpt_dict.get("epoch", 0)
        else:
            logger.warning(f"checkpoint `{ckpt_path}` does not exist for restart")

    # load the dataset
    train_data = load_batches(batch_size=4, num_views=config.consts.num_views, resolution=config.consts.img_size, num_samples=1)
    train_loader = val_loader = cycle(train_data)

    # summary function
    summary_fn = load_from_config(config.summary)

    # count parameters
    logger.info(
        f"total # of trainable parameters: {(count_parameters(model) / 1e6):.1f}M"
    )

    # train loop
    while True:
        batch = next(train_loader)
        for k,v in batch.items():
            if isinstance(v, th.Tensor):
                batch[k] = v.to(device)

        # forward pass
        use_amp = config.train.get("use_amp", False)
        with th.autocast(device_type="cuda", dtype=th.bfloat16, enabled=use_amp):
            preds = model(**batch)
            loss, loss_dict = loss_fn(preds)

        # backward pass
        loss.backward()

        # debugging info
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        unscaled_grad_norm = float(th.cat(grads).norm())

        if "clip_grad_norm" in config.train:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # log iteration stats
        log_now = iteration % config.train.log_every_n_steps == 0
        if log_now:
            _loss_dict = process_losses(loss_dict)
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
            logger.info(f"#{iteration} Loss: {loss_str}")

        # save checkpoint
        save_now = iteration % config.train.get("persistent_ckpt_every_n_steps", 1e6) == 0 and iteration > 0
        if save_now:
            os.makedirs(config.train.ckpt_dir, exist_ok=True)
            ckpt_path = f"{config.train.ckpt_dir}/{iteration:06d}.pt"
            logger.info(
                f"iter={iteration}: saving *persistent* checkpoint to `{ckpt_path}`"
            )
            ckpt_dict = {
                "model": getattr(model, "module", model),
                "optimizer": optimizer,
            }
            save_checkpoint(ckpt_path, ckpt_dict, iteration=iteration)

        # evaluate on validation set and saves visuals
        eval_now = iteration % config.train.summary_every_n_steps == 0 and iteration > 0
        if eval_now:
            model.eval()
            with th.no_grad():
                summary_args = dict(
                    preds=preds,
                    model=model,
                    val=val_loader,
                    train=train_loader,
                    iteration=iteration,
                    run_dir=config.train.run_dir,
                    config=config,
                )
                summaries = summary_fn(**summary_args)
            model.train()

        # update counters
        iteration += 1
        if iteration >= config.train.n_max_iters:
            logger.info(f"reached max number of iters ({config.train.n_max_iters})")
            break


if __name__ == "__main__":
    config_path = str(sys.argv[1])
    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])
    if config_cli:
        config = OmegaConf.merge(config, config_cli)
    if "USER" not in config:
        config.USER = os.environ["USER"]
    train(config)

"""
Usage:

CUDA_VISIBLE_DEVICES=2 python /home/$USER/rsc/latent_diffusion/latent_diffusion/overfit.py \
  /home/yashkant/rsc/latent_diffusion/config/multiview/release/sstk_with_ava.yml

"""
