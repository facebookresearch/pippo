# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch as th
import torch.nn as nn
import numpy as np
import imageio
import time
import os

import torch.nn.functional as F
from einops import rearrange, repeat
from latent_diffusion.models.control_mlp import ControlMLP
from latent_diffusion.models.dit import DiT
from latent_diffusion.blocks.patch import PatchifyPS
from latent_diffusion.blocks.pos_embed import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)
from latent_diffusion.blocks.ref_mlp import Ref2DMLP, RefMLP

from latent_diffusion.schedulers.diffusion import DiffusionSchedule as Schedule
from latent_diffusion.utils import load_from_config, to_device
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
th.set_float32_matmul_precision("highest")


class RefEncoder(nn.Module):
    """Encodes reference image with separate patchify and pos_embeds."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        in_channels: int,
        patch_size: int,
        dim: int = 768,
        init_zero_last: bool = False,
    ):
        super().__init__()

        self.patchify = PatchifyPS(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=dim,
            flatten=True,
        )
        self.pos_embed = nn.Parameter(
            th.zeros(1, self.patchify.num_patches, dim), requires_grad=True
        )

        if init_zero_last:
            self.last = nn.Linear(dim, dim)
            nn.init.constant_(self.last.weight, 0)
            nn.init.constant_(self.last.bias, 0)
        else:
            self.last = nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patchify.num_patches**0.5),
            return_tensor=True,
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(self, x: th.Tensor) -> th.Tensor:
        B = x.shape[0]
        device = x.device
        x = self.patchify(x)
        cond_attn = x + self.pos_embed
        cond_attn = self.last(cond_attn)
        return cond_attn


class Pippo(nn.Module):
    """ Pippo: Multiview Diffusion Transformer with ControlMLP"""

    def __init__(
        self,
        compressor: Mapping[str, Any],
        generator: Mapping[str, Any],
        schedule: Mapping[str, Any],
        multiview_size: int,  # size of each view in multiview
        img_size: int,  # size of image being denoised
        ref_img_size: int,  # size of reference view
        ref_encoder: Mapping[str, Any] = None,
        kpts_encoder: Mapping[str, Any] = None,
        face_cropper: Optional[Mapping[str, Any]] = None,  # encode faces separately
        control_mlp: Mapping[str, Any] = None, # encode spatial controls with MLP
        parameterization: str = "eps",
        scale_factor: float = 1.0,
        shift_factor: float = 0.0,
        freeze_compressor: bool = True,
        noise_offset: Optional[float] = None,
        ref_cond_attn: bool = False,  # reference image conditioning
        cam_mlp: Optional[Mapping[str, Any]] = None,  # encode camera pose with MLP
        plucker_mlp: Optional[
            Mapping[str, Any]
        ] = None,  # encode plucker camera rays with MLP
        p_drop_ref_img: float = 0.0,  # probability of dropping reference image for CFG
        p_drop_cam_pos_enc: float = 0.0,  # probability of dropping camera positional encodings for CFG
        p_drop_spatial: float = 0.0,  # probability of dropping all spatial signals (cam pose, kpts, plucker) for CFG
        p_drop_all_cond: float = 0.0,  # probability of dropping all conditioning signals (spatial + ref) for CFG
        multiview_grid_stack: bool = True,  # multiview images are NxN grid, True for backward compatibility
        num_views: int = 4,  # number of multiview images
        num_ref_views: int = 1,  # number of multiview images
        num_frames: int = 1,  # number of frames in video
        num_ref_frames: int = 1,  # number of frames in reference video
        multiview_pos_embed: bool = False,  # add separate positional encoding for multiview images
        cond_cat_kpts: bool = True,  # concatenate kpts instead of summing
        cond_cat_split_attn: bool = False,  # kpts conditioning via split and attention
        kpts_encode: bool = True,  # encode kpts via VAE
        null_cond_spatial: bool = False,  # use null conditioning for spatial signals
        p_drop_control: float = 0.0,  # probability of dropping controlmlp inputs (keypoints / plucker)
        run_post_init: bool = False,  # run post init after creating the model
        controlmlp_xt: bool = False, # add x_t into controlmlp
    ):
        super().__init__()


        # vae compressor
        self.compressor = AutoencoderKL.from_pretrained(**compressor)
        self.compressor = self.compressor.eval()
        for param in self.compressor.parameters():
            param.requires_grad = False

        # diffusion transformer
        self.generator = DiT(**generator)

        self.parameterization = parameterization
        if "class_name" not in schedule:
            logger.warning(
                "`class_name` not specified, using default DiffusionSchedule"
            )
            self.schedule = Schedule(**schedule)
            self.flow_mode = False
        else:
            self.schedule = load_from_config(schedule)
            self.flow_mode = True

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.control_mlp = None if control_mlp is None else ControlMLP(**control_mlp)
        self.control_encoder_latent = (
            False
            if control_mlp is None
            else control_mlp.get("control_encoder_latent", False)
        )

        self.noise_offset = noise_offset

        # (YK): changes for multiview below
        self.img_size = img_size
        self.ref_img_size = ref_img_size
        self.multiview_size = multiview_size

        self.ref_cond_attn = ref_cond_attn
        self.multiview_grid_stack = multiview_grid_stack

        # CFG scales for different conditionings
        self.p_drop_ref_img = p_drop_ref_img
        self.p_drop_cam_pos_enc = p_drop_cam_pos_enc
        self.p_drop_spatial = p_drop_spatial
        self.p_drop_all_cond = p_drop_all_cond

        self.num_views = num_views
        self.num_ref_views = num_ref_views
        self.num_frames = num_frames
        self.num_ref_frames = num_ref_frames
        self.cond_cat_kpts = cond_cat_kpts
        self.cond_cat_split_attn = cond_cat_split_attn
        self.kpts_encode = kpts_encode

        # controlmlp
        self.p_drop_control = p_drop_control
        self.controlmlp_xt = controlmlp_xt

        # encode flattened camera params with mlp
        self.cam_mlp = None if cam_mlp is None else RefMLP(**cam_mlp)
        self.plucker_mlp = None if plucker_mlp is None else Ref2DMLP(**plucker_mlp)

        # encode reference image with separate module
        self.ref_encoder = None
        if ref_encoder is not None:
            self.ref_encoder = RefEncoder(**ref_encoder)

        # crop face in the reference image separately
        self.face_cropper = None
        if face_cropper is not None:
            # TODO: replace with any lightweight face cropper
            self.face_cropper = FaceCropper(**face_cropper)

        # encode reference image with separate module (#TODO: try Ref2DMLP)
        self.kpts_encoder = None
        if kpts_encoder is not None:
            self.kpts_encoder = RefEncoder(**kpts_encoder)

        # classifier guidance null conds (expanded shape: [B, NV, N, D])
        self.num_ref_patches = 1
        if null_cond_spatial:
            self.num_ref_patches = self.ref_encoder.patchify.num_patches
        null_dict = {
            "ref": nn.Parameter(th.zeros(1, self.num_ref_patches, generator.dim)),
        }

        if self.kpts_encoder is not None:
            self.num_kpts_patches = self.kpts_encoder.patchify.num_patches
            null_dict["kpts"] = nn.Parameter(
                th.zeros(1, self.num_kpts_patches, generator.dim)
            )

        if self.cam_mlp is not None:
            null_dict["cam_pose"] = nn.Parameter(th.zeros(1, 1, generator.dim))

        if self.plucker_mlp is not None:
            self.num_plucker_patches = self.plucker_mlp.patchify.num_patches
            null_dict["plucker"] = nn.Parameter(
                th.zeros(1, self.num_plucker_patches, generator.dim)
            )

        # add separate positional encoding for multiviews
        self.multiview_pos_embed = None

        # null cfg conds
        self.null_conds = nn.ParameterDict(null_dict)

        # post init during first forward pass
        self.post_initialized = False
        if run_post_init:
            self.post_init()

        logger.info(f"model level precision: {th.get_float32_matmul_precision()}")

    def post_init(self):
        # modifications which cannot be made during initialization
        # as they would break backward compatibility (pretrained weights loading)
        if not self.post_initialized:
            if "ar_ref" in self.null_conds:
                self.null_conds["ar_ref"] = self.null_conds["ar_ref"][None]
            if "ref" in self.null_conds:
                self.null_conds["ref"] = self.null_conds["ref"][None]
            logger.info("ran post initialization")
            self.post_initialized = True

    def predict_noise(self, x_t, o_t, t):
        """eps from model prediction."""
        if self.parameterization == "eps":
            e_t = o_t
        elif self.parameterization == "v":
            e_t = self.schedule.predict_noise_from_v(x_t, o_t, t)
        else:
            raise ValueError("unsupported parameterization: ")
        return e_t

    def _add_fixed_noise(
        self, x_0: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        device = x_0.device
        B, C = x_0.shape[:2]
        # TODO: move this to schedule?
        T = self.schedule.num_timesteps
        t = th.linspace(1.0, T - 1.0, B).to(device=device, dtype=th.int64)
        noise = th.randn_like(x_0)
        if self.noise_offset is not None:
            noise = noise + self.noise_offset * th.randn(
                (B, C, 1, 1), dtype=x_0.dtype, device=device
            )
        x_t = self.schedule.q_sample(x_0, t, noise)
        return x_t, t, noise

    def predict_x_0(self, x_t, o_t, t):
        if self.parameterization == "eps":
            x_0_pred = self.schedule.predict_x_0(x_t, t, o_t)
        elif self.parameterization == "v":
            x_0_pred = self.schedule.predict_x_0_from_v(x_t=x_t, v=o_t, t=t)
        else:
            raise ValueError("unsupported parameterization: ")
        return x_0_pred

    def predict(
        self,
        x_t: th.Tensor,
        t: th.Tensor,
        conds: Dict[str, th.Tensor],
    ):
        return self.generator(x_t, t, **conds)

    def encode_image(self, image: th.Tensor) -> th.Tensor:
        # encode image with compressor
        image_norm = (image - 127.5) / 127.5
        if isinstance(self.compressor, AutoencoderKL):
            encode_fn = lambda x: self.compressor.encode(x).latent_dist.sample()
        else:
            encode_fn = lambda x: self.compressor.encode(x)

        try:
            x_0 = (
                encode_fn(image_norm) - self.shift_factor
            ) * self.scale_factor
        except:
            # chunk VAE to avoid OOM
            slices = th.split(image_norm, 2, dim=0)
            slices_enc = [encode_fn(x) for x in slices]
            enc = th.cat(slices_enc, dim=0)
            x_0 = (enc - self.shift_factor) * self.scale_factor

        return x_0

    def decode_image(self, latent: th.Tensor) -> th.Tensor:
        # decode image with compressor
        if isinstance(self.compressor, AutoencoderKL):
            decode_fn = lambda x: self.compressor.decode(x).sample
        else:
            decode_fn = lambda x: self.compressor.decode(x)

        image_norm = decode_fn(
            latent / self.scale_factor + self.shift_factor
        )
        image = image_norm * 127.5 + 127.5
        return image

    @staticmethod
    def encode_kpts(cls, kpts, B, p_drop=0, drop_mask=None):
        self = cls
        # encode kpts in latent or directly use rgb
        if self.kpts_encode:
            with th.no_grad():
                kpts_latent = self.encode_image(kpts)
        else:
            kpts_latent = (kpts - 127.5) / 127.5
        kpts_patches = self.kpts_encoder(kpts_latent)

        # replace with cfg null conds
        if p_drop > 0 or drop_mask is not None:
            device = kpts.device
            if drop_mask is None:
                drop_mask = th.rand(size=(B,), device=device) < p_drop
            kpts_patches = rearrange(kpts_patches, "(B X) N D -> B (X N) D", B=B)
            kpts_patches = th.where(
                drop_mask[:, None, None], self.null_conds["kpts"], kpts_patches
            )
            kpts_patches = rearrange(kpts_patches, "B (X N) D -> (B X) N D", N=N)

        return kpts_patches

    @staticmethod
    def encode_ref_image(
        cls,
        ref_image,
        ref_cam_pose,
        p_drop,
        return_cam_pos_enc=False,
        drop_mask=None,  # replaces the drop mask
        joint_drop_mask=None,  # gets acculumated with drop_mask
        encode_face=False, # encode only face
    ) -> th.Tensor:
        self = cls  # TODO: dirty, fix later

        # extract face from reference image
        if encode_face:
            ref_face = self.face_cropper(ref_image)
            ref_face = rearrange(ref_face, "B H W C -> B C H W")
            ref_face = th.clamp((ref_face + 1) * 127.5, min=0, max=255).to(th.uint8)

            # replace reference image with face
            assert ref_face.shape == ref_image.shape
            ref_image = ref_face

        with th.no_grad():
            ref_latent = self.encode_image(ref_image)

        # encode reference image
        if self.ref_encoder is not None:
            ref_patches = self.ref_encoder(ref_latent)
        else:
            # backward compatibility
            generator = (
                getattr(self, "unet")
                if hasattr(self, "unet")
                else getattr(self, "generator")
            )

            ref_patches = generator.patchify(ref_latent)
            # add positional encoding
            ref_patches = ref_patches + generator.pos_embed[:, : ref_patches.shape[1]]

        # encode reference camera pose
        ref_cam_pos_enc = None
        if self.cam_mlp is not None:
            ref_cam_pos_enc = self.cam_mlp(ref_cam_pose)[-1]

            # add camera pose encoding
            H = W = int(ref_patches.shape[1] ** 0.5)
            ref_cam_pos_enc = repeat(ref_cam_pos_enc, "B C -> B N C", N=H * W)
            ref_patches = ref_patches + ref_cam_pos_enc

        # replace with cfg null conds
        if p_drop > 0 or drop_mask is not None or joint_drop_mask is not None:
            B = ref_patches.shape[0] // (self.num_ref_views * self.num_ref_frames)
            N = ref_patches.shape[1]

            # cfg dropout works sample level
            if drop_mask is None:
                drop_mask = th.rand(size=(B,), device=ref_image.device) < p_drop

            # accumulate with joint_drop_mask
            if joint_drop_mask is not None:
                drop_mask = th.logical_or(drop_mask, joint_drop_mask)

            ref_patches = rearrange(ref_patches, "(B X) N D -> B X N D", B=B)
            drop_mask = drop_mask[:, None, None, None]
            assert len(drop_mask.shape) == len(self.null_conds["ref"].shape) == len(ref_patches.shape)
            ref_patches = th.where(
                drop_mask, self.null_conds["ref"], ref_patches
            )
            ref_patches = rearrange(ref_patches, "B X N D -> (B X) N D", N=N)

        if not self.multiview_grid_stack:
            # only place where we currently distinguish between image or video
            # TODO:  will be removed when we merge video and image models
            if isinstance(self, Pippo):
                NV_list, expand_time = [self.num_ref_views], False
            else:
                NV_list, expand_time = [(self.num_ref_views, self.num_ref_frames)], True

            ref_patches = Pippo._unpreprocess(
                self, data_list=[ref_patches], NV_list=NV_list, expand_time=expand_time
            )

        if return_cam_pos_enc:
            return ref_patches, ref_cam_pos_enc

        return ref_patches

    @staticmethod
    def get_cam_pos_enc(
        cls,
        cam_pose,
        p_drop=0,
        NV=None,
        drop_mask=None,
    ) -> th.Tensor:
        """Created positional encoding from camera pose"""
        self = cls  # TODO: dirty, fix later
        cam_pos_enc = self.cam_mlp(cam_pose)[-1]

        # V1, V2 and V3 compatibility
        if hasattr(self, "generator"):
            num_patches = self.generator.patchify.num_patches
        else:
            if hasattr(self.unet, "x_embedder"):
                num_patches = self.unet.x_embedder.num_patches
            else:
                # dit4d model
                num_patches = self.unet.patchify.num_patches

        # flexibly handle different number of views
        NV = NV if NV is not None else self.num_views
        H = W = int(num_patches**0.5)
        cam_pos_enc = repeat(cam_pos_enc, "B C -> B (H W) C", H=H, W=W)

        if p_drop > 0 or drop_mask is not None:
            B = cam_pose.shape[0] // (self.num_views * self.num_frames)
            N = cam_pos_enc.shape[1]
            device = cam_pose.device
            if drop_mask is None:
                drop_mask = th.rand(size=(B,), device=device) < p_drop
            cam_pos_enc = rearrange(cam_pos_enc, "(B X) N D -> B (X N) D", B=B)
            cam_pos_enc = th.where(
                drop_mask[:, None, None], self.null_conds["cam_pose"], cam_pos_enc
            )
            cam_pos_enc = rearrange(cam_pos_enc, "B (X N) D -> (B X) N D", N=N)

        if not self.multiview_grid_stack:
            # only place where we currently distinguish between image or video
            # TODO:  will be removed when we merge video and image models
            if isinstance(self, Pippo):
                NV_list, expand_time = [NV], False
                cam_pos_enc = Pippo._unpreprocess(
                    self,
                    data_list=[cam_pos_enc],
                    NV_list=NV_list,
                    expand_time=expand_time,
                )
            else:
                # cameras don't have time dimension
                NV_list, expand_time = [(NV, 1)], True
                cam_pos_enc = Pippo._unpreprocess(
                    self,
                    data_list=[cam_pos_enc],
                    NV_list=NV_list,
                    expand_time=expand_time,
                )
                # video model assumes pos_enc_x has same shape as x
                cam_pos_enc = repeat(
                    cam_pos_enc, "B NV 1 ... -> B NV T ...", T=self.num_frames
                )
        else:
            # only place where we currently distinguish between image or video
            if not isinstance(self, Pippo):
                # video model assumes pos_enc_x has same shape as x
                cam_pos_enc = repeat(cam_pos_enc, "B N D -> B T N D", T=self.num_frames)

        return cam_pos_enc

    @staticmethod
    def get_plucker_enc(
        cls,
        plucker: th.Tensor,
        p_drop=0,
        NV=None,
        drop_mask=None,
    ) -> th.Tensor:
        """Created positional encoding from plucker coordinates"""
        self = cls  # TODO: dirty, fix later

        B, NV, T, C, H, W = plucker.shape
        plucker = rearrange(plucker, "B NV T C H W -> (B NV T) C H W")
        plucker_pos_enc = self.plucker_mlp(plucker)
        plucker_pos_enc = rearrange(
            plucker_pos_enc, "(B NV T) ...-> B NV T ...", B=B, NV=NV, T=T
        )

        if p_drop > 0 or drop_mask is not None:
            plucker_pos_enc = rearrange(plucker_pos_enc, "B NV T ...-> B (NV T) ...")
            B, device = plucker_pos_enc.shape[0], plucker_pos_enc.device
            if drop_mask is None:
                drop_mask = th.rand(size=(B,), device=device) < p_drop
            plucker_pos_enc = th.where(
                drop_mask[:, None, None, None],
                self.null_conds["plucker"],
                plucker_pos_enc,
            )
            plucker_pos_enc = rearrange(
                plucker_pos_enc, "B (NV T) ... -> B NV T ...", NV=NV
            )

        # only place where we currently distinguish between image or video
        # TODO:  will be removed when we merge video and image models
        if isinstance(self, Pippo):
            plucker_pos_enc = plucker_pos_enc.squeeze(2)
        else:
            # video model assumes pos_enc_x has same shape as x
            plucker_pos_enc = plucker_pos_enc.repeat(self.num_frames)

        return plucker_pos_enc

    def get_conds(
        self,
        image: th.Tensor = None,
        kpts: Optional[th.Tensor] = None,
        t: Optional[th.Tensor] = None,
        x_t: Optional[th.Tensor] = None,
        text: Optional[List[str]] = None,
        drop_clip: bool = False,
        drop_dino: bool = False,
        ref_image: Optional[th.Tensor] = None,
        ref_kpts: Optional[th.Tensor] = None,
        ref_cam_pose: Optional[th.Tensor] = None,
        cam_pose: Optional[th.Tensor] = None,
        plucker: Optional[th.Tensor] = None,
        ref_plucker: Optional[th.Tensor] = None,
        noisy_views_mask: Optional[th.Tensor] = None, # autoregressive
        noisy_timestep_ids: Optional[th.Tensor] = None, # autoregressive
        training: bool = True,
        attn_bias: float = None,
    ) -> Dict[str, th.Tensor]:

        # post initialization
        self.post_init()

        try:
            # needed when called from sampler (during inference)
            # squash time dimension in cam poses (ref and views), and flatten (encode independently)
            if len(ref_image.shape) > 4:
                data_list = [image, ref_image, cam_pose, ref_cam_pose]
                NV_list = [
                    self.num_views,
                    self.num_ref_views,
                    self.num_views,
                    self.num_ref_views,
                ]
                image, ref_image, cam_pose, ref_cam_pose = (
                    Pippo._preprocess(
                        self, data_list=data_list, NV_list=NV_list, squash_time=True
                    )
                )
        except:
            breakpoint()

        # drop all conditions jointly (and at same places)
        drop_mask = None
        B = ref_image.shape[0]
        device = ref_image.device
        if (self.p_drop_all_cond > 0) and training:
            drop_mask = th.rand(size=(B,), device=device) < self.p_drop_all_cond

        # attention based image conditioning
        cond_attn = None
        if self.ref_cond_attn:
            p_drop = self.p_drop_ref_img if training else 0.0

            if self.face_cropper is not None:
                # face and ref joint drop mask
                fr_drop_mask = th.rand(size=(B,), device=ref_image.device) < p_drop

                # encode only face
                ref_image_attn, ref_cam_pose_enc = self.encode_ref_image(
                    self,
                    ref_image,
                    ref_cam_pose,
                    p_drop=p_drop,
                    return_cam_pos_enc=True,
                    drop_mask=fr_drop_mask,
                    encode_face=True
                )
                cond_attn = ref_image_attn
                if ref_cam_pose_enc is not None:
                    ref_cam_pose_enc = ref_cam_pose_enc[:, None]

                # encode full image and attach
                if not self.replace_ref_w_face:
                    full_ref_image_attn, full_ref_cam_pose_enc = self.encode_ref_image(
                        self,
                        ref_image,
                        ref_cam_pose,
                        p_drop=p_drop,
                        return_cam_pos_enc=True,
                        drop_mask=fr_drop_mask,
                    )

                    # merge both along number of reference views
                    cond_attn = th.cat([cond_attn, full_ref_image_attn], dim=1)

                    if ref_cam_pose_enc is not None:
                        full_ref_cam_pose_enc = full_ref_cam_pose_enc[:, None]
                        ref_cam_pose_enc = th.cat([ref_cam_pose_enc, full_ref_cam_pose_enc], dim=1)
            else:
                ref_image_attn, ref_cam_pose_enc = self.encode_ref_image(
                    self,
                    ref_image,
                    ref_cam_pose,
                    p_drop=p_drop,
                    return_cam_pos_enc=True,
                    drop_mask=drop_mask,
                )
                cond_attn = ref_image_attn

                if ref_cam_pose_enc is not None:
                    ref_cam_pose_enc = ref_cam_pose_enc[:, None]

        # camera based positional encoding
        pos_enc_x = None
        if self.cam_mlp is not None:
            p_drop = self.p_drop_cam_pos_enc if training else 0.0
            pos_enc_x = self.get_cam_pos_enc(
                self, cam_pose, p_drop=p_drop, drop_mask=drop_mask
            )

        # plucker grid based positional encoding
        if self.plucker_mlp is not None:
            plucker_pos_enc_x = self.get_plucker_enc(
                self, plucker=plucker, drop_mask=drop_mask
            )
            pos_enc_x = (
                pos_enc_x + plucker_pos_enc_x
                if pos_enc_x is not None
                else plucker_pos_enc_x
            )

        # keypoints conditioning
        cond_cat, cond_cat_split_attn = None, None
        if self.kpts_encoder is not None:
            # handle generation keypoints
            if kpts is not None:
                B = kpts.shape[0]
                kpts = Pippo._preprocess(
                    self, data_list=[kpts], NV_list=[self.num_views], squash_time=True
                )
                x_kpts = self.encode_kpts(self, kpts, B=B, drop_mask=drop_mask)
                kpts, x_kpts = Pippo._unpreprocess(
                    self, data_list=[kpts, x_kpts], NV_list=[self.num_views] * 2
                )  # [B, NV, C, H, W]

                # split attention
                if self.cond_cat_split_attn:
                    PH = PW = int(pos_enc_x.shape[-2] ** 0.5)
                    KH = KW = int(x_kpts.shape[-2] ** 0.5)
                    pos_enc_x_interp = rearrange(
                        pos_enc_x,
                        "B NV (H W) C -> (B NV) C H W",
                        H=PH,
                        W=PW,
                        NV=self.num_views,
                    )
                    # doensn't matter which interpolation mode we use (because all the posenc values are the same in HxW dimension)
                    pos_enc_x_interp = F.interpolate(
                        pos_enc_x_interp,
                        size=(KH, KW),
                        mode="bilinear",
                        antialias=False,
                        align_corners=True,
                    )
                    pos_enc_x_interp = rearrange(
                        pos_enc_x_interp,
                        "(B NV) C H W -> B NV (H W) C",
                        NV=self.num_views,
                    )
                    cond_cat_split_attn = x_kpts + pos_enc_x_interp

                # add to pos_enc_x or cond_cat
                elif not self.cond_cat_kpts:
                    if pos_enc_x is not None:
                        pos_enc_x += x_kpts
                    else:
                        pos_enc_x = x_kpts

                else:
                    cond_cat = x_kpts

            # handle reference keypoints
            if ref_kpts is not None:
                B = ref_kpts.shape[0]
                ref_kpts = Pippo._preprocess(
                    self,
                    data_list=[ref_kpts],
                    NV_list=[self.num_ref_views],
                    squash_time=True,
                )
                x_ref_kpts = self.encode_kpts(self, ref_kpts, B=B)
                ref_kpts, x_ref_kpts = Pippo._unpreprocess(
                    self,
                    data_list=[ref_kpts, x_ref_kpts],
                    NV_list=[self.num_ref_views] * 2,
                )  # [B, NV, C, H, W]

                # concatenate for split attention (x_kpts and x_ref_kpts)
                if self.cond_cat_split_attn:
                    if cond_cat_split_attn is not None:
                        PH = PW = int(ref_cam_pose_enc.shape[-2] ** 0.5)
                        KH = KW = int(x_ref_kpts.shape[-2] ** 0.5)
                        ref_cam_pose_enc_interp = rearrange(
                            ref_cam_pose_enc,
                            "B NV (H W) C -> (B NV) C H W",
                            H=PH,
                            W=PW,
                            NV=self.num_ref_views,
                        )
                        ref_cam_pose_enc_interp = F.interpolate(
                            ref_cam_pose_enc_interp,
                            size=(KH, KW),
                            mode="bilinear",
                            antialias=False,
                            align_corners=True,
                        )
                        ref_cam_pose_enc_interp = rearrange(
                            ref_cam_pose_enc_interp,
                            "(B NV) C H W -> B NV (H W) C",
                            NV=self.num_ref_views,
                        )
                        x_ref_kpts = x_ref_kpts + ref_cam_pose_enc_interp
                        cond_cat_split_attn = th.cat(
                            [cond_cat_split_attn, x_ref_kpts], dim=1
                        )

                # add to ref_cond_attn
                elif not self.cond_cat_kpts:
                    cond_attn += x_ref_kpts
                else:
                    cond_attn = th.cat([cond_attn, x_ref_kpts], dim=-1)

        controls = None
        if self.control_mlp is not None:

            all_NV = self.num_views + 1
            if self.face_cropper is not None  and not self.replace_ref_w_face:
                # we have extra face reference vector (add corresponding control vectors)
                ref_kpts = th.cat([ref_kpts, ref_kpts], dim=1)
                ref_plucker = th.cat([ref_plucker, ref_plucker], dim=1)
                all_NV = self.num_views + ref_kpts.shape[1]

            p_drop = self.p_drop_control if training else 0.0
            if len(t.shape) == 1:
                tr = repeat(t, "B -> (B NV)", NV=all_NV)

            all_kpts = None
            if kpts is not None:
                # NOTE: important to note the order (denoise views, then reference view)
                all_kpts = th.cat([kpts, ref_kpts], dim=1)
                all_kpts = Pippo._preprocess(
                    self, data_list=[all_kpts], NV_list=[all_NV], squash_time=True
                )

                # pass control signal through VAE
                if self.control_encoder_latent:
                    with th.no_grad():
                        all_kpts = self.encode_image(all_kpts)
                else:
                    all_kpts = (all_kpts - 127.5) / 127.5

            if plucker is not None:
                plucker = th.cat([plucker, ref_plucker], dim=1)
                plucker = Pippo._preprocess(
                    self, data_list=[plucker], NV_list=[all_NV], squash_time=True
                )
                # assert plucker.shape[0] == all_kpts.shape[0]

            all_x_t = None

            if self.controlmlp_xt:
                x_t_shape = list(x_t.shape)
                x_t_shape[1] = ref_kpts.shape[1] if ref_kpts is not None else 1
                zero_ref_xt = th.zeros(x_t_shape, device=x_t.device)
                all_x_t = th.cat([x_t, zero_ref_xt], dim=1)
                all_x_t = rearrange(all_x_t, "B NV ... -> (B NV) ...")

            controls = self.control_mlp(all_kpts, t=tr, p_drop=p_drop, plucker=plucker, x_t=all_x_t)

        # autoregressive generation
        cond, patch_pos = None, True

        preds = {
            "cond_attn": cond_attn,
            "pos_enc_x": pos_enc_x,
            "cond_cat": cond_cat,
            "controls": controls,
            "cond_cat_split_attn": cond_cat_split_attn,

            "cond": cond,
            "patch_pos": patch_pos,
            "x_t": x_t,
            "t": t,
            "attn_bias": attn_bias,
        }

        return preds

    def get_unconds(
        self,
        image: th.Tensor = None,
        kpts: th.Tensor = None,
        t: Optional[th.Tensor] = None,
        x_t: Optional[th.Tensor] = None,
        ref_image: Optional[th.Tensor] = None,
        ref_kpts: Optional[th.Tensor] = None,
        ref_cam_pose: Optional[th.Tensor] = None,
        cam_pose: Optional[th.Tensor] = None,
        plucker: Optional[th.Tensor] = None,
        ref_plucker: Optional[th.Tensor] = None,
        noisy_views_mask: Optional[th.Tensor] = None, # autoregressive
        noisy_timestep_ids: Optional[th.Tensor] = None, # autoregressive
        ar_uncond_type: Optional[str] = None, # autoregressive
        attn_bias: float = None,
        **kwargs,
    ) -> Dict[str, th.Tensor]:

        # post initialization
        self.post_init()

        # needed when called from sampler (during inference)
        # squash time dimension in cam poses (ref and views), and flatten (encode independently)
        if (not self.multiview_grid_stack) and len(ref_image.shape) > 4:
            data_list = [image, ref_image, cam_pose, ref_cam_pose]
            NV_list = [
                self.num_views,
                self.num_ref_views,
                self.num_views,
                self.num_ref_views,
            ]
            image, ref_image, cam_pose, ref_cam_pose = (
                Pippo._preprocess(
                    self, data_list=data_list, NV_list=NV_list, squash_time=True
                )
            )

        # drop all conditions jointly (and at same places)
        p_drop_all = 0.0
        B = ref_image.shape[0]
        if self.p_drop_all_cond > 0:
            p_drop_all = float(self.p_drop_all_cond > 0)

        # attention based image conditioning
        cond_attn = None
        if self.ref_cond_attn:
            p_drop = float(
                self.p_drop_ref_img > 0
            )  # only removed if model was trained with CFG
            p_drop = max(p_drop, p_drop_all, p_drop_ref_ar_joint)

            # only drop noisy reference view (during cfg)
            if ar_uncond_type == "noisy_ref": p_drop = 0.0

            if self.face_cropper is not None:
                # face and ref joint drop mask
                fr_drop_mask = th.rand(size=(B,), device=ref_image.device) < p_drop

                # encode only face
                ref_image_attn, ref_cam_pose_enc = self.encode_ref_image(
                    self,
                    ref_image,
                    ref_cam_pose,
                    p_drop=p_drop,
                    return_cam_pos_enc=True,
                    drop_mask=fr_drop_mask,
                    encode_face=True
                )
                cond_attn = ref_image_attn
                if ref_cam_pose_enc is not None:
                    ref_cam_pose_enc = ref_cam_pose_enc[:, None]

                # encode full image and attach
                if not self.replace_ref_w_face:
                    full_ref_image_attn, full_ref_cam_pose_enc = self.encode_ref_image(
                        self,
                        ref_image,
                        ref_cam_pose,
                        p_drop=p_drop,
                        return_cam_pos_enc=True,
                        drop_mask=fr_drop_mask,
                    )

                    # merge both along number of reference views
                    cond_attn = th.cat([cond_attn, full_ref_image_attn], dim=1)

                    if ref_cam_pose_enc is not None:
                        full_ref_cam_pose_enc = full_ref_cam_pose_enc[:, None]
                        ref_cam_pose_enc = th.cat([ref_cam_pose_enc, full_ref_cam_pose_enc], dim=1)
            else:
                ref_image_attn, ref_cam_pose_enc = self.encode_ref_image(
                    self,
                    ref_image,
                    ref_cam_pose,
                    p_drop=p_drop,
                    return_cam_pos_enc=True,
                )
                cond_attn = ref_image_attn

                if ref_cam_pose_enc is not None:
                    ref_cam_pose_enc = ref_cam_pose_enc[:, None]

        # camera based positional encoding
        pos_enc_x = None
        if self.cam_mlp is not None:
            p_drop = float(
                self.p_drop_cam_pos_enc > 0
            )  # only removed if model was trained with CFG
            p_drop = max(p_drop, p_drop_all)
            pos_enc_x = self.get_cam_pos_enc(self, cam_pose, p_drop=p_drop)

        # plucker grid based positional encoding
        if self.plucker_mlp is not None:
            plucker_pos_enc_x = self.get_plucker_enc(
                self, plucker=plucker, p_drop=p_drop_all
            )
            pos_enc_x = (
                pos_enc_x + plucker_pos_enc_x
                if pos_enc_x is not None
                else plucker_pos_enc_x
            )

        # keypoints conditioning (TODO: cfg dropout)
        cond_cat, cond_cat_split_attn = None, None
        if self.kpts_encoder is not None:
            # handle generation keypoints
            if kpts is not None:
                B = kpts.shape[0]
                kpts = Pippo._preprocess(
                    self, data_list=[kpts], NV_list=[self.num_views], squash_time=True
                )
                x_kpts = self.encode_kpts(self, kpts, B=B)
                kpts, x_kpts = Pippo._unpreprocess(
                    self, data_list=[kpts, x_kpts], NV_list=[self.num_views] * 2
                )  # [B, NV, C, H, W]

                # split attention
                if self.cond_cat_split_attn:
                    PH = PW = int(pos_enc_x.shape[-2] ** 0.5)
                    KH = KW = int(x_kpts.shape[-2] ** 0.5)
                    pos_enc_x_interp = rearrange(
                        pos_enc_x,
                        "B NV (H W) C -> (B NV) C H W",
                        H=PH,
                        W=PW,
                        NV=self.num_views,
                    )
                    # doensn't matter which interpolation mode we use (because all the posenc values are the same in HxW dimension)
                    pos_enc_x_interp = F.interpolate(
                        pos_enc_x_interp,
                        size=(KH, KW),
                        mode="bilinear",
                        antialias=False,
                        align_corners=True,
                    )
                    pos_enc_x_interp = rearrange(
                        pos_enc_x_interp,
                        "(B NV) C H W -> B NV (H W) C",
                        NV=self.num_views,
                    )
                    cond_cat_split_attn = x_kpts + pos_enc_x_interp

                # add to pos_enc_x or cond_cat
                elif not self.cond_cat_kpts:
                    if pos_enc_x is not None:
                        pos_enc_x += x_kpts
                    else:
                        pos_enc_x = x_kpts
                else:
                    cond_cat = x_kpts

            # handle reference keypoints
            if ref_kpts is not None:
                B = ref_kpts.shape[0]
                ref_kpts = Pippo._preprocess(
                    self,
                    data_list=[ref_kpts],
                    NV_list=[self.num_ref_views],
                    squash_time=True,
                )
                x_ref_kpts = self.encode_kpts(self, ref_kpts, B=B)
                ref_kpts, x_ref_kpts = Pippo._unpreprocess(
                    self,
                    data_list=[ref_kpts, x_ref_kpts],
                    NV_list=[self.num_ref_views] * 2,
                )  # [B, NV, C, H, W]

                # concatenate for split attention (x_kpts and x_ref_kpts)
                if self.cond_cat_split_attn:
                    if cond_cat_split_attn is not None:
                        PH = PW = int(ref_cam_pose_enc.shape[-2] ** 0.5)
                        KH = KW = int(x_ref_kpts.shape[-2] ** 0.5)
                        ref_cam_pose_enc_interp = rearrange(
                            ref_cam_pose_enc,
                            "B NV (H W) C -> (B NV) C H W",
                            H=PH,
                            W=PW,
                            NV=self.num_ref_views,
                        )
                        ref_cam_pose_enc_interp = F.interpolate(
                            ref_cam_pose_enc_interp,
                            size=(KH, KW),
                            mode="bilinear",
                            antialias=False,
                            align_corners=True,
                        )
                        ref_cam_pose_enc_interp = rearrange(
                            ref_cam_pose_enc_interp,
                            "(B NV) C H W -> B NV (H W) C",
                            NV=self.num_ref_views,
                        )
                        x_ref_kpts = x_ref_kpts + ref_cam_pose_enc_interp
                        cond_cat_split_attn = th.cat(
                            [cond_cat_split_attn, x_ref_kpts], dim=1
                        )

                # add to ref_cond_attn
                elif not self.cond_cat_kpts:
                    cond_attn += x_ref_kpts
                else:
                    cond_attn = th.cat([cond_attn, x_ref_kpts], dim=-1)

        controls = None
        if self.control_mlp is not None:
            if self.face_cropper is not None  and not self.replace_ref_w_face:
                # we have extra face reference vector (add corresponding control vectors)
                ref_kpts = th.cat([ref_kpts, ref_kpts], dim=1)
                ref_plucker = th.cat([ref_plucker, ref_plucker], dim=1)

            # NOTE: important to note the order (denoise views, then reference view)
            all_NV = self.num_views + ref_kpts.shape[1]

            if len(t.shape) == 1:
                tr = repeat(t, "B -> (B NV)", NV=all_NV)

            all_kpts = None
            if kpts is not None:
                # NOTE: important to note the order (denoise views, then reference view)
                all_kpts = th.cat([kpts, ref_kpts], dim=1)
                all_kpts = Pippo._preprocess(
                    self, data_list=[all_kpts], NV_list=[all_NV], squash_time=True
                )

                # pass control signal through VAE
                if self.control_encoder_latent:
                    with th.no_grad():
                        all_kpts = self.encode_image(all_kpts)
                else:
                    all_kpts = (all_kpts - 127.5) / 127.5

            if plucker is not None:
                plucker = th.cat([plucker, ref_plucker], dim=1)
                plucker = Pippo._preprocess(
                    self, data_list=[plucker], NV_list=[all_NV], squash_time=True
                )
                # assert plucker.shape[0] == all_kpts.shape[0]

            all_x_t = None
            if self.controlmlp_xt:
                x_t_shape = list(x_t.shape)
                x_t_shape[1] = ref_kpts.shape[1]
                zero_ref_xt = th.zeros(x_t_shape, device=x_t.device)
                all_x_t = th.cat([x_t, zero_ref_xt], dim=1)
                all_x_t = rearrange(all_x_t, "B NV ... -> (B NV) ...")

            # cfg dropout
            p_drop = float(self.p_drop_control > 0)
            p_drop = max(p_drop, p_drop_all)
            controls = self.control_mlp(all_kpts, t=tr, p_drop=p_drop, x_t=all_x_t, plucker=plucker)

        # handle cfg for autoregressive generation
        cond, patch_pos = None, True

        preds = {
            "cond_attn": cond_attn,
            "pos_enc_x": pos_enc_x,
            "cond_cat": cond_cat,
            "controls": controls,
            "cond_cat_split_attn": cond_cat_split_attn,

            "cond": cond,
            "patch_pos": patch_pos,
            "x_t": x_t,
            "t": t,
            "attn_bias": attn_bias,
        }

        return preds

    def forward(
        self,
        image: th.Tensor,
        kpts: Optional[th.Tensor] = None,
        _index: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        ref_cam_pose: Optional[th.Tensor] = None,
        ref_image: Optional[th.Tensor] = None,
        ref_kpts: Optional[th.Tensor] = None,
        cam_pose: Optional[th.Tensor] = None,
        plucker: Optional[th.Tensor] = None,
        ref_plucker: Optional[th.Tensor] = None,
        test_vae: bool = False,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        """
        Args:
            image: [B, NV, T=1, C, H, W]
        """
        # post initialization
        self.post_init()

        B = image.shape[0]
        device = image.device

        # squash time dimension in cam poses (ref and views), and flatten (encode independently)
        data_list = [image, ref_image, cam_pose, ref_cam_pose]
        NV_list = [
            self.num_views,
            self.num_ref_views,
            self.num_views,
            self.num_ref_views,
        ]
        try:
            image, ref_image, cam_pose, ref_cam_pose = (
                Pippo._preprocess(
                    self, data_list=data_list, NV_list=NV_list, squash_time=True
                )
            )  # [B*NV*T, ...]
        except:
            breakpoint()

        with th.no_grad():
            x_0 = self.encode_image(image)

            # sanity checks
            if test_vae:
                self.recon_vae(image, ref_image)
                self.dump_noisy_samples(x_0)

        image, x_0 = Pippo._unpreprocess(
            self, data_list=[image, x_0], NV_list=[self.num_views] * 2
        )  # [B, NV, C, H, W]

        # handle autoregressive noisy reference views
        noisy_views_mask, noisy_timestep_ids = None, None
        x_t, t, noise = self.schedule.add_noise(x_0)

        image = Pippo._preprocess(
            self, data_list=[image], NV_list=[self.num_views], squash_time=False
        )  # [B*NV, ...]

        conds = self.get_conds(
            image,
            kpts=kpts,
            t=t,
            x_t=x_t,
            ref_image=ref_image,
            ref_cam_pose=ref_cam_pose,
            ref_kpts=ref_kpts,
            cam_pose=cam_pose,
            plucker=plucker,
            ref_plucker=ref_plucker,
            noisy_views_mask=noisy_views_mask,
            noisy_timestep_ids=noisy_timestep_ids,
            training=True,
        )  # unflattens conds to [B, NV, ...] internally

        # x_t and t are updated for autoregressive generation
        x_t, t = conds.pop("x_t", x_t), conds.pop("t", t)
        model_preds = self.predict(x_t, t, conds=conds)

        preds = {
            "x_0": x_0,
            "t": t,
            "x_t": x_t,
            "noise": noise,
            "image": image,
        }
        schedule_preds: Dict[str, th.Tensor] = self.schedule(
            model_preds,
            x_t=x_t,
            x_0=x_0,
            t=t,
            noise=noise,
        )
        preds.update(**schedule_preds)

        return preds

    def recon_vae(self, image, ref_image):
        """ utility method to dump vae reconstructed samples """

        import time
        import einops
        import imageio
        import numpy as np
        import torch.nn.functional as F

        # interpolate to match video size
        B, C, H, W = image.shape
        ref_image = F.interpolate(ref_image, size=(H, W))
        ref_image = (
            ref_image.permute(0, 2, 3, 1)
            .cpu()
            .float()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        x_0 = self.encode_image(image)
        decode_image = self.decode_image(x_0)
        ov = (
            einops.einops.rearrange(image, "B C H W -> B H W C")
            .cpu()
            .float()
            .numpy()
            .astype(np.uint8)
        )
        dv = (
            einops.einops.rearrange(decode_image, "B C H W -> B H W C")
            .cpu()
            .float()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        if ref_image.shape[0] != ov.shape[0]:
            ref_image = ref_image.repeat(ov.shape[0] // ref_image.shape[0], 0)

        # reference, original, decoded
        rv = np.concatenate([ref_image, ov, dv], axis=2)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        save_dir = "/".join(os.path.abspath(__file__).split("/")[:-3] + ["outputs", "debug"])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"video_recon_{ts}.mp4")
        imageio.mimsave(save_path, rv, fps=2)
        print(f"saved video to {save_path}")
        breakpoint()

    def dump_noisy_samples(self, x_0):
        """ utility method to dump noisy samples """

        breakpoint()

        import time
        import imageio
        import numpy as np
        import torch.nn.functional as F
        from einops import rearrange

        num_samples = 10
        B, C, H, W = x_0.shape
        x_0 = x_0[:1].repeat(num_samples, 1, 1, 1)  # only dump one sample
        noise = th.randn_like(x_0)

        # noise at uniform scales
        t = th.linspace(0, self.schedule.num_timesteps - 1, num_samples)
        t = t.to(th.int64).to(x_0.device)

        # add noise to image and decode
        x_t = self.schedule.q_sample(x_0, t, noise)

        # stack horizontally and save images
        images = self.decode_image(x_t)
        image_grid = (
            rearrange(images, "B C H W -> H (B W) C")
            .cpu()
            .float()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        rr = self.schedule.rescale_ratio
        save_dir = "/".join(os.path.abspath(__file__).split("/")[:-3] + ["outputs", "debug"])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,f"noised_{ts}_rescale_ratio_{rr}_imgsize_{self.img_size}.jpg")
        imageio.imsave(save_path, image_grid)
        print(f"saved image to {save_path}")
        breakpoint()

    @staticmethod
    def _preprocess(cls, data_list, NV_list=None, squash_time=False):
        """
        Converts video frames as images for the model by squashing time axis.
        Also, if not using multiview grid stack, flatten batch and num_views dimensions.
        Shared by video and image models both.
        """
        if cls.multiview_grid_stack:
            return_list = [
                data.squeeze(1) if data is not None else None for data in data_list
            ]
        else:
            return_list = []
            for data, NV in zip(data_list, NV_list):
                if data is None:
                    pass
                elif squash_time:
                    data = rearrange(data, "B NV T ... -> (B NV T) ...", NV=NV)
                else:
                    data = rearrange(data, "B NV ... -> (B NV) ...", NV=NV)
                return_list.append(data)

        return return_list if len(return_list) > 1 else return_list[0]

    @staticmethod
    def _unpreprocess(cls, data_list, NV_list, expand_time=False):
        """
        If not using multiview grid stack, separate batch and num_views dimensions.
        Shared by video and image models both.
        """
        if cls.multiview_grid_stack:
            return_list = data_list
        else:
            return_list = []
            for data, NV in zip(data_list, NV_list):
                if expand_time:
                    NV, T = NV
                    data = rearrange(data, "(B NV T) ... -> B NV T ...", NV=NV, T=T)
                else:
                    data = rearrange(data, "(B NV) ... -> B NV ...", NV=NV)
                return_list.append(data)

        return return_list if len(return_list) > 1 else return_list[0]


def test_pippo():
    import einops, os
    import torch.multiprocessing as mp
    from omegaconf import OmegaConf
    from torchvision.utils import make_grid
    from latent_diffusion.data import load_batches

    th.set_float32_matmul_precision("high")
    mp.set_start_method("spawn", force=True)
    device = th.device("cuda:0")
    config_path = f"/home/{os.environ['USER']}/rsc/latent_diffusion/config/pippo/head_only/1K_mugsy_kpts_plucker_face_only_2v_fp16.yml"
    config = OmegaConf.load(config_path)

    # faster loadup (no checkpoint)
    del config.model.ckpt
    config.dataset.multiview.shared.verbose = True
    config.dataset.multiview.shared.debug = True

    model = load_from_config(config.model).to(device)
    print(f"num of params: {sum(p.numel() for p in model.parameters()) / 1e9} B")

    batches = load_batches(batch_size=2)
    loader = iter(batches)

    for _ in range(2):
        batch = next(loader)
        batch = to_device(batch, device)
        with th.no_grad():
            preds = model(test_vae=False, **batch)

        for k, v in preds.items():
            print(f"{k} {v.shape}")


if __name__ == "__main__":
    test_pippo()
