# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import imageio
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import glob
import torchvision.transforms.v2 as T
import logging
import pprint
from easydict import EasyDict as edict
from collections import defaultdict
from copy import deepcopy
from functools import partial
from more_itertools import chunked
import json


th.set_float32_matmul_precision("highest")


# configure logger before importing anything else
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logging intialized.")

registry = edict()


def load_lightglue(device="cuda:0"):
    """Load LightGlue model if not loaded."""
    # load lightglue model if not loaded
    if "lightglue" in registry:
        return

    import kornia.feature as KF
    
    # Initialize DISK feature extractor and LightGlue matcher
    disk = KF.DISK.from_pretrained("depth").eval().to(device)
    lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)
    
    # attach to registry
    registry.disk = disk
    registry.lightglue = lg_matcher
    
    # preprocess image for lightglue
    def lightglue_preprocess(image):
        assert image.ndim == 3 and image.dtype == np.uint8
        # Convert to float32 and normalize to [0, 1]
        image = th.as_tensor(image/255.0, dtype=th.float32)
        # Add batch dimension and move to device
        image = image.permute(2, 0, 1)[None].to(device)
        return image
    
    # attach to registry
    registry.lightglue_preprocess = lightglue_preprocess


def match_features(img1, img2, conf_thresh=0.2, matches_thresh=5, num_features=2048):
    """Match features between two images using LightGlue.
    
    Args:
        img1: First image (H, W, C)
        img2: Second image (H, W, C)
        conf_thresh: Confidence threshold for matches
        matches_thresh: Minimum number of matches required
        num_features: Number of features to detect
        
    Returns:
        dict: Contains matches, keypoints and None if matching fails
    """
    import kornia.feature as KF

    # Load models if not loaded
    load_lightglue()
    
    # Preprocess images
    pimg1 = registry.lightglue_preprocess(img1)
    pimg2 = registry.lightglue_preprocess(img2)
    
    # Get image dimensions
    hw1 = th.tensor(pimg1.shape[2:], device=pimg1.device)
    hw2 = th.tensor(pimg2.shape[2:], device=pimg2.device)
    
    with th.inference_mode():
        # Extract features using DISK
        inp = th.cat([pimg1, pimg2], dim=0)
        features1, features2 = registry.disk(inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        
        # Create LAFs (Local Affine Frames)
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], 
            th.ones(1, len(kps1), 1, 1, device=pimg1.device)
        )
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], 
            th.ones(1, len(kps2), 1, 1, device=pimg2.device)
        )
        
        # Match features using LightGlue
        dists, idxs = registry.lightglue(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
    
    # Convert to numpy
    idxs = idxs.cpu().numpy()
    kps1 = kps1.cpu().numpy()
    kps2 = kps2.cpu().numpy()
    
    if len(idxs) < matches_thresh:
        print(f"skipping pair: {len(idxs)=} < {matches_thresh=}")
        return None
    
    # Convert to OpenCV format
    matches = [cv2.DMatch(i[0], i[1], 0) for i in idxs]
    keypoints1 = [cv2.KeyPoint(x, y, 1) for (x, y) in kps1]
    keypoints2 = [cv2.KeyPoint(x, y, 1) for (x, y) in kps2]
    
    return {
        'matches': matches,
        'keypoints1': keypoints1,
        'keypoints2': keypoints2,
        'kpts0': kps1,
        'kpts1': kps2
    }


def get_l2_reproj(video_grid, metrics_dict, batch, output_dir, sample_name, H, W, NV):
    output_dir = os.path.join(output_dir, "reproj")
    os.makedirs(output_dir, exist_ok=True)

    import itertools
    from scipy.optimize import linear_sum_assignment

    # only generated views (no reference views)
    row_view_grid = rearrange(video_grid, "B T (NR H) (NC W) C -> (B T NR) NC H W C", H=H, W=W)
    
    if isinstance(row_view_grid, th.Tensor):
        row_view_grid = row_view_grid.numpy().astype(np.uint8)

    # only generated views (no reference views)
    row_view_grid = row_view_grid[:, -NV:]
    cam_intrins = batch["cam_intrin"][-NV:]
    cam_extrins = batch["cam_extrin"][-NV:]

    # adjust camera intrinsics for downsample
    # original HxW: 4096, 2668
    # rescale shorter side (width) to given size
    rescale_factor = W / 2668
    cam_intrins[:, :2] *= rescale_factor

    # adjust camera intrinsics for longer side (height) for cropping
    cam_intrins[:, 1, 2] = cam_intrins[:, 1, 2] - (4096 * rescale_factor - H) * 0.5

    # added just for completeness (difference is zero)
    cam_intrins[:, 0, 2] = cam_intrins[:, 0, 2] - (2668 * rescale_factor - W) * 0.5

    # compute krts
    cam_krts = [(ci @ ce).numpy() for ci, ce in zip(cam_intrins, cam_extrins)]

    # select camera pairs that are close in 3D for reprojection error
    cam_ts = cam_extrins[:, :3, 3]
    cam_dist_matrix = th.linalg.norm(cam_ts.unsqueeze(0) - cam_ts.unsqueeze(1), dim=2)

    # use bipartite matching to find the best matching
    cam_dist_matrix = cam_dist_matrix.cpu().numpy()
    np.fill_diagonal(cam_dist_matrix, np.inf)
    row_ind, col_ind = linear_sum_assignment(cam_dist_matrix)

    # retain only unique pairs (a,b) = (b,a)
    pairs = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
    pairs = list(set(tuple(sorted((row, col))) for row, col in pairs))

    metrics = []
    for row_idx, row in tqdm(enumerate(row_view_grid), total=len(row_view_grid), desc="computing reprojection error"):
        num_views = row.shape[0]
        row_metrics = []

        # compute pairwise reprojection error
        for pair in pairs:
            img1, img2 = row[pair[0]], row[pair[1]]
            match_result = match_features(img1, img2)
            
            if match_result is None:
                row_metrics.append(None)
                continue

            matches = match_result['matches']
            keypoints1 = match_result['keypoints1']
            keypoints2 = match_result['keypoints2']

            # draw matches for visualization
            matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
            save_path = os.path.join(output_dir, f"{sample_name}_row_{row_idx}_matches_{pair[0]}_{pair[1]}.png")
            imageio.imsave(save_path, matches_img)

            # visualize ground truth keypoints (to ensure camera intrinsics are correct)
            if "keypoints_3d" in batch and row_idx == 0:
                # shape: num_views x num_points x 4 (valid_index, x, y, z)
                gt_keypoints3d = batch["keypoints_3d"].squeeze().cpu().numpy()

                gt_keypoints_inds = gt_keypoints3d[:, :, 0].astype(int)  # original indices out of 274
                gt_keypoints3d = gt_keypoints3d[:, :, 1:4]  # 3d keypoints
                gt_keypoints3d = np.append(gt_keypoints3d, np.ones_like(gt_keypoints3d)[:,:,:1], axis=-1)  # homogenous coords

                # project to 2d
                gt_keypoints = [np.dot(cam_krt, np.transpose(gt_kpts3d)) for gt_kpts3d, cam_krt in zip(gt_keypoints3d, cam_krts)]
                gt_keypoints = [gt_kpts2d[:2] / gt_kpts2d[-1] for gt_kpts2d in gt_keypoints]

                # fit back into all detected keypoints and attach confidence of 0/1
                gt_keypoints = rearrange(np.stack(gt_keypoints), "NV C P -> NV P C")
                gt_keypoints_with_conf = np.zeros((NV, 274, 3))
                gt_keypoints_with_conf[:, gt_keypoints_inds[0], :2] = gt_keypoints
                gt_keypoints_with_conf[:, gt_keypoints_inds[0], 2] = 1.0
                gt_keypoints = gt_keypoints_with_conf

                gt_keypoints1 = [cv2.KeyPoint(x, y, 1) for (x, y, c) in gt_keypoints[pair[0]]]
                gt_keypoints2 = [cv2.KeyPoint(x, y, 1) for (x, y, c) in gt_keypoints[pair[1]]]

                gt_matches = [cv2.DMatch(i, i, 0) for i in range(len(gt_keypoints[0]))]
                gt_fil_inds = [(gt_keypoints[pair[0]][i][2] > 0 and gt_keypoints[pair[1]][i][2] > 0) for i in range(len(gt_keypoints[0]))]
                gt_matches = np.array(gt_matches)[gt_fil_inds].tolist()

                gt_matches_img = cv2.drawMatches(img1, gt_keypoints1, img2, gt_keypoints2, gt_matches, None, flags=2)
                gt_save_path = os.path.join(output_dir, f"gt_row_{row_idx}_matches_{pair[0]}_{pair[1]}.png")
                imageio.imsave(gt_save_path, gt_matches_img)

            fil_inds_kpts1 = [match.queryIdx for match in matches]
            fil_inds_kpts2 = [match.trainIdx for match in matches]

            tkpts1 = np.array([kp.pt for kp in keypoints1])[fil_inds_kpts1].T
            tkpts2 = np.array([kp.pt for kp in keypoints2])[fil_inds_kpts2].T

            krt1, krt2 = cam_krts[pair[0]], cam_krts[pair[1]]

            triangulated_points = cv2.triangulatePoints(krt1, krt2, tkpts1, tkpts2)
            points_3d = triangulated_points[:3] / triangulated_points[3]

            # compute reprojection error
            kpts1_reprojected = (krt1 @ triangulated_points)
            kpts1_reprojected /= kpts1_reprojected[2]
            kpts_1_reproj_error = np.linalg.norm(kpts1_reprojected[:2] - tkpts1, axis=0).mean()

            kpts2_reprojected = (krt2 @ triangulated_points)
            kpts2_reprojected /= kpts2_reprojected[2]
            kpts_2_reproj_error = np.linalg.norm(kpts2_reprojected[:2] - tkpts2, axis=0).mean()

            # append reprojection error to list
            reproj_error = (kpts_1_reproj_error + kpts_2_reproj_error) / 2
            row_metrics.append(reproj_error)

        # append row metrics to list
        metrics.append(row_metrics)

    # compute mean reprojection error for each row
    # we exclude nans in each row and average
    unbalanced_metrics = np.array(metrics)
    row_lens = (unbalanced_metrics != None).sum(-1)
    unbalanced_metrics[unbalanced_metrics == None] = 0.0
    unbalanced_metrics = unbalanced_metrics.sum(-1) / (row_lens + 1e-6)

    # we only average columns where all rows have valid value
    balanced_metrics = np.array(metrics)
    valid_columns = np.all(balanced_metrics != None, axis=0)

    if sum(valid_columns) == 0:
        logger.warning("cannot compute balanced reprojection error, no valid columns")
        balanced_metrics = np.ones_like(unbalanced_metrics) * -1
    else:
        balanced_metrics = balanced_metrics[:, valid_columns].mean(-1)

    # attach to metrics dict
    metrics_dict["gt_unbalanced_l2_reproj"] = float(unbalanced_metrics[0])
    metrics_dict["gt_balanced_l2_reproj"] = float(balanced_metrics[0])
    metrics_dict["gen_unbalanced_l2_reproj"] = float(unbalanced_metrics[1:].mean())
    metrics_dict["gen_balanced_l2_reproj"] = float(balanced_metrics[1:].mean())

    return metrics_dict


if __name__ == "__main__":
    local_dir = os.getcwd().split("scripts/")[0]

    # load generated and ground truth images 
    sample_paths = [
        os.path.join(local_dir, "sample_data/gen_samples/gen_sample_0.npy"),
        os.path.join(local_dir, "sample_data/gen_samples/gen_sample_1.npy"),
    ]

    for sample_path in sample_paths:
        sample = np.load(sample_path, allow_pickle=True).item()
        cameras = sample["cameras"]

        # add image grid
        ref, gen, gt = sample["ref_image"], sample["gen_samples"], sample["ground_truth"]
        gen = gen.squeeze(0)[:gt.shape[0]]
        gt_row = np.concatenate([ref, gt], axis=1)
        gen_row = np.concatenate([ref, gen], axis=1)
        image_grid = np.concatenate([gt_row, gen_row], axis=0)
        # batch and time dims
        image_grid = image_grid[None, None] 

        # add cameras 
        batch = dict(cam_intrin=[], cam_extrin=[])
        for cam in sample['id']['cams'][0]:
            found = False
            for _cam in cameras["KRT"]:
                if str(cam) == _cam['cameraId']:
                    
                    # make intrinsics from 3x3 to 3x4
                    K = np.concatenate([np.array(_cam['K']).T, np.zeros((3, 1))], axis=-1)
                    batch['cam_intrin'].append(K)

                    batch['cam_extrin'].append(np.array(_cam['T']).T)
                    found = True
                    break
            if not found:
                print(f"camera {cam} not found in {camera_path}")
                breakpoint()
        batch["cam_intrin"] = np.array(batch["cam_intrin"])
        batch["cam_extrin"] = np.array(batch["cam_extrin"])
        
        H = W = 512
        NV = image_grid.shape[-2] // H - 1
        
        # make tensors 
        image_grid = th.from_numpy(image_grid)
        batch = {k: th.from_numpy(v) for k, v in batch.items()}

        
        # compute reprojection error
        metrics = get_l2_reproj(image_grid, {}, batch, ".", "gen_sample_0", H, W, NV)
        print(f"{metrics=}")