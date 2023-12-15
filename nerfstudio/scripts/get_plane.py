# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
get_plane.py
"""
from __future__ import annotations

# import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE
import matplotlib.pyplot as plt
# import dataclasses
# import functools
import os
# import time
# from dataclasses import dataclass, field
from pathlib import Path
# from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
# from nerfstudio.configs.experiment_config import ExperimentConfig
# from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
# from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
# from nerfstudio.engine.optimizers import Optimizers
# from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
# from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
# from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
# from nerfstudio.utils.writer import EventName, TimeWriter
# from nerfstudio.viewer.server.viewer_state import ViewerState
# from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState
from nerfstudio.model_components.ray_generators import RayGenerator_surface_detection
# from rich import box, style
# from rich.panel import Panel
# from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler
from sklearn.linear_model import LinearRegression,TheilSenRegressor, HuberRegressor
import numpy as np

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from itertools import compress

from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.datasets.depth_dataset import DepthDataset


DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore

# new
import scipy

def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)

def transform_yx(yx, transform, center):
    yx = yx - center
    yx = (transform @ yx.T).T
    yx = yx + center
    return yx

def plane_estimation(config: TrainerConfig):
    config.setup(local_rank=0, world_size=1)
    pipeline = config.pipeline.setup(device = "cuda", load_dir=config.load_dir)
    # optimizers = Optimizers(config.optimizers.copy(), pipeline.get_param_groups())
    grad_scaler = GradScaler(enabled=True)

    # load in the checkpoint
    load_dir = config.load_dir
    load_checkpoint = config.load_checkpoint
    if load_dir is not None:
        load_step = config.load_step
        if load_step is None:
            print("Loading latest Nerfstudio checkpoint from load_dir...")
            # NOTE: this is specific to the checkpoint name format
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
        load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        _start_step = loaded_state["step"] + 1
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        # optimizers.load_optimizers(loaded_state["optimizers"])
        # if "schedulers" in loaded_state and config.load_scheduler:
        #     optimizers.load_schedulers(loaded_state["schedulers"])
        grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
    elif load_checkpoint is not None:
        assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
        loaded_state = torch.load(load_checkpoint, map_location="cpu")
        _start_step = loaded_state["step"] + 1
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        #optimizers.load_optimizers(loaded_state["optimizers"])
        #if "schedulers" in loaded_state and config.load_scheduler:
            #optimizers.load_schedulers(loaded_state["schedulers"])
        grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
    else:
        CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    # generate ray for surface detection from evaluation dataset
    # TODO: avoid "pipeline." prefix
   
    #pipeline.datamanager.surface_detection_dataset = pipeline.datamanager.eval_dataset
    # pipeline.datamanager.surface_detection_dataset = pipeline.datamanager.train_dataset
    # force the dataset type to be DepthDataset
    pipeline.datamanager.dataset_type = DepthDataset
    pipeline.datamanager.surface_detection_dataset = pipeline.datamanager.create_train_dataset()

    pipeline.datamanager.mask = [item for item in pipeline.datamanager.surface_detection_dataset] 
    # pipeline.datamanager.surface_detection_pixel_sampler = pipeline.datamanager._get_pixel_sampler(pipeline.datamanager.surface_detection_dataset, pipeline.datamanager.config.eval_num_rays_per_batch)
    
    # white is 1, black is 0
    y_outbound = pipeline.datamanager.mask[0]['mask'].shape[0]
    x_outbound = pipeline.datamanager.mask[0]['mask'].shape[1]
    # y_outbound: 738
    # x_outbound: 994

    pipeline.datamanager.surface_detection_ray_generator = RayGenerator_surface_detection(pipeline.datamanager.train_dataset.cameras.to(pipeline.datamanager.device))
    pipeline.datamanager.surface_detection_camera = pipeline.datamanager.surface_detection_dataset.cameras

    world_xyz = []
    colors = []
    # HARDCODED for polycam, which means the right direction is actually downwards in the real world
    for i, item in enumerate(pipeline.datamanager.mask):
        mask_array = item['mask'].numpy().squeeze()
        y, x = np.where(mask_array == 0)
        max_y = np.max(y)
        min_y = np.min(y)
        max_x = np.max(x)
        min_x = np.min(x)

        image_idx = item['image_idx']
        camera = pipeline.datamanager.surface_detection_camera[image_idx]
        # decide which direction (out of "down", "up", "left", "right") to search for the bottom pixel
        # read depth image
        depth_filepath = pipeline.datamanager.surface_detection_dataset.depth_filenames[item['image_idx']]
        depth_height = int(camera.height)
        depth_width = int(camera.width)
        # depth_height: 738
        # depth_width: 994
        depth_image = get_depth_image_from_path(filepath=depth_filepath, height=depth_height, width=depth_width, scale_factor=1.0)
        # depth_image.shape: torch.Size([738, 994, 1])
        # split depth_image into 9 parts using max_y, min_y, max_x, min_x, and extract the "up", "down", "left", "right" parts
        depth_image_up = depth_image[:min_y, min_x:max_x, :]
        depth_image_down = depth_image[max_y:, min_x:max_x, :]
        depth_image_left = depth_image[min_y:max_y, :min_x, :]
        depth_image_right = depth_image[min_y:max_y, max_x:, :]
        # find the mean depth of each part
        depth_mean_up = depth_image_up.mean()
        depth_mean_down = depth_image_down.mean()
        depth_mean_left = depth_image_left.mean()
        depth_mean_right = depth_image_right.mean()
        # find the part with the smallest mean depth (in case there is NaN, replace it with torch.inf)
        depth_means = torch.tensor([depth_mean_up, depth_mean_down, depth_mean_left, depth_mean_right])
        depth_means[torch.isnan(depth_means)] = torch.inf
        downward_direction_index = int(torch.argmin(depth_means).cpu().numpy())
        direction_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
        downward_direction = direction_dict[downward_direction_index]
        print(f"downward_direction: {downward_direction}")

        # center = torch.tensor([depth_height / 2, depth_width / 2])

        # # transform yx coordinates of any other directions to the "right" direction
        # if downward_direction == "right":
        #     transform = torch.tensor([[1, 0], 
        #                               [0, 1]])
        # elif downward_direction == "left":
        #     transform = torch.tensor([[1, 0], 
        #                               [0, -1]])
        #     assert transform_yx(torch.tensor([0, 0]), transform, center).allclose(torch.tensor([0, depth_width-1]))
        #     print("----------------------------------")
        # elif downward_direction == "up":
        #     transform = torch.tensor([[0, 1], 
        #                               [-1, 0]])
        #     assert transform_yx(torch.tensor([0, depth_width/2]), transform, center).allclose(torch.tensor([depth_height-1, 0]))
        # elif downward_direction == "down":
        #     transform = torch.tensor([[0, -1], 
        #                               [1, 0]])
        # else:
        #     print(f"downward_direction: {downward_direction} is not valid")
        #     raise NotImplementedError
        



        # width = max_x - min_x
        if downward_direction in ["right", "left"]:
            height = max_x - min_x
        else:
            height = max_y - min_y
        offset_proportion = 0.1
        offset = int(height * offset_proportion)

        if downward_direction == "right":
            offset_shaped = np.array([[0, offset], 
                                      [0, offset]])
            bottom_corners = np.array([[min_y, max_x], 
                                      [max_y, max_x]])
            max_num_offset = (x_outbound - max_x) // offset
        elif downward_direction == "left":
            offset_shaped = np.array([[0, -offset], 
                                      [0, -offset]])
            bottom_corners = np.array([[min_y, min_x], 
                                      [max_y, min_x]])
            max_num_offset = min_x // offset
        elif downward_direction == "up":
            offset_shaped = np.array([[-offset, 0], 
                                      [-offset, 0]])
            bottom_corners = np.array([[min_y, min_x], 
                                      [min_y, max_x]])
            max_num_offset = min_y // offset
        elif downward_direction == "down":
            offset_shaped = np.array([[offset, 0], 
                                      [offset, 0]])
            bottom_corners = np.array([[max_y, min_x], 
                                      [max_y, max_x]])
            max_num_offset = (y_outbound - max_y) // offset
        else:
            print(f"downward_direction: {downward_direction} is not valid")
            raise NotImplementedError
        # bottom_corners_init = bottom_corners + offset_shaped

        # max_num_offset = (x_outbound - max_x) // offset
        if max_num_offset <= 1:
            continue

        # create a new np.array 'corner_candidates' with shape (max_num_offset, 2, 2) where corner_candidates[i] is bottom_corners + i * offset_shaped
        corner_candidates = bottom_corners + np.arange(1, max_num_offset).reshape(-1, 1, 1) * offset_shaped
        # corner_candidates.shape: (31, 2, 2)

        num_candidates = corner_candidates.shape[0] * 2
        idx_array = np.full((num_candidates, 1), image_idx)

        candidate_points = np.hstack((idx_array, corner_candidates.reshape(num_candidates, 2)))
        ray_indices = torch.from_numpy(candidate_points).int() # ray_indices.shape: torch.Size([62, 3])

        # create ray bundle from ray indices
        ray_bundle_corner_candidates = pipeline.datamanager.surface_detection_ray_generator(ray_indices)
        depth_candidates, colors_candidates = pipeline.get_surface_detection(pipeline, ray_bundle_corner_candidates)
        # depth_candidates.shape: torch.Size([62, 1])
        # colors_candidates.shape: torch.Size([62, 3])

        depth_candidates_reshaped = depth_candidates.reshape(corner_candidates.shape[0], 2, 1) # (31, 2, 1)
        colors_candidates_reshaped = colors_candidates.reshape(corner_candidates.shape[0], 2, 3) # (31, 2, 3)
        
        depth_diff = depth_candidates_reshaped[1:] - depth_candidates_reshaped[:-1] # (30, 2, 1)
        # print(f"depth_diff < 0: {depth_diff < 0}") # looks reliable
        # print(f"depth_diff[1:] > depth_diff[:-1]: {depth_diff[1:] > depth_diff[:-1]}") # turned out to contain many Falses even at the start

        # depth_criteria = (depth_diff < 0) * torch.cat([torch.tensor([[[True], 
        #                                                               [True]]], device=depth_diff.device), depth_diff[1:] > depth_diff[:-1]], dim=0)
        depth_criteria = depth_diff < 0
        # depth_criteria.shape: torch.Size([30, 2, 1])

        depth_criteria_every_offset = depth_criteria.all(dim=1).squeeze(-1)
        # a new boolean tensor 'depth_safe' with same shape as depth_criteria_every_offset, and depth_safe[i] is True if depth_criteria_every_offset[i] is True when depth_criteria_every_offset[:i+1] are all True
        depth_criteria_every_offset_finally_false = torch.cat([depth_criteria_every_offset, torch.tensor([False], device=depth_criteria_every_offset.device)], dim=0)
        # false_indices = torch.where(depth_criteria_every_offset == False)[0]
        false_indices = torch.where(depth_criteria_every_offset_finally_false == False)[0] # make sure there is always a terminating False
        if len(false_indices) == 0 or false_indices[0] == 0:
            continue
        first_false = false_indices[0]
        depth_safe = torch.zeros_like(depth_criteria_every_offset)
        depth_safe[:first_false] = True # depth_safe.shape: torch.Size([30])
        # concat a True at the start of depth_safe
        depth_safe = torch.cat([torch.tensor([True], device=depth_safe.device), depth_safe], dim=0) # depth_safe.shape: torch.Size([31])

        depth_reshaped = depth_candidates_reshaped[depth_safe] # (n_safe_pairs, 2, 1)
        depth_corners = depth_reshaped.reshape(-1) # (n_safe,)
        colors_reshaped = colors_candidates_reshaped[depth_safe] # (n_safe_pairs, 2, 3)
        colors_corners = colors_reshaped.reshape(-1, 3) # (n_safe, 3)
        corner_candidates_torch = torch.from_numpy(corner_candidates).to(depth_safe.device) # (31, 2, 2)
        corners_reshaped = corner_candidates_torch[depth_safe] # (n_safe_pairs, 2, 2)
        corners = corners_reshaped.reshape(-1, 2) # (n_safe, 2)

        # randomly sample points in the rectangle area surrounded by corners
        n_every_offset = 5
        # first and final pair of corners_reshaped make up the rectangle area
        starting_pair = corners_reshaped[0]
        ending_pair = corners_reshaped[-1]
        corners_safe = torch.cat([starting_pair, ending_pair], dim=0) # (4, 2)
        x_safe_min = corners_safe[:, 1].min()
        x_safe_max = corners_safe[:, 1].max()
        y_safe_min = corners_safe[:, 0].min()
        y_safe_max = corners_safe[:, 0].max()

        # left_pair = corners_reshaped[0]
        # right_pair = corners_reshaped[-1]
        # x_safe_min = left_pair[0][1]
        # x_safe_max = right_pair[1][1]
        # y_safe_min = left_pair[0][0]
        # y_safe_max = right_pair[1][0]
        # bottom_corners = np.array([[min_y, max_x], 
        #                            [max_y, max_x]])

        # print(f"randomly sampling {n_every_offset} points in the rectangle area surrounded by corners")
        # print(f"x_safe_min: {x_safe_min}")
        # print(f"x_safe_max: {x_safe_max}")
        # print(f"y_safe_min: {y_safe_min}")
        # print(f"y_safe_max: {y_safe_max}")
        
        n_safe_pairs = corners_reshaped.shape[0]
        n_random = n_safe_pairs * n_every_offset
        x_random = torch.randint(low=x_safe_min, high=x_safe_max, size=(n_random, 1), device=depth_safe.device)
        y_random = torch.randint(low=y_safe_min, high=y_safe_max, size=(n_random, 1), device=depth_safe.device)
        random_points_yx = torch.cat([y_random, x_random], dim=1) # (n_random, 2)

        idx_tensor = torch.full((n_random, 1), image_idx).to(random_points_yx.device)
        random_points = torch.cat([idx_tensor, random_points_yx], dim=1) # (n_random, 3)
        ray_bundle_random = pipeline.datamanager.surface_detection_ray_generator(random_points)
        depth_random, colors_random = pipeline.get_surface_detection(pipeline, ray_bundle_random)

        all_yx = torch.cat([corners, random_points_yx], dim=0) # (n_safe + n_random, 2)
        all_depth = torch.cat([depth_corners, depth_random.squeeze()], dim=0) # (n_safe + n_random,)
        all_colors = torch.cat([colors_corners, colors_random], dim=0) # (n_safe + n_random, 3)

        fx = camera.fx.to(depth_corners.device)
        fy = camera.fy.to(depth_corners.device)
        cx = camera.cx.to(depth_corners.device)
        cy = camera.cy.to(depth_corners.device)
        c2w = camera.camera_to_worlds.to(depth_corners.device)
        # fx: tensor([737.9322])
        # fy: tensor([737.9322])
        # cx: tensor([499.0845])
        # cy: tensor([371.1886])
        # c2w.shape: torch.Size([3, 4])

        # x = corners[:, 1]
        # y = corners[:, 0]
        x = all_yx[:, 1]
        y = all_yx[:, 0]

        # xyz in camera coordinates
        # X = (x - cx) * depth_corners / fx
        # Y = -(y - cy) * depth_corners / fy
        # Z = -depth_corners
        X = (x - cx) * all_depth / fx
        Y = -(y - cy) * all_depth / fy
        Z = -all_depth
        
        # Convert to world coordinates
        camera_xyz = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1)
        world_coordinates = (c2w @ camera_xyz.T).T[..., :3]

        world_xyz.append(world_coordinates.cpu().numpy())
        # colors.append(colors_corners.cpu().numpy())
        colors.append(all_colors.cpu().numpy())

    world_xyz_np = np.concatenate(world_xyz, axis=0)
    colors_np = np.concatenate(colors, axis=0)

    # find the median of each column
    median = np.median(colors_np, axis=0) # (3,)
    # find the covariance matrix of the colors_np values
    cov = np.cov(colors_np.T) # (3, 3)
    # inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov) # (3, 3)
    # find the mahalanobis distance of each point from the median
    mahalanobis = scipy.spatial.distance.cdist(colors_np, [median], metric='mahalanobis', VI=cov_inv) # (N, 1)
    mahalanobis_similarity = 1 / (1 + mahalanobis) # (N, 1)

    #     above_bottom_pixel = (y_lower, bottom_pixel[1])
    
    #     # Find the indices where y is between y_up_bottom and y_bottom
    #     indices = np.where((y >= above_bottom_pixel[0]) & (y <= bottom_pixel[0]))

    #     # Extract the corresponding x values
    #     x_values = x[indices]

    #     # Compute the width as the difference between the maximum and minimum x values
    #     width = np.max(x_values) - np.min(x_values)
    #     #TODO: find the width of the area between y_bottom and y_up_bottom  
    #     pipeline.datamanager.mask[i]['bottom_pixel'] = bottom_pixel
    #     #print(bottom_pixel)
    #     #pipeline.datamanager.mask[i]['y_up_bottom'] = y_up_bottom
    #     pipeline.datamanager.mask[i]['width'] = width
    #     #pipeline.datamanager.mask[i]['width'] = 2
        
    
    
    # y_sample_range = 60
    # num_samples = 50

    # # filter out out-of-bound indices
    # pipeline.datamanager.surface_detection_camera = pipeline.datamanager.surface_detection_dataset.cameras

    # # Initialize an empty numpy array
    # all_points_np = np.empty((0, 3))        
    
    # camera_list = []
    # image_list = []

    # for i, item in enumerate(pipeline.datamanager.mask):

    #     idx = item['image_idx']
    #     bottom_y = int(item['bottom_pixel'][0])
    #     bottom_x = int(item['bottom_pixel'][1])
    #     width = int(item['width'])
    #     camera = pipeline.datamanager.surface_detection_camera[idx]
    #     image = pipeline.datamanager.surface_detection_dataset[idx]
    #     # Append the camera to the camera_list
    #     camera_list.append([camera] * num_samples)
    #     image_list.append([image] * num_samples)
    #     # Generate an idx array of the same length as points
    #     idx_array = np.full((num_samples, 1), idx)
    #     # Generate random x coordinates within the width
    #     x_samples = np.random.randint(low= bottom_x - width/2, high= bottom_x + width/2, size=num_samples)

    #     # Generate random y coordinates between y_up_bottom and y_bottom
    #     y_samples = np.random.randint(low=bottom_y + y_sample_range -10, high=bottom_y + y_sample_range, size=num_samples)
        

    #     # Combine the idx, x and y coordinates into a 2D array
    #     points = np.column_stack((idx_array, x_samples, y_samples)) # (num_samples, 3)
    #     #print(points.shape)
    #     # Concatenate the points to all_points_np
    #     all_points_np = np.concatenate((all_points_np, points), axis=0)
    #     #print(all_points_np.shape)
    # # Flatten the camera_list
    # camera_list = [camera for sublist in camera_list for camera in sublist]
    # image_list = [image for sublist in image_list for image in sublist]
    
    # #print(all_points_np)
    # #raise NotImplementedError
    

    # pipeline.datamanager.ray_indices = torch.from_numpy(all_points_np).int()
    # mask = pipeline.datamanager.ray_indices[:, 1] + y_sample_range < y_outbound

    # #print(pipeline.datamanager.ray_indices)
    # #mask = pipeline.datamanager.ray_indices[:, :1] + y_sample_range < y_outbound

    # camera_list = list(compress(camera_list, mask))

    # camera_list = [camera.to(pipeline.datamanager.device) for camera in camera_list]
    # image_list = list(compress(image_list, mask))

    # pipeline.datamanager.surface_detection_ray_generator = RayGenerator_surface_detection(pipeline.datamanager.train_dataset.cameras.to(pipeline.datamanager.device))

    # # create ray bundle from ray indices
    # pipeline.datamanager.ray_bundle_surface_detection = pipeline.datamanager.surface_detection_ray_generator(pipeline.datamanager.ray_indices, mask = mask)

    # # expand pipeline.datamanager.filtered_cameras to align with pipeline.datamanager.ray_indices
    # #pipeline.datamanager.expanded_cameras = [camera for camera in pipeline.datamanager.filtered_cameras for _ in range(num_samples)]
    # pipeline.datamanager.expanded_cameras = camera_list
    # pipeline.datamanager.filtered_data = image_list
    # #print(pipeline.datamanager.ray_indices.shape, len(pipeline.datamanager.expanded_cameras))
    # #print(pipeline.datamanager.ray_indices.shape, len(camera_list))

    # output, colors = pipeline.get_surface_detection(pipeline, pipeline.datamanager.ray_bundle_surface_detection)
    # num_image = len(pipeline.datamanager.expanded_cameras)
    # assert num_image == len(output) , f"false length"

    # print(f"output.shape: {output.shape}")
    # print(f"colors.shape: {colors.shape}")
    # print(f"num_image: {num_image}")
    # colors = colors.cpu()
    # # find the median of each column
    # median = np.median(colors, axis=0) # (3,)
    # # find the covariance matrix of the colors values
    # cov = np.cov(colors.T) # (3, 3)
    # # inverse of the covariance matrix
    # cov_inv = np.linalg.inv(cov) # (3, 3)
    # # find the mahalanobis distance of each point from the median
    # mahalanobis = scipy.spatial.distance.cdist(colors, [median], metric='mahalanobis', VI=cov_inv) # (N, 1)
    # mahalanobis_similarity = 1 / (1 + mahalanobis) # (N, 1)

    # world_xyz = []
    # for i in range(num_image):
        
    #     fx = pipeline.datamanager.expanded_cameras[i].fx
    #     fy = pipeline.datamanager.expanded_cameras[i].fy
    #     cx = pipeline.datamanager.expanded_cameras[i].cx
    #     cy = pipeline.datamanager.expanded_cameras[i].cy
    #     c2w = pipeline.datamanager.expanded_cameras[i].camera_to_worlds
    #     depth = output[i].to(fx.device)
    #     y = pipeline.datamanager.ray_indices[i][1]
    #     x = pipeline.datamanager.ray_indices[i][2]
    #     # xyz in camera coordinates
    #     X = (x - cx) * depth / fx
    #     # Y = (y - cy) * depth / fy
    #     # Z = depth
    #     Y = -(y - cy) * depth / fy
    #     Z = -depth
    #     # Convert to world coordinates
    #     camera_xyz = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1)
    #     c2w = c2w.to(camera_xyz.device)
    #     #world_xyz.append((c2w @ camera_xyz.T).T[..., :3])
    #     world_coordinates = (c2w @ camera_xyz.T).T[..., :3]
    #     world_xyz.append(world_coordinates)
    
    # print(len(world_xyz))
    # # calculate the plane equation using linear regression
    # # Flatten the world_xyz list and convert it to a numpy array
    # world_xyz_np = np.concatenate([xyz.cpu().numpy() for xyz in world_xyz], axis=0)

    # Create a LinearRegression object
    # reg = LinearRegression()
    # reg = TheilSenRegressor(random_state=0)
    # use a Huber regressor
    # reg = HuberRegressor()
    reg = HuberRegressor(fit_intercept=False)

    # filter out points with mahalanobis similarity less than some threshold
    # similarity_threshold = 0.6
    similarity_threshold = 0.5
    n_points = world_xyz_np.shape[0]
    world_xyz_np = world_xyz_np[mahalanobis_similarity.flatten() > similarity_threshold]
    mahalanobis_similarity = mahalanobis_similarity[mahalanobis_similarity.flatten() > similarity_threshold]
    print(f"Filtering out points with color mahalanobis similarity less than {similarity_threshold}, number of remaining points: {world_xyz_np.shape[0]}/{n_points}")

    # Fit the model to the data
    # ax + by + cz + d = 0
    # reg.fit(world_xyz_np[:, :2], world_xyz_np[:, 2])
    # reg.fit(world_xyz_np[:, :2], world_xyz_np[:, 2], sample_weight=mahalanobis_similarity.flatten())
    reg.fit(world_xyz_np, np.ones(world_xyz_np.shape[0]), sample_weight=mahalanobis_similarity.flatten())
    print(f"Used {reg.__class__.__name__} weighted by mahalanobis similarity")

    # The coefficients a, b are in reg.coef_, and the intercept d is in reg.intercept_
    # a, b = reg.coef_
    # d = reg.intercept_
    # c = -1
    a, b, c = reg.coef_
    d = -1

    # vertices = pipeline.datamanager.vertices
    # print("Vertices of bbox\n")
    # print(vertices)

    # read object_occupancy from datamanager as numpy on cpu
    object_occupancy = pipeline.datamanager.object_occupancy.cpu().numpy()

    # read aabb from occupancy grid as numpy on cpu
    object_aabb = pipeline.datamanager.object_aabb.cpu().numpy()
    min_point = object_aabb[0]
    max_point = object_aabb[1]
    # get the vertices of the aabb, in the order of     
    # edges = {
    #     "x": [(0, 4), (1, 5), (2, 6), (3, 7)],
    #     "y": [(0, 2), (1, 3), (4, 6), (5, 7)],
    #     "z": [(0, 1), (2, 3), (4, 5), (6, 7)]
    # }
    vertices = np.array([[min_point[0], min_point[1], min_point[2]],
                         [min_point[0], min_point[1], max_point[2]],
                         [min_point[0], max_point[1], min_point[2]],
                         [min_point[0], max_point[1], max_point[2]],
                         [max_point[0], min_point[1], min_point[2]],
                         [max_point[0], min_point[1], max_point[2]],
                         [max_point[0], max_point[1], min_point[2]],
                         [max_point[0], max_point[1], max_point[2]]])
    
    # read object_obb from datamanager as numpy on cpu
    # object_obb = pipeline.datamanager.object_obb
    # obb_vertices = object_obb.get_corners().cpu().numpy()
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('bbox with transformation')
    ## Plot the points in world_xyz
    #for xyz in world_xyz:
    #    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # Plot the plane
    # xx, yy = np.meshgrid(range(-5, 5), range(-5, 5))
    xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    # # plot the reference plane
    # xx1, yy1 = np.meshgrid(range(-5, 5), range(-5, 5))
    # zz1 = (0.7365434765815735 * xx1 + 0.14943844079971313 * yy1 + 0.01226318534463644)/(-1)
    # ax.plot_surface(xx1, yy1, zz1, alpha=0.5, color = 'r')

    
    #print(vertices.shape) # 8 x 3
    #TODO: plot the vertices
    # Plot the vertices
    # for vertex in vertices:
    #     ax.scatter(*vertex)

    # Plot the bbox
    edges = {
        "x": [(0, 4), (1, 5), (2, 6), (3, 7)],
        "y": [(0, 2), (1, 3), (4, 6), (5, 7)],
        "z": [(0, 1), (2, 3), (4, 5), (6, 7)]
    }
    # Plot the edges of the bbox
    for direction, edge_indices in edges.items():
        for i, j in edge_indices:
            # Get the starting and ending vertices for this edge
            starting_vertex = vertices[i]
            ending_vertex = vertices[j]
            # Plot the edge
            ax.plot([starting_vertex[0], ending_vertex[0]], [starting_vertex[1], ending_vertex[1]], 
                    [starting_vertex[2], ending_vertex[2]], color="red")    

    # # Plot the edges of the obb
    # for direction, edge_indices in edges.items():
    #     for i, j in edge_indices:
    #         # Get the starting and ending vertices for this edge
    #         starting_vertex = obb_vertices[i]
    #         ending_vertex = obb_vertices[j]
    #         # Plot the edge
    #         ax.plot([starting_vertex[0], ending_vertex[0]], [starting_vertex[1], ending_vertex[1]], 
    #                 [starting_vertex[2], ending_vertex[2]], color="red")    

    # plt.show()
    
    # The equation of the plane is `ax + by + cz + d = 0`
    CONSOLE.print(f"The points used for the plane equation are: {world_xyz_np}")
    CONSOLE.print(f"The equation of the plane is {a}x + {b}y + {c}z + {d} = 0")
    CONSOLE.print(f"The object aabb vertices are {vertices}")
    # CONSOLE.print(f"The object obb vertices are {obb_vertices}")

    # config.set_timestamp()
    plot_dir = os.path.join(str(load_dir).replace('nerfstudio_models', ''), 'wandb/plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # save the object occupancy grid as npy file
    occupancy_path = os.path.join(plot_dir, f"object_occupancy.npy")
    np.save(occupancy_path, object_occupancy)
    print(f"Saved the object occupancy grid to {occupancy_path}")
    # save the points for the plane equation as npy file
    samples_path = os.path.join(plot_dir, f"plane_samples.npy")
    np.save(samples_path, world_xyz_np)
    print(f"Saved the plane samples to {samples_path}")
    # save the colors of the points as npy file
    colors_path = os.path.join(plot_dir, f"sample_colors.npy")
    np.save(colors_path, colors_np)
    print(f"Saved the colors of the points to {colors_path}")
    # save the coefficients of the plane equation as npy file
    plane_path = os.path.join(plot_dir, f"plane_coefficients.npy")
    np.save(plane_path, np.array([a, b, c, d]))
    print(f"Saved the plane coefficients to {plane_path}")
    # save the aabb as npy file
    aabb_path = os.path.join(plot_dir, f"aabb.npy")
    np.save(aabb_path, object_aabb)
    print(f"Saved the aabb to {aabb_path}")

    bbox_intersections = derive_nsa(a, b, c, d, vertices)
    # save the intersections as npy file
    intersections_path = os.path.join(plot_dir, f"aabb_intersections.npy")
    np.save(intersections_path, np.array(bbox_intersections))
    print(f"The aabb intersections are: {bbox_intersections}")
    print(f"Saved the aabb intersections to {intersections_path}")

    # obb_intersections = derive_nsa(a, b, c, d, obb_vertices)
    # # save the intersections as npy file
    # obb_intersections_path = os.path.join(plot_dir, f"obb_intersections.npy")
    # np.save(obb_intersections_path, np.array(obb_intersections))
    # print(f"The obb intersections are: {obb_intersections}")
    # print(f"Saved the obb intersections to {obb_intersections_path}")

    # plot the intersections in the 3D plot
    for intersection in bbox_intersections:
        ax.scatter(*intersection, color="green")

    # # plot the obb intersections in the 3D plot
    # for intersection in obb_intersections:
    #     ax.scatter(*intersection, color="blue", marker="x")

    # save the 3D plot locally
    plot_path = os.path.join(plot_dir, f"nsa_plot.png")
    plt.savefig(plot_path)
    CONSOLE.print(f"Saved the NSA plot to {plot_path}")



def derive_nsa(a, b, c, d, vertices):
    # Initialize the list of intersections
    intersections = []

    # Define the edges
    edges = {
        "x": [(0, 4), (1, 5), (2, 6), (3, 7)],
        "y": [(0, 2), (1, 3), (4, 6), (5, 7)],
        "z": [(0, 1), (2, 3), (4, 5), (6, 7)]
    }

    # Iterate over each edge
    for direction, edge_indices in edges.items():
        if direction == "x":
            directional_vector = vertices[4] - vertices[0]
        elif direction == "y":
            directional_vector = vertices[2] - vertices[0]
        else:  # direction == "z"
            directional_vector = vertices[1] - vertices[0]

        for i, j in edge_indices:
            # Get the starting vertex and directional vector for this edge
            starting_vertex = vertices[i]

            # Solve for t
            t = -(a * starting_vertex[0] + b * starting_vertex[1] + c * starting_vertex[2] + d) / (a * directional_vector[0] + b * directional_vector[1] + c * directional_vector[2])

            # If t is in the range [0, 1], the edge intersects with the plane
            if 0 <= t <= 1:
                # Calculate the 3D coordinate of the intersection point
                intersection = starting_vertex + t * directional_vector
                # Append the intersection to the list of intersections
                intersections.append(intersection)

    # If no intersections were found, project the bbox along the z axis (strong assumption that the z axis is the vertical axis)
    if len(intersections) == 0:
        print("No intersection found, will project along the z axis")
        direction = "z"
        edge_indices = edges[direction]
        directional_vector = vertices[1] - vertices[0]
        for i, j in edge_indices:
            # Get the starting vertex and directional vector for this edge
            starting_vertex = vertices[i]

            # Solve for t
            t = -(a * starting_vertex[0] + b * starting_vertex[1] + c * starting_vertex[2] + d) / (a * directional_vector[0] + b * directional_vector[1] + c * directional_vector[2])

            # Calculate the 3D coordinate of the intersection point
            intersection = starting_vertex + t * directional_vector
            # Append the intersection to the list of intersections
            intersections.append(intersection)

    # Return the list of intersections
    return intersections

def main(config: TrainerConfig) -> None:
    """Main function."""
    # config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
    # assert self.output_path.suffix == ".json"
    # if self.render_output_path is not None:
    #     self.render_output_path.mkdir(parents=True)
    # metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
    # self.output_path.parent.mkdir(parents=True, exist_ok=True)
    # # Get the output and define the names to save to
    
    # # Save output to output file
    # self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
    # CONSOLE.print(f"Saved results to: {self.output_path}")
    # config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)     #setting up the models

    # camera_indices, pixel_coords = self.sample_pixels()
    # raybundles = pipeline.datamanager.train_dataset.cameras._generate_rays_from_coords(camera_indices=camera_indices,coords=pixel_coords,disable_distortion = True)
    # ## raybundles reshape
    # renderer_depth = DepthRenderer(method="median")
    # outputs[f"prop_depth_{i}"] = renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    launch(
        main_func=plane_estimation,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )





def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
