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

"""
Parallel data manager that generates training data in multiple python processes.
"""
from __future__ import annotations

import concurrent.futures
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.multiprocessing as mp
from rich.progress import track
from torch.nn import Parameter

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    VanillaDataManagerConfig,
    TDataset,
    variable_res_collate,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
    PixelSamplerConfig,
    PatchPixelSamplerConfig,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator, RayGenerator_surface_detection
from nerfstudio.utils.rich_utils import CONSOLE

# new
from nerfstudio.data.scene_box import OrientedBox
import numpy as np
from itertools import compress
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time


@dataclass
class ParallelDataManagerConfig(VanillaDataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

    _target: Type = field(default_factory=lambda: ParallelDataManager)
    """Target class to instantiate."""
    num_processes: int = 1
    """Number of processes to use for train data loading. More than 1 doesn't result in that much better performance"""
    queue_size: int = 2
    """Size of shared data queue containing generated ray bundles and batches.
    If queue_size <= 0, the queue size is infinite."""
    max_thread_workers: Optional[int] = None
    """Maximum number of threads to use in thread pool executor. If None, use ThreadPool default."""


class DataProcessor(mp.Process):
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
    """

    def __init__(
        self,
        out_queue: mp.Queue,
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        dataset: TDataset,
        pixel_sampler: PixelSampler,
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.dataset = dataset
        self.exclude_batch_keys_from_device = self.dataset.exclude_batch_keys_from_device
        self.pixel_sampler = pixel_sampler
        self.ray_generator = RayGenerator(self.dataset.cameras)
        self.cache_images()

    def run(self):
        """Append out queue in parallel with ray bundles and batches."""
        while True:
            batch = self.pixel_sampler.sample(self.img_data)
            ray_indices = batch["indices"]
            ray_bundle: RayBundle = self.ray_generator(ray_indices)
            # check that GPUs are available
            if torch.cuda.is_available():
                ray_bundle = ray_bundle.pin_memory()
            while True:
                try:
                    self.out_queue.put_nowait((ray_bundle, batch))
                    break
                except queue.Full:
                    time.sleep(0.0001)
                except Exception:
                    CONSOLE.print_exception()
                    CONSOLE.print("[bold red]Error: Error occured in parallel datamanager queue.")

    def cache_images(self):
        """Caches all input images into a NxHxWx3 tensor."""
        indices = range(len(self.dataset))
        batch_list = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description="Loading data batch", transient=False):
                batch_list.append(res.result())
        self.img_data = self.config.collate_fn(batch_list)


class ParallelDataManager(DataManager, Generic[TDataset]):
    """Data manager implementation for parallel dataloading.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def __init__(
        self,
        config: ParallelDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.dataset_type: Type[TDataset] = kwargs.get("_dataset_type", getattr(TDataset, "__default__"))
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        cameras = self.train_dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device

                
        # # refined oriented box
        # # another way to get R (Rodrigues)
        # target_z = scaled_transformed_vertices[1] - scaled_transformed_vertices[0]
        # normalized_target_z = target_z / torch.linalg.norm(target_z)
        # source_z = torch.tensor([0., 0., 1.])
        # # print(source_z, normalized_target_z)
        # v = torch.cross(source_z, normalized_target_z)
        # c = torch.dot(source_z, normalized_target_z)
        # s = v.norm()
        # kmat = torch.tensor([[0, -v[2], v[1]],
        #                     [v[2], 0, -v[0]],
        #                     [-v[1], v[0], 0]])
        # obb_R = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2)) #obb_R.shape

        # # get S
        # target_x = scaled_transformed_vertices[4] - scaled_transformed_vertices[0]
        # target_y = scaled_transformed_vertices[2] - scaled_transformed_vertices[0]
        # obb_S = torch.tensor([torch.linalg.norm(target_x), torch.linalg.norm(target_y), torch.linalg.norm(target_z)])

        # # get T
        # obb_T = torch.mean(scaled_transformed_vertices, dim=0)

        # self.object_obb = OrientedBox(R=obb_R, T=obb_T, S=obb_S)


        # train_dataparser_outputs.scene_box.aabb: tensor([[-1., -1., -1.],
        # [ 1.,  1.,  1.]])

        # train_dataparser_outputs.dataparser_transform: tensor([[ 0.0237,  0.5714, -0.8204,  0.4603],
        # [ 0.9987,  0.0237,  0.0453,  0.5088],
        # [ 0.0453, -0.8204, -0.5701,  0.0188]])  

        # train_dataparser_outputs.dataparser_scale: 0.18077436331220487

        # eval_dataparser_outputs.dataparser_transform: tensor([[ 0.0237,  0.5714, -0.8204,  0.4603],
        # [ 0.9987,  0.0237,  0.0453,  0.5088],
        # [ 0.0453, -0.8204, -0.5701,  0.0188]])

        # eval_dataparser_outputs.dataparser_scale: 0.18077436331220487

        # self.train_dataparser_outputs.mask_filenames see base_dataset.py
        
              
        # train_dataparser_outputs.cameras.camera_to_worlds.shape: torch.Size([92, 3, 4])
        # eval_dataparser_outputs.cameras.camera_to_worlds.shape: torch.Size([10, 3, 4])

        # train_dataparser_outputs.cameras.camera_to_worlds: tensor([[[-0.6875, -0.2085,  0.6956,  0.4388],
        #  [ 0.7217, -0.3026,  0.6226,  0.5472],
        #  [ 0.0807,  0.9300,  0.3585, -0.1125]],

        # [[-0.7197, -0.2948,  0.6286,  0.4750],
        #  [ 0.6891, -0.4142,  0.5947,  0.5637],
        #  [ 0.0850,  0.8611,  0.5012,  0.0806]],

        # [[-0.7125, -0.4204,  0.5617,  0.4329],
        #  [ 0.6960, -0.5244,  0.4904,  0.4963],
        #  [ 0.0884,  0.7404,  0.6663,  0.2943]],

        # ...,

        # [[ 0.6586,  0.0963, -0.7463, -0.8793],
        #  [-0.7522,  0.0567, -0.6565, -0.5146],
        #  [-0.0209,  0.9937,  0.1098, -0.3204]],

        # [[ 0.9955, -0.0134,  0.0938, -0.0869],
        #  [ 0.0888,  0.4768, -0.8745, -0.7867],
        #  [-0.0330,  0.8789,  0.4759,  0.0319]],

        # [[-0.4821, -0.1433,  0.8643,  0.6406],
        #  [ 0.8727, -0.1656,  0.4594,  0.4534],
        #  [ 0.0773,  0.9757,  0.2049, -0.1915]]])

        # Getting Object Occupancy Grid
        self.grid_resolution = 128 # 256 CUDA out of memory
        self.threshold = 0.9
        self.object_occupancy = self.object_mask_from_2d_masks(resolution=self.grid_resolution, threshold=self.threshold)        
        num_occupied_voxels = torch.sum(self.object_occupancy)
        print(f"number of occupied voxels: {num_occupied_voxels}")

        # Optionally save object_occupancy for visualization
        np.save("object_occupancy.npy", self.object_occupancy.cpu().numpy())
        print("Saved object_occupancy.npy")

        # get new self.object_aabb by finding the min and max points of the object_occupancy grid
        self.occupied_coordinates = self.voxel_coords[:, self.object_occupancy]
        # print(f"self.occupied_coordinates.shape: {self.occupied_coordinates.shape}")
        min_point = torch.min(self.occupied_coordinates, dim=1)[0]
        max_point = torch.max(self.occupied_coordinates, dim=1)[0]
        self.object_aabb = torch.vstack([min_point, max_point])
        print(f"self.object_aabb derived from occupancy grid: {self.object_aabb}")
        torch.cuda.empty_cache()

        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        # check if spawn is already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        super().__init__()

    # define a new method "object_mask_from_2d_masks" that:
    # 1. initialize a 3D grid "self.object_grid" with specified resolution (e.g. 16) bounded by train_dataparser_outputs.scene_box.aabb ([-1, 1] on each axis)
    # 2. project all vertices of "self.object_grid" to all 2D image planes (using information from train_dataparser_outputs.cameras) 
    # 3. Sample the corresponding 2D masks at the projected points 
    # 4. define a new tensor "self.objectness_grid" of shape [resolution, resolution, resolution] storing the proportion of the projected points of each vertex fall in the "False" area of the 2D mask
    # 5. return a boolean tensor "object_occupancy" of the same shape as "self.objectness_grid" where each vertex is True if the corresponding vertex in "self.objectness_grid" is higher than a specified threshold (e.g. 0.9), False otherwise
    def object_mask_from_2d_masks(self, resolution=16, threshold=1):
        start_time = time.time()
        # 1. Initialize a 3D grid
        print(f"Initializing a 3D grid with resolution {resolution}...")
        self.voxel_coords = self.initialize_grid(resolution, self.train_dataparser_outputs.scene_box.aabb)

        # camera extrinsics and intrinsics
        cameras = self.train_dataparser_outputs.cameras
        c2w = cameras.camera_to_worlds.to(self.device)
        # make c2w homogeneous
        c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=self.device)], dim=1)
        c2w[:, 3, 3] = 1
        K = cameras.get_intrinsics_matrices().to(self.device)

        # mask images
        mask_paths = self.train_dataparser_outputs.mask_filenames
        mask_images = []
        for mask_path in mask_paths:
            mask_tensor = get_image_mask_tensor_from_path(mask_path)
            mask_images.append(mask_tensor)
        mask_images = torch.stack(mask_images, dim=0).to(self.device).permute(0, 3, 1, 2) # shape (N, 1, H, W)

        # 2. Project all vertices of the grid to all 2D image planes
        batch_size = c2w.shape[0]
        image_size = torch.tensor([mask_images.shape[-1], mask_images.shape[-2]], device=self.device)  # [width, height]
        print(f"Projecting all vertices of the grid to all {batch_size} 2D image planes...")

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat([voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0)
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        voxel_world_coords = voxel_world_coords.expand(batch_size, *voxel_world_coords.shape[1:])  # [batch, 4, N]
        voxel_cam_coords = torch.bmm(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]

        # TODO: check if this is correct
        # flip the z axis
        voxel_cam_coords[:, 2, :] = -voxel_cam_coords[:, 2, :]
        # flip the y axis
        voxel_cam_coords[:, 1, :] = -voxel_cam_coords[:, 1, :]

        voxel_cam_coords_z = voxel_cam_coords[:, 2:3, :]
        voxel_cam_points = torch.bmm(K, voxel_cam_coords[:, 0:3, :] / voxel_cam_coords_z)  # [batch, 3, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :]  # [batch, 2, N]

        # 3. Sample the corresponding 2D masks at the projected points
        print("Sampling the corresponding 2D masks at the projected points...")
        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        # sample masks
        sampled_masks = F.grid_sample(input=mask_images.float(), grid=grid, mode="nearest", padding_mode="border", align_corners=False) # [batch, 1, 1, N]

        # 4. Define a new tensor storing the proportion of the projected points that fall in the "False" area of the 2D mask
        self.objectness_grid = self.calculate_objectness(sampled_masks, resolution)

        # 5. Return a boolean tensor where each vertex is True if the corresponding vertex in the objectness grid is higher than a specified threshold
        object_occupancy = self.objectness_grid >= threshold

        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time} seconds.")
        return object_occupancy

    def initialize_grid(self, resolution, aabb):
        # Initialize a 3D grid with the specified resolution and bounding box
        origin = aabb[0]
        voxel_size = (aabb[1] - aabb[0]) / resolution
        xdim = torch.arange(resolution)
        ydim = torch.arange(resolution)
        zdim = torch.arange(resolution)
        grid = torch.stack(torch.meshgrid([xdim, ydim, zdim], indexing="ij"), dim=0)
        voxel_coords = origin.view(3, 1, 1, 1) + grid * voxel_size.view(3, 1, 1, 1)
        return voxel_coords.to(self.device)

    def calculate_objectness(self, sampled_masks, resolution):
        # Define a new tensor storing the proportion of the projected points that fall in the "False" area of the 2D mask
        objectness = (sampled_masks == 0).float().mean(dim=0)
        return objectness.view(resolution, resolution, resolution)

    def plot_object_occupancy(self, resolution=16):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D grid
        grid = torch.linspace(-1, 1, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(grid, grid, grid)

        # Get the vertices where object_occupancy is True
        occupied = self.object_occupancy.cpu()
        x = grid_x[occupied].numpy()
        y = grid_y[occupied].numpy()
        z = grid_z[occupied].numpy()

        # Plot the vertices
        ax.scatter(x, y, z, c='b')

        # Save the plot
        plt.savefig("object_occupancy.png")
        print("Saved object_occupancy.png")

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training."""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation."""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular, num_rays_per_batch=num_rays_per_batch
        )

    def setup_train(self):
        """Sets up parallel python data processes for training."""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.data_queue = mp.Manager().Queue(maxsize=self.config.queue_size)
        self.data_procs = [
            DataProcessor(
                out_queue=self.data_queue,  # type: ignore
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
            )
            for i in range(self.config.num_processes)
        ]
        for proc in self.data_procs:
            proc.start()
        print("Started threads")

        # Prime the executor with the first batch
        self.train_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers)
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)

    def setup_eval(self):
        """Sets up the data loader for evaluation."""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)  # type: ignore
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))

        '''
        #TODO: surface detection

        # generate ray for surface detection from evaluation dataset
        
        self.surface_detection_dataset = self.eval_dataset

        self.mask = [item for item in self.surface_detection_dataset] 
        self.surface_detection_pixel_sampler = self._get_pixel_sampler(self.surface_detection_dataset, self.config.eval_num_rays_per_batch)
        
        #print(self.mask[0]['mask'].shape) # torch.Size([764, 1015, 1])
        # white is 1, black is 0
        y_outbound = self.mask[0]['mask'].shape[0] # 764

        for i, item in enumerate(self.mask):
            mask_array = item['mask'].numpy().squeeze()
            y, x = np.where(mask_array == 0)
            max_y = np.max(y)
            max_x = np.max(x)
            min_x = np.min(x)
            width = max_x - min_x
            bottom_pixel = (max_y+5, x[np.argmax(y)]) # find a pixel that is below the mask
            self.mask[i]['bottom_pixel'] = bottom_pixel
            self.mask[i]['width'] = width

        # filter out out-of-bound indices
        self.surface_detection_camera = self.surface_detection_dataset.cameras
        
        #print(len(self.surface_detection_camera))
        #print(self.surface_detection_camera[0])
        mask_dataset = np.array([item['bottom_pixel'][0] < y_outbound for item in self.mask])
        # Get the indices of the items to keep
        indices_to_keep = [i for i, mask in enumerate(mask_dataset) if mask]

        # Store the indices to keep
        self.indices_to_keep = indices_to_keep
        #print(self.indices_to_keep)
        #Create a filtered list of cameras
        self.filtered_cameras = [self.surface_detection_camera[i] for i in self.indices_to_keep]
        #print(filtered_cameras)
        # create ray indices from bottom pixels, 3 rays per indices
        self.ray_indices = torch.zeros((3*len(self.mask), 3),dtype=int)
        offset = 0
        for i, item in enumerate(self.mask):
  
            self.ray_indices[offset][0] = int(item['image_idx']) # camera index
            self.ray_indices[offset][1] = int(item['bottom_pixel'][0]) # y
            self.ray_indices[offset][2] = int(item['bottom_pixel'][1]) # x
            self.ray_indices[offset+1][0] = int(item['image_idx'])
            self.ray_indices[offset+1][1] = int(item['bottom_pixel'][0]) # y
            self.ray_indices[offset+1][2] = int(item['bottom_pixel'][1]) + item['width']/8 # x
            self.ray_indices[offset+2][0] = int(item['image_idx'])
            self.ray_indices[offset+2][1] = int(item['bottom_pixel'][0]) # y
            self.ray_indices[offset+2][2] = int(item['bottom_pixel'][1]) - item['width']/8 # x
            offset += 3

        mask = self.ray_indices[:, 1] < y_outbound

        # Use the mask to filter self.ray_indices
        self.ray_indices = self.ray_indices[mask]
        #print('ray indices',self.ray_indices)
        #print(len(self.surface_detection_dataset)) # 9
        #print(len(self.surface_detection_camera))
        self.surface_detection_ray_generator = RayGenerator(self.surface_detection_camera.to(self.device))


        # create ray bundle from ray indices
        self.ray_bundle_surface_detection = self.surface_detection_ray_generator(self.ray_indices)

        # expand self.filtered_cameras to align with self.ray_indices
        self.expanded_cameras = [camera for camera in self.filtered_cameras for _ in range(3)]
        '''

        #TODO: surface detection

        # generate ray for surface detection from evaluation dataset
        
        #self.surface_detection_dataset = self.eval_dataset
        self.surface_detection_dataset = self.train_dataset


        self.mask = [item for item in self.surface_detection_dataset] 
        self.surface_detection_pixel_sampler = self._get_pixel_sampler(self.surface_detection_dataset, self.config.eval_num_rays_per_batch)
        
        #print(self.mask[0]['mask'].shape) # torch.Size([764, 1015, 1])
        # white is 1, black is 0
        y_outbound = self.mask[0]['mask'].shape[0] # 764

        for i, item in enumerate(self.mask):
            mask_array = item['mask'].numpy().squeeze()
            y, x = np.where(mask_array == 0)
            max_y = np.max(y)
            #max_x = np.max(x)
            #min_x = np.min(x)
            #width = max_x - min_x
            bottom_pixel = (max_y, x[np.argmax(y)]) # findv the bottom pixel of the mask
            y_lower = bottom_pixel[0] - 5  # Subtract 5 from the y-coordinate of bottom_pixel
             # Find the x-coordinate at this new y-coordinate

            above_bottom_pixel = (y_lower, bottom_pixel[1])
     
            # Find the indices where y is between y_up_bottom and y_bottom
            indices = np.where((y >= above_bottom_pixel[0]) & (y <= bottom_pixel[0]))

            # Extract the corresponding x values
            x_values = x[indices]

            # Compute the width as the difference between the maximum and minimum x values
            width = np.max(x_values) - np.min(x_values)
            #TODO: find the width of the area between y_bottom and y_up_bottom  
            self.mask[i]['bottom_pixel'] = bottom_pixel
            #print(bottom_pixel)
            #self.mask[i]['y_up_bottom'] = y_up_bottom
            self.mask[i]['width'] = width
            #self.mask[i]['width'] = 2
            
        
        
        y_sample_range = 25
        num_samples = 30

        # filter out out-of-bound indices
        self.surface_detection_camera = self.surface_detection_dataset.cameras

        # Initialize an empty numpy array
        all_points_np = np.empty((0, 3))        
        
        camera_list = []
        image_list = []

        for i, item in enumerate(self.mask):

            idx = item['image_idx']
            bottom_y = int(item['bottom_pixel'][0])
            bottom_x = int(item['bottom_pixel'][1])
            width = int(item['width'])
            camera = self.surface_detection_camera[idx]
            image = self.surface_detection_dataset[idx]
            # Append the camera to the camera_list
            camera_list.append([camera] * num_samples)
            image_list.append([image] * num_samples)
            # Generate an idx array of the same length as points
            idx_array = np.full((num_samples, 1), idx)
            # Generate random x coordinates within the width
            x_samples = np.random.randint(low= bottom_x - width/2, high= bottom_x + width/2, size=num_samples)

            # Generate random y coordinates between y_up_bottom and y_bottom
            y_samples = np.random.randint(low=bottom_y + y_sample_range -10, high=bottom_y + y_sample_range, size=num_samples)
            

            # Combine the idx, x and y coordinates into a 2D array
            points = np.column_stack((idx_array, x_samples, y_samples))
            #print(points.shape)
            # Concatenate the points to all_points_np
            all_points_np = np.concatenate((all_points_np, points), axis=0)
            #print(all_points_np.shape)
        # Flatten the camera_list
        camera_list = [camera for sublist in camera_list for camera in sublist]
        image_list = [image for sublist in image_list for image in sublist]
        
        #print(all_points_np)
        #raise NotImplementedError
        

        self.ray_indices = torch.from_numpy(all_points_np).int()
        mask = self.ray_indices[:, 1] + y_sample_range < y_outbound

        #print(self.ray_indices)
        #mask = self.ray_indices[:, :1] + y_sample_range < y_outbound

        camera_list = list(compress(camera_list, mask))

        camera_list = [camera.to(self.device) for camera in camera_list]
        image_list = list(compress(image_list, mask))

        self.surface_detection_ray_generator = RayGenerator_surface_detection(self.train_dataset.cameras.to(self.device))

        # create ray bundle from ray indices
        self.ray_bundle_surface_detection = self.surface_detection_ray_generator(self.ray_indices, mask = mask)

        # expand self.filtered_cameras to align with self.ray_indices
        #self.expanded_cameras = [camera for camera in self.filtered_cameras for _ in range(num_samples)]
        self.expanded_cameras = camera_list
        self.filtered_data = image_list
        #print(self.ray_indices.shape, len(self.expanded_cameras))
        #print(self.ray_indices.shape, len(camera_list))




        #######################################################################
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            object_aabb=self.object_aabb, # new
            # object_obb=self.object_obb, # new
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the parallel training processes."""
        self.train_count += 1

        # Fetch the next batch in an executor to parallelize the queue get() operation
        # with the train step
        bundle, batch = self.train_batch_fut.result()
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Retrieve the next eval image."""
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def __del__(self):
        """Clean up the parallel data processes."""
        if hasattr(self, "data_procs"):
            for proc in self.data_procs:
                proc.terminate()
                proc.join()