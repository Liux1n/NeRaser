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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
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
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator, RayGenerator_surface_detection
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import get_orig_class

# new
from nerfstudio.data.scene_box import OrientedBox
import numpy as np
from itertools import compress
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time


def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    imgdata_lists = defaultdict(list)
    for data in batch:
        image = data.pop("image")
        images.append(image)
        topop = []
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                # if the value has same height and width as the image, assume that it should be collated accordingly.
                if len(val.shape) >= 2 and val.shape[:2] == image.shape[:2]:
                    imgdata_lists[key].append(val)
                    topop.append(key)
        # now that iteration is complete, the image data items can be removed from the batch
        for key in topop:
            del data[key]

    new_batch = nerfstudio_collate(batch)
    new_batch["image"] = images
    new_batch.update(imgdata_lists)

    return new_batch


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    masks_on_gpu: bool = False
    """Process masks on GPU for speed at the expense of memory, if True."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. A Raybundle: This will contain the rays we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Retrieve the next eval image.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the step number, the ray bundle for the image, and a dictionary of
            additional batch information such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""
    camera_optimizer: Optional[CameraOptimizerConfig] = field(default=None)
    """Deprecated, has been moved to the model config."""
    pixel_sampler: PixelSamplerConfig = PixelSamplerConfig()
    """Specifies the pixel sampler used to sample pixels from images."""

    def __post_init__(self):
        """Warn user of camera optimizer change."""
        if self.camera_optimizer is not None:
            import warnings

            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class VanillaDataManager(DataManager, Generic[TDataset]):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
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

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        
        print('1111', self.train_dataset[0])

        #surface detection
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






        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

        # TODO: initialize a 3D grid "self.object_grid" with specified resolution (can start with coarser ones, e.g. 16) inside the scene_box (can be extracted from dataparser_outputs)
        # where each vertex stores a boolean indicating objectness (whether it is inside the masked object)
        # add a new method "object_mask_from_2d_masks", where vertices are projected (using projection matrix derived from dataparser_outputs.cameras) 
        # to all 2D image planes to identify those falling into all 2D masks
        # can consider tricks such as coarse-to-fine or sorting the 2D mask areas

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default
    

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


    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
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
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
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
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))




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
            
        
        
        y_sample_range = 60
        num_samples = 50

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
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
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

    # TODO: query self.object_grid and modify the "ray_bundle.near" attribute to avoid the object
    # add a new method "mask_ray_bundle" that samples from ray bundles and query self.object_grid and output a "masked" ray bundle with its "near" modified
    # Then the object should be skipped at least while rendering eval images
    # If successful, do the same thing for renderer and viewer (not sure if they use this method but quite sure they also use a datamanager)
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
