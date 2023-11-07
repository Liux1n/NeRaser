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
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE


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

        # TODO: as an argument / derive it here
        # hardcoded object_aabb for lego
        # self.object_aabb = torch.tensor([[-1.58240465, -2.3752433 , -3.95161623],
        #                                  [ 0.92683118,  0.15093034, -0.33374367]])
        self.object_aabb = torch.tensor([[-1.58240465, -0.15093034, 0.33374367],
                                         [ 0.92683118, 2.3752433, 3.95161623]])
        
        # transform using dataparser_transform and dataparser_scale
        transform_matrix = self.train_dataparser_outputs.dataparser_transform
        scale_factor = self.train_dataparser_outputs.dataparser_scale

        # Extract min and max points
        min_point = self.object_aabb[0]
        max_point = self.object_aabb[1]

        # Compute all 8 vertices of the AABB
        bbox_vertices = torch.tensor([
            [min_point[0], min_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], max_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [max_point[0], max_point[1], max_point[2]]
        ])
        homogeneous_vertices = torch.cat((bbox_vertices, torch.ones(bbox_vertices.size(0), 1)), dim=1)

        transformed_vertices = (transform_matrix @ homogeneous_vertices.T).T
        scaled_transformed_vertices = transformed_vertices * scale_factor
        # print(scaled_transformed_vertices)
        # tensor([[ 0.4171, -0.2363,  0.7499],
        #         [-0.1194, -0.2066,  0.3771],
        #         [ 0.6781, -0.2254,  0.3753],
        #         [ 0.1415, -0.1958,  0.0024],
        #         [ 0.4279,  0.2168,  0.7705],
        #         [-0.1087,  0.2464,  0.3976],
        #         [ 0.6888,  0.2276,  0.3958],
        #         [ 0.1523,  0.2572,  0.0230]])

        # construct new aabb (now it's bbox of bbox)
        # TODO: use oriented bbox to further refine
        new_min_point = torch.min(scaled_transformed_vertices, dim=0)[0]
        new_max_point = torch.max(scaled_transformed_vertices, dim=0)[0]
        self.object_aabb = torch.vstack([new_min_point, new_max_point])


        # TODO: initialize a 3D grid "self.object_grid" with specified resolution (can start with coarser ones, e.g. 16) inside the scene_box (can be extracted from dataparser_outputs)
        # where each vertex stores a boolean indicating objectness (whether it is inside the masked object)
        # add a new method "object_mask_from_2d_masks", where vertices are projected (using projection matrix derived from dataparser_outputs.cameras) 
        # to all 2D image planes to identify those falling into all 2D masks
        # can consider tricks such as coarse-to-fine or sorting the 2D mask areas

        # train_dataparser_outputs.scene_box.aabb: tensor([[-1., -1., -1.],
        # [ 1.,  1.,  1.]])

        # train_dataparser_outputs.dataparser_transform: tensor([[ 0.0237,  0.5714, -0.8204,  0.4603],
        # [ 0.9987,  0.0237,  0.0453,  0.5088],
        # [ 0.0453, -0.8204, -0.5701,  0.0188]])  

        # train_dataparser_outputs.dataparser_scale: 0.18077436331220487


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
        
        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        # check if spawn is already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        super().__init__()

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
