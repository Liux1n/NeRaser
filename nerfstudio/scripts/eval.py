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
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    
    # if True, also run eval on training data
    render_all_images: bool = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)

        # save the c2w of every image in the dataset
        correct_c2w = list()
        def save_c2w(dataset):
            for idx, file_name in enumerate(dataset.image_filenames):
                cam_idx = 0  # one camera per img
                file_name = file_name.name
                cams_json = dataset.cameras.to_json(idx)  # this is the only correct way to get the corresponding camera
                cams_json["file_path"] = file_name
                cams_json["transform_matrix"] = cams_json["camera_to_world"]
                cams_json["camera_index"] += 1
                correct_c2w.append(cams_json)

        save_c2w(pipeline.datamanager.train_dataset)
        save_c2w(pipeline.datamanager.eval_dataset)
        sorted(correct_c2w, key=lambda d: d["camera_index"])
        c2w_out = self.output_path.parent / "correct_c2w.json"
        c2w_out.write_text(json.dumps({"frames": correct_c2w}, indent=2), 'utf8')
        CONSOLE.print(f"Saved correct c2w to: {c2w_out}")
        
        print(f"{pipeline.datamanager.object_obb=}")
        pipeline.datamanager.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=pipeline.datamanager.eval_dataset,
            device=pipeline.datamanager.fixed_indices_eval_dataloader.device,
            object_obb=pipeline.datamanager.object_obb,
        )
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        if self.render_all_images:
            CONSOLE.log("performing additional eval on test images")
            pipeline.datamanager.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
                input_dataset=pipeline.datamanager.train_dataset,
                device=pipeline.datamanager.fixed_indices_eval_dataloader.device,
                object_obb=pipeline.datamanager.object_obb,
            )
            additional_metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
            metrics_dict["additional_eval"] = additional_metrics_dict
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
