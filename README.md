<p align="center">
  <h1 align="center">NeRaser: NeRF-based 3D Object Eraser</h1>
  <p align="center">
    <a href="https://github.com/Liux1n"><strong>Liuxin Qing*</strong></a>
    ·
    <a href="https://github.com/billyzs"><strong>Shao Zhou*</strong></a>
    ·
    <a href="https://github.com/XichongLing"><strong>Xichong Ling*</strong></a>
    ·
    <a href="https://github.com/cs-vision"><strong>Shi Chen*</strong></a>
  </p>
  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h3 align="center"><a href="https://github.com/MixedRealityETHZ/Let-Objects-vanish-/blob/main/documents/poster.pdf">Poster</a> | <a href="https://github.com/MixedRealityETHZ/Let-Objects-vanish-/blob/main/documents/Mixed_Reality_Report.pdf">Report</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./documents/demo.gif" alt="Logo" width="80%">
  </a>
</p>
<p align="center">
Rendering Result of NeRaser
</p>
<p align="center">
  
# About NeRaser

NeRaser: A NeRF-based object eraser

# Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.7 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name NeRaser -y python=3.8
conda activate NeRaser
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` and GCC <= 10 is required for building `tiny-cuda-nn` (GCC version > 10 is not compatible with the version of pybind used in tiny-cuda-nn).
** Installing tiny-cuda-nn is strongly recommended; not only does it speed up training, but stability is also significantly improved with tiny-cuda-nn installed.** We found that training without tiny-cuda-nn often diverge after a small number of iterations, and suspect that default hyperparameters in nerfstudio are chosen with tiny-cuda-nn in use, leading to instabilities when tiny-cuda-nn is not used.

For CUDA 11.7:

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

### Installing nerfstudio

```bash
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

### Installing Segmentation module
```bash
cd MiVOS_segmentation
python download_model.py # download all required models
```


### Installing LaMa
```bash
git clone https://github.com/advimman/lama.git lama
pip install lama/requirements.txt
mkdir -p lama/model && cd lama/model
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip | unzip -
```


## 1. Training your first model!

Modify the directory `path_dataset` in the NeRaser.sh to where you store the zip file from PolyCam.

Run the following script and follow the instructions to generate masks and start training!
```bash
./NeRaser.sh
```

If everything works, you should see training progress like the following:

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766069-cadfd34f-8833-4156-88b7-ad406d688fc0.png">
</p>

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

## 2. Inpainting and post-processing

The previous step would generate NeRF rendering of every viewpoint in the dataset, an estimate of the plane that supports the object, as well as the area on the plane that should be inpainted. Run nsa_processing.ipynb by giving it the `images_render` folder of the dataset created in the previous step, as well as the `plane_coefficients.npy` and `object_occupancy.npy` files, which contain the estimated plane and object occupancy in nerfstudio cooridnates. These two files are located in the `wandb/plots` folder in the output directory of training. The notebook will generate mask for the area to be inpainted, in the `masks_nsa` directory of the dataset directory specified in Neraser.sh. The generated mask can be visually inspected in the notebook for consistency.

Inpainting can be done using the [`predict.py`](https://github.com/advimman/lama/blob/main/bin/predict.py) script in LaMa. The script will inpaint every image in the dataset, but we will only use one of the inpainted image, typically the first frame in the dataset. The notebook can be used again to warp the inpainted area to the rest of the images.

You now have everything you need for the second round of training. Run [`ns-train`](https://github.com/advimman/lama/blob/main/bin/predict.py) by giving it the processed dataset and the `config.yaml` file generated from first round of training, modifying its parameters as needed. 
```bash
ns-train nerfacto --data <path-to-inpainted-data> --load-config <config-yaml-file>
```

## 3. Exporting Results

Once you have a NeRF model you can either render out a video or export a point cloud.

### Render Video

First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press "RENDER" which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and run the command to generate the video.

Other video export options are available, learn more by running

```bash
ns-render --help
```

## 4. App

A webapp is made to demonstrate the typical workflow of such a tool. The app guides the user through the process of data preparation, mask generation and inspection, and presents the refined NeRF scene without the object. It can be launched with

```bash
pip install streamlit streamlit_image_coordinates
streamlit run ./app.py
```

# Acknowledgement
This project was undertaken as part of the 2023HS Mixed Reality course at ETH Zurich. We would like to express our sincere appreciation for the valuable guidance and support provided by our supervisor, [Sandro Lombardi](https://github.com/sandrlom).

