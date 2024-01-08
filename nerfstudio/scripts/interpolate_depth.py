from PIL import Image
import numpy as np
import argparse
import os


def main():
    inpainted_depth_dir = os.path.join(args.load_dir, 'eval_images')
    real_depth_dir = args.real_depth_dir
    nsa_path = os.path.join(args.load_dir, 'wandb/plots/aabb_intersection.npy')

    # read the nsa vertices
    nsa_vertices = np.load(nsa_path) 
    """
    [[-0.09375    -0.203125   -0.83986217]
    [-0.09375    -0.0625     -0.8418088 ]
    [ 0.0625     -0.203125   -0.8338717 ]
    [ 0.0625     -0.0625     -0.83581835]]
    """
    
    # below are copied from cs's notebook
    # TODO: adapt this to our nerfstudio codebase
    """
    # read "data/nerfstudio/polycam_mate_floor_hack/inpainted/depth_inpaint/frame00001.png" from png file as a numpy array
    # inpainted_depth_path = 'data/nerfstudio/polycam_mate_floor_hack/inpainted/depth_inpaint/frame00001.png'
    # inpainted_depth_path = 'outputs/polycam_mate_floor/nerfacto/2023-11-22_200027/eval_images/frame_00001_depth.png'
    real_depth_path = 'data/nerfstudio/polycam_mate_floor/depth/frame_00001.png'

    # inpainted_depth = np.array(Image.open(inpainted_depth_path))
    real_depth = np.array(Image.open(real_depth_path))

    # for the region 300<x<500, 250<y<400, do linear interpolation
    x_lower, x_upper = 300, 500
    y_lower, y_upper = 250, 400
    upper_border = real_depth[y_lower, x_lower:x_upper] # (200,)
    lower_border = real_depth[y_upper, x_lower:x_upper] # (200,)
    left_border = real_depth[y_lower:y_upper, x_lower].reshape([-1, 1]) # (150, 1)
    right_border = real_depth[y_lower:y_upper, x_upper].reshape([-1, 1]) # (150, 1)

    interpolated_depth = real_depth.copy()
    x_unit_linspace = np.linspace(0, 1, x_upper - x_lower) # (200,)
    y_unit_linspace = np.linspace(0, 1, y_upper - y_lower).reshape([-1, 1]) # (150,1)
    x_difference = right_border - left_border # (150, 1)
    y_difference = lower_border - upper_border # (200,)
    # linear interpolation on the x-direction
    x_inpainting = x_difference * x_unit_linspace + left_border # (150, 200)
    # linear interpolation on the y-direction
    y_inpainting = y_difference * y_unit_linspace + upper_border # (150, 200)

    interpolated_depth_x = interpolated_depth.copy()
    interpolated_depth_x[y_lower:y_upper, x_lower:x_upper] = x_inpainting
    interpolated_depth_y = interpolated_depth.copy()
    interpolated_depth_y[y_lower:y_upper, x_lower:x_upper] = y_inpainting

    interpolation_diff = interpolated_depth_x - interpolated_depth_y
    interpolated_depth_average = (interpolated_depth_x + interpolated_depth_y) / 2

    # save interpolated_depth_average as 16-bit png
    # check if the depth values are within the range of 0-65535
    print(np.min(interpolated_depth_average), np.max(interpolated_depth_average))
    assert np.min(interpolated_depth_average) >= 0 and np.max(interpolated_depth_average) <= 65535
    interpolated_depth_average = interpolated_depth_average.astype(np.uint16)
    # save the depth map as a png file
    interpolated_depth_average_path = 'interpolated_depth_average.png'
    interpolated_depth_average_image = Image.fromarray(interpolated_depth_average)
    interpolated_depth_average_image.save(interpolated_depth_average_path)
    """


if __name__ == "__main__":
    """
    required arguments:
    --load-dir: the directory of the checkpoint, e.g. "outputs/polycam_mate_floor/depth-nerfacto/2023-12-09_000432"
    --real-depth-dir: the directory of the real depth map, e.g. "data/nerfstudio/polycam_mate_floor/depth"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--real-depth-dir", type=str, required=True)
    args = parser.parse_args()
    main()