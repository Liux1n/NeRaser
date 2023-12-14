import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import argparse


def process_mask(data_dir):
    # Paths
    # data_dir G:\Mixed_Reality\mr_dataset\dataset_test/polycam_firstAidkit

    input_path = os.path.join(data_dir, 'transforms.json')
    output_path = os.path.join(data_dir, 'modified_transforms.json')
    new_mask_base_path = os.path.join(data_dir, 'masks_2')
    images_path = os.path.join(data_dir, 'images')

    # Update the JSON data
    with open(input_path, "r") as file:
        data = json.load(file)

    if "frames" in data:
        for frame in data["frames"]:
            file_path = frame.get("file_path", "")
            filename = os.path.basename(file_path)
            image_file_name, _ = os.path.splitext(filename)

            # Look for a file with the same name in the new_mask_base_path directory, regardless of its extension
            new_mask_path = None
            for file in os.listdir(new_mask_base_path):
                if file.startswith(image_file_name):
                    new_mask_path = os.path.join(new_mask_base_path, file)
                    break

            if new_mask_path is not None:
                frame["mask_path"] = new_mask_path

    with open(input_path, "w") as file:
        json.dump(data, file, indent=2)

    print(f"Modified data saved to: {input_path}")

    # Define dilation amount
    dilation_amount = 15
    kernel = np.ones((dilation_amount, dilation_amount), np.uint8)

    # Resize images in mask_2 to match the resolution of images in the images directory
    for mask_file in os.listdir(new_mask_base_path):
        mask_file_path = os.path.join(new_mask_base_path, mask_file)
        
        if mask_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_image = Image.open(mask_file_path)
            
            # Get the corresponding image file name by removing the extension of the mask file
            image_file_name, _ = os.path.splitext(mask_file)
            
            # Look for a file with the same name in the images directory, regardless of its extension
            corresponding_image_path = None
            for file in os.listdir(images_path):
                if file.startswith(image_file_name):
                    corresponding_image_path = os.path.join(images_path, file)
                    break
            
            if corresponding_image_path is not None:
                image = Image.open(corresponding_image_path)
                
                # Check if the resolutions are different
                if mask_image.size != image.size:
                    mask_image = mask_image.resize(image.size)
                    mask_image.save(mask_file_path)
                    #print(f"Resized {mask_file} does not match the resolution of {corresponding_image_path}")
                else:
                    mask_image.save(mask_file_path)
                    #print(f"Resolution of {mask_file} already matches {corresponding_image_path}")
            
            
            
                



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    process_mask(args.data_dir)