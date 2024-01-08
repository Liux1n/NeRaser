import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import argparse

def process_mask(data_dir):
    #input_path = os.path.join(data_dir, 'transforms.json')
    input_path = data_dir + '/transforms.json'
    #output_path = os.path.join(data_dir, 'modified_transforms.json')
    output_path = data_dir + '/modified_transforms.json'
    #new_mask_base_path = os.path.join(data_dir, 'masks_2')
    new_mask_base_path = data_dir + '/masks_2'

    #images_path = os.path.join(data_dir, 'images')
    images_path = data_dir + '/images'


    # # Define dilation amount
    # dilation_amount = 15
    # kernel = np.ones((dilation_amount, dilation_amount), np.uint8)

    # Resize images in mask_2 to match the resolution of images in the images directory

    
    # Update the JSON data
    with open(input_path, "r") as file:
        data = json.load(file)

    if "frames" in data:
        for frame in data["frames"]:
            file_path = frame.get("file_path", "")
            # change ".jpg" to ".png"
            filename = os.path.basename(file_path).replace(".jpg", ".png")
            # filename = os.path.basename(file_path)
            #print(filename)
            #new_mask_path = os.path.join("./masks_2", filename)
            #new_mask_path =  './masks_2/' + filename
            frame["mask_path"] = './masks_2/' + filename
            #print(frame["mask_path"])
        #change every "file_path": "./images/frame_00001.jpg" to "file_path": "./images/frame_00001.png"
        for frame in data["frames"]:
            file_path = frame.get("file_path", "")
            filename = os.path.basename(file_path).replace(".jpg", ".png")
            frame["file_path"] = './images/' + filename
            #print(frame["file_path"])
    
    # open images folder and change every file from .png if it is not .png
    for file in os.listdir(images_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            if file.lower().endswith(('.jpg')):
                os.rename(os.path.join(images_path, file), os.path.join(images_path, file.replace(".jpg", ".png")))
            elif file.lower().endswith(('.jpeg')):
                os.rename(os.path.join(images_path, file), os.path.join(images_path, file.replace(".jpeg", ".png")))

    for mask_file in os.listdir(new_mask_base_path):
        mask_file_path = os.path.join(new_mask_base_path, mask_file)
        
        if mask_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_image = Image.open(mask_file_path)
            # mask_image = Image.eval(mask_image, lambda x: 255 - x) # invert

            corresponding_image_path = os.path.join(images_path, mask_file)
            #print(f"Corresponding image path: {corresponding_image_path}")
            
            if os.path.exists(corresponding_image_path):
                #print(f"Found corresponding image: {corresponding_image_path}")
                image = Image.open(corresponding_image_path)
                
                # Check if the resolutions are different
                if mask_image.size != image.size:
                    mask_image = mask_image.resize(image.size)
                    mask_image.save(mask_file_path)
                    #print(f"Resized {mask_file} to match the resolution of {corresponding_image_path}")
                else:
                    mask_image.save(mask_file_path)
        

    with open(input_path, "w") as file:
         json.dump(data, file, indent=2)

                        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    process_mask(args.data_dir)