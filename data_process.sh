#!/bin/bash

echo "Preparing the dataset..."

# path of the dataset
# change to the path where you store the polycam raw data
path_dataset='G:\Mixed_Reality\mr_dataset\datasets'
path_segmentation='G:\Mixed_Reality\mr_dataset\MiVOS_segmentation'


cd $path_dataset 
files=$(ls)

for file in $files
do

    file_name=$(echo $file | cut -f 1 -d '.')
    # skip the file if it is a directory
    if [ -d $file_name ]
    then
        continue
    fi
    echo "Processing "$file_name
    mkdir $file_name
    output_dir=$path_dataset/$file_name
    # process the data under the path_data/dataset_name
    ns-process-data polycam --data $file --output-dir $output_dir --use_depth
    # delete the .zip file
    rm $file
    
    # execute the the path_segmentation/interactive_gui.py
    
done

echo "Converted All Datasets to NerfStudio Format!"

echo "Preparing Segmentation Masks..."


# List all directories in the path_dataset
dirs=( $(find "$path_dataset"/* -maxdepth 0 -type d) )

# Print all directories
echo "Datasets:"
for (( i=0; i<${#dirs[@]}; i++ )); do
    # Check if masks_2 subdirectory exists and is not empty
    if [ -d "${dirs[$i]}/masks_2" ] && [ "$(ls -A ${dirs[$i]}/masks_2)" ]; then
        echo "$((i+1)): $(basename ${dirs[$i]}) (masks exist)"
    else
        echo "$((i+1)): $(basename ${dirs[$i]})"
    fi
done

# Ask the user to choose a directory
echo "Enter the number of the dataset you want to generate/regenerate mask or enter 'all' to generate masks for all datasets:"
read dir_num

# Check if the user chose to process all directories
if [[ $dir_num == "all" ]]; then
    echo "You chose to process all directories."
    # Loop over all directories
    for chosen_dir in "${dirs[@]}"; do
        # if [ -d $chosen_dir/masks_2 ]
        # then
        #     echo "Masks already exist. Skipping."
        # else
        python $path_segmentation/interactive_gui.py \
        --images $chosen_dir/images \
        --output_dir $chosen_dir \
        --prop_model $path_segmentation/saves/propagation_model.pth \
        --fusion_model $path_segmentation/saves/fusion.pth \
        --s2m_model $path_segmentation/saves/s2m.pth \
        --fbrs_model $path_segmentation/saves/fbrs.pth
        # fi

        echo "Modifying transforms.json..."
        python $path_segmentation/add_mask.py --data_dir $chosen_dir

        echo "Done!"
        #process_directory $chosen_dir
    done
else
    # Subtract 1 because arrays are 0-indexed
    dir_num=$((dir_num-1))

    # Check if the chosen number is valid
    if [[ dir_num -lt 0 || dir_num -ge ${#dirs[@]} ]]; then
        echo "Invalid number. Exiting."
        exit 1
    fi

    # Get the chosen directory
    chosen_dir=${dirs[$dir_num]}
    # if [ -d $chosen_dir/masks_2 ]
    # then
    #     echo "Masks already exist. Skipping."
    # else
    python $path_segmentation/interactive_gui.py \
    --images $chosen_dir/images \
    --output_dir $chosen_dir \
    --prop_model $path_segmentation/saves/propagation_model.pth \
    --fusion_model $path_segmentation/saves/fusion.pth \
    --s2m_model $path_segmentation/saves/s2m.pth \
    --fbrs_model $path_segmentation/saves/fbrs.pth
    # fi

    echo "Modifying transforms.json..."
    python $path_segmentation/add_mask.py --data_dir $chosen_dir

    echo "Done!"
    #process_directory $chosen_dir
fi




