#!/bin/bash

# modify the path_dataset to your own dataset path
path_dataset='G:\Mixed_Reality\mr_dataset\demo_dataset'



# number of iterations
max_num_iterations=200
lama_refine=0

# get the path of the script
path_script=$(dirname $(realpath $0))
# path_dataset='G:\Mixed_Reality\mr_dataset\datasets'

# path_segmentation='G:\Mixed_Reality\nerfstudio\MiVOS_segmentation'\
path_segmentation=$path_script/'MiVOS_segmentation'\


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
    
    

    python $path_segmentation/interactive_gui.py \
    --images $chosen_dir/images \
    --output_dir $chosen_dir \
    --prop_model $path_segmentation/saves/propagation_model.pth \
    --fusion_model $path_segmentation/saves/fusion.pth \
    --s2m_model $path_segmentation/saves/s2m.pth \
    --fbrs_model $path_segmentation/saves/fbrs.pth
    

    echo "Modifying transforms.json..."
    python $path_segmentation/add_mask.py --data_dir $chosen_dir

    echo "Done!"
    #process_directory $chosen_dir
fi


path_script=$(dirname $(realpath $0))
#echo $path_script # /g/Mixed_Reality/mr_liuxin/nerfstudio

nerf_dir=$path_script/nerfstudio/scripts
# mr_liuxin\outputs

output_dir=$path_script/outputs
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

echo 'Saving outputs to '$output_dir
echo $nerf_dir 

dirs=( $(find "$path_dataset"/* -maxdepth 0 -type d) )

# Print all directories
echo "Training Datasets..."
for (( i=0; i<${#dirs[@]}; i++ )); do
    # Check if masks_2 subdirectory exists and is not empty
    if [ -d "${dirs[$i]}/masks_2" ] && [ "$(ls -A ${dirs[$i]}/masks_2)" ]; then
        echo "$((i+1)): $(basename ${dirs[$i]}) (masks exist)"
    else
        echo "$((i+1)): $(basename ${dirs[$i]})"
    fi
done

# Ask the user to choose a directory
echo "Enter the number of the dataset you want to train NeRF model:"
read dir_num


# Subtract 1 because arrays are 0-indexed
dir_num=$((dir_num-1))

# Check if the chosen number is valid
if [[ dir_num -lt 0 || dir_num -ge ${#dirs[@]} ]]; then
    echo "Invalid number. Exiting."
    exit 1
fi

# check if masks_2 subdirectory exists and is not empty
if [ -d "${dirs[$dir_num]}/masks_2" ] && [ "$(ls -A ${dirs[$dir_num]}/masks_2)" ]; then
   
    echo "Masks ready. Start training..."
    # 1st round training

    ns-train depth-nerfacto \
            --data ${dirs[$dir_num]} \
            --output-dir $output_dir \
            --vis wandb \
            --max-num-iterations $max_num_iterations \
            --pipeline.datamanager.flag-training-round 1

    # G:\Mixed_Reality\mr_liuxin\nerfstudio\outputs\polycam_christmasTree\depth-nerfacto\
    ckpt_dir_round_1=$output_dir/$(basename ${dirs[$dir_num]})/depth-nerfacto
    ckpt_dir_round_1=$ckpt_dir_round_1/$(ls -t $ckpt_dir_round_1 | head -1)
    
    # get .ckpt file under ckpt_dir/nerstudio_models
    ckpt_file_round_1=$(ls $ckpt_dir_round_1/nerfstudio_models/*.ckpt | head -n 1)

    echo 'Checkpoint: '$ckpt_file_round_1
    echo 'Start calculating plane...'

    #run get_plane.py

    python $nerf_dir/get_plane.py depth-nerfacto \
            --data ${dirs[$dir_num]} \
            --load_dir $ckpt_dir_round_1/nerfstudio_models

    # run ns-eval

    eval_dir=${dirs[$dir_num]}/eval_images
    eval_dir=${eval_dir//\\//}
    if [ ! -d $eval_dir ]; then
        mkdir -p $eval_dir
    fi
    # source ~/nerfstudio/.local/venv/ns/bin/activate && export PYTHONPATH=~/nerfstudio && ns-eval
    # --load-config=$1 --output-path=./nseval.json --render-output-path=$2 --render-all-images
    
    
    ns-eval --load-config=$ckpt_dir_round_1/config.yml \
            --output-path=$ckpt_dir_round_1/nseval.json \
            --render-output-path=$eval_dir\
            --render-all-images

    
    # run lama
    # echo 'Start inpainting NSA...'
    # lama_dir=$path_script/lama
    # echo $lama_dir
    # lama_model_dir=$lama_dir/big-lama
    # lama_in_dir=$eval_dir
    # lama_out_dir="$eval_dir/images_inpainted/"
    # if [ ! -d $lama_out_dir ]; then
    #     mkdir -p $lama_out_dir
    # fi
    # echo "Saving inpainted images to "$lama_out_dir
    # export PYTHONPATH=$lama_dir
    # python $lama_dir/bin/predict.py model.path=$lama_model_dir indir=$lama_in_dir outdir=$lama_out_dir refine=$lama_refine

    
    # 2nd round training
    echo 'Start 2nd round training...'






else
    echo "Masks do not exist. Please generate masks first."
    echo 'Run ./data_process.sh to generate masks.'
    # run data_process.sh
    

fi  


#process_directory $chosen_dir





