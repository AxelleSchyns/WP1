#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns

# Number of models to test
nb_models=2

# path to data
path_test='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
path_validation='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K-NONORM'
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K' 
#path_validation='/home/labsig/Documents/Axelle/cytomine/Data/CrC/CRC-VAL-HE-7K'

# path to weights (each model)
common_path='/home/labsig/Documents/Axelle/cytomine/cbir-tfe/weights_folder'
weights=("$common_path/test_env/test_env_og_expo" "$common_path/test_env/old2" 
        "$common_path/resnet/v0_alan/epoch_49" "$common_path/resnet/version_1/last_epoch" 
        "$common_path/resnet/version_3/last_epoch" "$common_path/resnet/version_2/epoch_49"
         "$common_path/deit/v0_alan/last_epoch" "$common_path/deit/v5_alan/last_epoch"
         "$common_path/deit/v7_alan/last_epoch" "$common_path/deit/step_1/old2"
         "$common_path/deit/step_2/weights" "$common_path/deit/step_3/old_newEnv"
         "$common_path/deit/v8_alan/last_epoch" "$common_path/deit/v9_alan/last_epoch"
         "$common_path/deit/v1_alan/epoch_49" "$common_path/deit/version_1/last_epoch" 
         "$common_path/deit/v2_alan/last_epoch" "$common_path/deit/v4_alan/last_epoch" 
         "$common_path/deit/version_0/epoch_43" "$common_path/deit/v6_alan/last_epoch"
         "$common_path/deit/vold1_alan/weights" "$common_path/deit/vold2_alan/weights"
         "$common_path/Dino/Resnet_scratch/checkpoint_res.pth" "$common_path/Dino/Resnet_pre/checkpoint.pth" 
         "$common_path/Dino/Vit_scratch/checkpoint0099_scratch.pth" "$common_path/Dino/Vit_tiny/checkpoint.pth"
         "$common_path/Dino/Vit_pretrained/checkpoint0099_pretrained.pth"
         "$common_path/Dino/Vit_pretrained/pretrained_vit_small_checkpoint.pth"
         
          )

# Extractors
extractors=('deit' 'deit' 'resnet' 'resnet' 'resnet' 'resnet' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'deit' 'dino_resnet' 'dino_resnet' 'dino_vit'  'dino_tiny' 'dino_vit' 'dino_vit' )

# Number of features
num_features=(128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 2048 2048 384 192 384 384)

# Type of measure
measures=('stat' 'all' 'weighted')

# Output files
output_file='output_crc.log'
warnings_file='warnings_crc.log'

for ((nb=1; nb<nb_models; nb++)); do
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "Weights: ${weights[nb]}" >> "$output_file"
    echo "Indexing" >> "$output_file"
    python database/add_images.py --path "$path_test" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --num_features "${num_features[nb]}" --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"

    for i in "${!measures[@]}"; do
        echo "${measures[i]}" >> "$output_file"
        python database/test_accuracy.py --num_features "${num_features[nb]}" --path "$path_validation" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[i]}" --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
    done
done