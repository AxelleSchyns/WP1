#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns

# Number of models to test
nb_models=16

# path to data
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
#path_validation='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K-NONORM'
path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K' 
path_validation='/home/labsig/Documents/Axelle/cytomine/Data/CrC/CRC-VAL-HE-7K'
#path_test='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/test'
#path_validation='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation'

# path to weights (each model)
common_path='/home/labsig/Documents/Axelle/cytomine/cbir-tfe/weights_folder'
weights=("$common_path/resnet/v9_alan/last_epoch" "$common_path/deit/version_18/last_epoch"
         "$common_path/Dino/Vit_pretrained/pretrained_vit_small_checkpoint.pth" "$common_path/Dino/Vit_pretrained/checkpoint0099_pretrained.pth"
         "$common_path/Dino/Vit_scratch/checkpoint0099_scratch.pth" 
          "$common_path/byol_light/v2_alan/epoch=67-step=112132.ckpt" "$common_path/byol_light/v1_alan/epoch=45-step=75854.ckpt"
        "$common_path/ret/ret_pretrained.pth" "$common_path/ibot/scratch_vit_small/checkpoint24.pth"  
        "$common_path/ctranspath/ctranspath.pth" "$common_path/ibot/finetuned/checkpoint_new.pth"
        "$common_path/ibot/vit_small_backbone/checkpoint_teacher.pth" "$common_path/ibot/scratch_vit_small/checkpoint24.pth" 
        "$common_path/byol_light/epoch=99-step=500400.ckpt"
        "$common_path/cdpath/CAMELYON17.ckpt" "$common_path/cdpath/0190.ckpt")

# Extractors
extractors=('resnet' 'deit' 'dino_vit' 'dino_vit' 'dino_vit' 
'byol_light' 'byol_light' 
 'ret_ccl' 'phikon' 'ctranspath' 'ibot_vits' 'ibot_vits'
 'ibot_vits' 'byol_light' 'cdpath' 'uni' )

# Number of features
num_features=(128 128 384 384 384 256 256 2048 768 768 384 384 384 256 512 1024)

# Type of measure
#measures=('stat' 'all' 'weighted')
measures=('all')
# Output files
output_file='crc_uni.log'
warnings_file='warnings_uni_crc.log'

for ((nb=15; nb<nb_models; nb++)); do
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file" 
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$warnings_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file" 
    echo "Weights: ${weights[nb]}" >> "$output_file"
    echo "Indexing" >> "$output_file"
    #python database/add_images.py --path "$path_test" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --num_features "${num_features[nb]}" --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"

    for i in "${!measures[@]}"; do
        echo "${measures[i]}" >> "$output_file" 
        echo "${measures[i]}" >> "$warnings_file"
        python database/test_cam_acc.py --num_features "${num_features[nb]}" --path "$path_validation" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[i]}" --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
    done
done
