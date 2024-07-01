#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns

# Number of models to test
nb_models=14

# path to data
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
#path_validation='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K-NONORM'
#path_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K' 
#path_validation='/home/labsig/Documents/Axelle/cytomine/Data/CrC/CRC-VAL-HE-7K'
path_test='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/test'
path_validation='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation'

# path to weights (each model)
#common_path='/home/labsig/Documents/Axelle/cytomine/cbir-tfe/weights_folder'
common_path='/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder' 
weights=("$common_path/resnet/model2/last_epoch"
        "$common_path/deit/model5/last_epoch"
        "$common_path/Dino/model11/pretrained_vit_small_checkpoint.pth"
        "$common_path/Dino/model12/checkpoint0099_scratch.pth"
        "$common_path/Dino/model13/checkpoint0099_pretrained.pth"
        "$common_path/byol_light/model15/epoch=67-step=112132.ckpt"
        "$common_path/byol_light/model18/epoch=45-step=75854.ckpt"
        "$common_path/byol_light/model19/epoch=99-step=500400.ckpt"
        "$common_path/retccl/model24/ret_pretrained.pth"
        "$common_path/ibot/model27/checkpoint_teacher.pth"
        "$common_path/ibot/model30/checkpoint_new.pth"
        "$common_path/ibot/model31/checkpoint24.pth"
        "$common_path/ctranspath/model29/ctranspath.pth"
        "$common_path/uni/model32/placeholder.txt"
        #"$common_path/cdpath/model33/CAMELYON17.ckpt"
        #"$common_path/cdpath/model25/0190.ckpt"
        #"$common_path/phikon/model26/placeholder.txt"
          )

# Extractors
extractors=('resnet' 'deit' 'dino_vit' 'dino_vit' 'dino_vit' 'byol_light' 'byol_light' 'byol_light' 'retccl' 'ibot_vits' 'ibot_vits' 'ibot_vits' 'ctranspath' 'uni' 'cdpath' 'cdpath' 'phikon')

# Number of features
num_features=(128 128 384 384 384 256 256 256 768 384 384 384 768 512 512 512 2048)

# Type of measure
#measures=('stat' 'all' 'weighted')
measures=('all')
# Output files
output_file='cam17_2.log'
warnings_file='warnings_cam17_2.log'

for ((nb=2; nb<nb_models; nb++)); do
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file" 
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$warnings_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file" 
    echo "Weights: ${weights[nb]}" >> "$output_file"
    echo "Indexing" >> "$output_file"
    python database/add_images.py --path "$path_test" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --num_features "${num_features[nb]}" --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"

    for i in "${!measures[@]}"; do
        echo "${measures[i]}" >> "$output_file" 
        echo "${measures[i]}" >> "$warnings_file"
        python database/test_cam_acc.py --num_features "${num_features[nb]}" --path "$path_validation" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[i]}" --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
    done
done
