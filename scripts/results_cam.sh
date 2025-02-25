#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns

# Number of models to test
nb_models=17

# path to data
#path_test='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/test'
#path_validation='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation'
path_test='/home/labsig/Documents/Axelle/cytomine/Data/cam17/test' 
path_validation='/home/labsig/Documents/Axelle/cytomine/Data/cam17/validation'

# path to weights (each model)
#common_path='/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder' 
# path to weights (each model)
common_path='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder'
weights=("$common_path/resnet/model2/last_epoch"
        "$common_path/deit/model5/last_epoch"
        "$common_path/Dino/model11/dino_deitsmall16_pretrain.pth"
        "$common_path/Dino/model13/checkpoint0099_pretrained.pth"
        "$common_path/Dino/model12/checkpoint0099_scratch.pth"
        "$common_path/byol_light/model19/epoch=99-step=500400.ckpt"
        "$common_path/byol_light/model18/epoch=99-step=164900.ckpt"
        "$common_path/byol_light/model15/epoch=99-step=164900.ckpt"
        "$common_path/ibot/model27/checkpoint_teacher.pth"
        "$common_path/ibot/model30/checkpointv1_99.pth"
        "$common_path/ibot/model31/checkpointV0_0085.pth"
        "$common_path/retccl/model24/ret_pretrained.pth"
        "$common_path/cdpath/model33/CAMELYON17.ckpt"
        "$common_path/phikon/model26/placeholder.txt"
        "$common_path/ctranspath/model29/ctranspath.pth"
        "$common_path/uni/model32/placeholder.txt"
        "$common_path/hoptimus/model34/placeholder.txt"
          )

# Extractors
extractors=('resnet' 'deit' 'dino_vit' 'dino_vit' 'dino_vit' 'byol_light' 'byol_light' 'byol_light'  'ibot_vits' 'ibot_vits' 'ibot_vits' 'ret_ccl'  'cdpath' 'phikon' 'ctranspath'  'uni' 'hoptim')
models_name=('ResNet' 'DeiT' 'DINO pre' 'DINO fine' 'DINO scratch' 'BYOL pre' 'BYOL fine' 'BYOL scratch' 'iBOT pre' 'iBOT fine' 'iBOT scratch' 'RetCCL' 'CDPath' 'Phikon' 'CTransPath' 'Uni' 'Hoptimus')

# Type of measure
measures=('all')
# Output files
output_file='cam_24_02.log'
warnings_file='warnings_cam_24_02.log'

for ((nb=15; nb<nb_models; nb++)); do
    
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- ${models_name[nb]}  --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file" 
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file"
    echo "------------------------------------- ${models_name[nb]} --------------------------------------------------" >> "$warnings_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file" 
    echo "Weights: ${weights[nb]}" >> "$output_file"
    echo "Indexing" >> "$output_file"
    python database/add_images.py --path "$path_test" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"

    for i in "${!measures[@]}"; do
        echo "${measures[i]}" >> "$output_file" 
        echo "${measures[i]}" >> "$warnings_file"
        python database/test_cam_acc.py  --path "$path_validation" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[i]}" --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
    done
done
