#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns


# Number of models to test
nb_models=6

# path to data
path_train='/home/labsig/Documents/Axelle/cytomine/Data/our/train'
path_test='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
path_validation='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'

# path to weights (each model)
common_path='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder'

weights=("$common_path/resnet/model2/last_epoch"
        "$common_path/deit/model5/last_epoch"
        "$common_path/Dino/model13/checkpoint0099_pretrained.pth"
        "$common_path/ctranspath/model29/ctranspath.pth"
        "$common_path/phikon/model26/placeholder.txt"
        "$common_path/uni/model32/placeholder.txt"
          )

# Extractors
extractors=('resnet' 'deit' 'dino_vit'   'ctranspath' 'phikon' 'uni')

#dbs
dbs=('resnet' 'deit' 'dino_fin'  'ctranspath' 'phikon' 'uni')

# Number of features
num_features=(128 128 384 768 768 1024)

# Type of measure
measures=('all')
# Output files
output_file='uliege_testSoft.log'
warnings_file='warnings_testsoft.log'

stat=false

for ((nb=0; nb<nb_models; nb++)); do
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file" 
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file"
    echo "------------------------------------- Model $((nb+1)) --------------------------------------------------" >> "$warnings_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file" 
    echo "Weights: ${weights[nb]}" >> "$output_file"
    echo "Indexing" >> "$output_file"
    python database/add_images.py --path "$path_train" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --db_name  "${dbs[nb]}" --num_features "${num_features[nb]}" --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
    python database/add_images.py --path "$path_test" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --db_name "${dbs[nb]}" --num_features "${num_features[nb]}"  --gpu_id 0 >> "$output_file" 2>> "$warnings_file"

    
    for i in "${!measures[@]}"; do
        echo "${measures[i]}" >> "$output_file" 
        echo "${measures[i]}" >> "$warnings_file"
        python database/test_accuracy.py --num_features "${num_features[nb]}" --path "$path_validation" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --db_name "${dbs[nb]}" --measure "${measures[i]}" --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
        
    done
done




