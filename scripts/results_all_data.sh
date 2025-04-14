#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to test the accuracy
#--------------------------------------------------------------------------------------------------------
# aschyns


# Number of models to test
nb_models=10

# path to data
uliege_test='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
uliege_validation='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'
cam_test='/home/labsig/Documents/Axelle/cytomine/Data/cam17/test' 
cam_validation='/home/labsig/Documents/Axelle/cytomine/Data/cam17/validation'
crc_test='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K' 
crc_validation='/home/labsig/Documents/Axelle/cytomine/Data/CrC/CRC-VAL-HE-7K'

path_test=("$crc_test" "$cam_test" "$uliege_test"  )
path_validation=("$crc_validation" "$cam_validation"  "$uliege_validation" )

# path to weights (each model)
common_path='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder'

weights=("$common_path/resnet/model1/last_epoch"
        "$common_path/deit/model2/last_epoch"
        "$common_path/Dino/model3/dino_deitsmall16_pretrain.pth"
        "$common_path/Dino/model4/checkpoint0099_pretrained.pth"
        "$common_path/Dino/model5/checkpoint0099_scratch.pth"
        "$common_path/byol_light/model6/epoch=99-step=500400.ckpt"
        "$common_path/byol_light/model7/epoch=99-step=164900.ckpt"
        "$common_path/byol_light/model8/epoch=99-step=164900.ckpt"
        "$common_path/ibot/model9/checkpoint_teacher.pth"
        "$common_path/ibot/model10/checkpointv1_99.pth"
        "$common_path/ibot/model11/checkpointV0_0085.pth"
        "$common_path/retccl/model12/ret_pretrained.pth"
        "$common_path/cdpath/model13/CAMELYON17.ckpt"
        "$common_path/phikon/model14/placeholder.txt"
        "$common_path/ctranspath/model15/ctranspath.pth"
        "$common_path/uni/model16/uni"
        "$common_path/hoptimus/model17/placeholder.txt"
        "$common_path/uni/model18/placeholder.txt"
        "$common_path/virchow/model19/virchow2"
        "$common_path/phikon/model20/placeholder.txt"
        "$common_path/hoptimus/model17/placeholder.txt"
          )

# Extractors
extractors=('resnet' 'deit' 'dino_vit' 'dino_vit' 'dino_vit' 'byol_light' 'byol_light' 'byol_light'  'ibot_vits' 'ibot_vits' 'ibot_vits' 'ret_ccl'  'cdpath' 'phikon' 'ctranspath'  'uni' 'hoptim' 'uni2' 'virchow2' 'phikon2' 'hoptim1')
models_name=('ResNet' 'DeiT' 'DINO pre' 'DINO fine' 'DINO scratch' 'BYOL pre' 'BYOL fine' 'BYOL scratch' 'iBOT pre' 'iBOT fine' 'iBOT scratch' 'RetCCL' 'CDPath' 'Phikon' 'CTransPath' 'Uni' 'Hoptimus' 'UNI V2' 'Virchow V2' 'Phikon v2' 'Hoptimus 1')


# Type of measure
measures=('all' )
# Output files


stat=false
current_date=$(date "+%Y-%m-%d %H:%M:%S") 

for ((nb=9; nb<nb_models; nb++)); do
    output_file="$common_path/${models_name[nb]}_results.txt"
    warnings_file="$common_path/${models_name[nb]}_warnings.txt"

    echo "-----------------------------------------------------------------------------------------------" >> "$output_file"
    echo "------------------------------------- ${models_name[nb]} --------------------------------------------------" >> "$output_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$output_file" 
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file"
    echo "------------------------------------- ${models_name[nb]} --------------------------------------------------" >> "$warnings_file"
    echo "-----------------------------------------------------------------------------------------------" >> "$warnings_file" 
    echo "Date is: ${current_date}" >> "$output_file"
    echo "Weights: ${weights[nb]}" >> "$output_file"
    for ((i=0; i<3; i++)); do
        echo "Indexing" >> "$output_file"
        echo "Dataset is: ${path_test[i]}" >> "$output_file"
        
        python database/add_images.py --path "${path_test[i]}" --extractor "${extractors[nb]}" --weights "${weights[nb]}"  --rewrite --gpu_id 0 >> "$output_file" 2>> "$warnings_file"
            if ((i > 1 )); then
                python database/test_accuracy.py --path "${path_validation[i]}" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[0]}" --gpu_id 1 >> "$output_file" 2>> "$warnings_file"
            else
                python database/test_cam_acc.py --path "${path_validation[i]}" --extractor "${extractors[nb]}" --weights "${weights[nb]}" --measure "${measures[0]}" --gpu_id 1 >> "$output_file" 2>> "$warnings_file"
            fi
        done
    done
done




