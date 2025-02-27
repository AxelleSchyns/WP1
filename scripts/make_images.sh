#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to generate qualitative results for each model
#--------------------------------------------------------------------------------------------------------
# aschyns



# path to data
path_uliege_index='/home/labsig/Documents/Axelle/cytomine/Data/our/test'
#'/home/axelle/Documents/Doctorat/WP1/data/uliege/test'
path_camelyon_index='/home/labsig/Documents/Axelle/cytomine/Data/cam17/test' 
path_crc_index='/home/labsig/Documents/Axelle/cytomine/Data/CrC/NCT-CRC-HE-100K' 

path_validation=

# path to weights (each model)
common_path='/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder' 
common_path='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder'
common_our='/home/labsig/Documents/Axelle/cytomine/Data/our/validation'
common_crc='/home/labsig/Documents/Axelle/cytomine/Data/CrC/CRC-VAL-HE-7K'
common_cam='/home/labsig/Documents/Axelle/cytomine/Data/cam17/validation'

# Loop over the specified number of iterations
for ((i=1; i<=0; i++))
do
  folder_name="results/uliege"
  mkdir "$folder_name"
  python database/retrieve_images.py \
  --path "$common_our/camelyon16_0/27665488_9984_129024_768_768.png","$common_our/ulg_lbtd2_chimio_necrose_36362044/36360773_66312823.png","$common_our/patterns_no_aug_0/8124013_18091437.png","$common_our/janowczyk6_1/8867_idx5_x501_y1051_class1.png","$common_our/mitos2014_1/A10_113799388_52_1503_323_323_0.png","$common_our/tupac_mitosis_1/11104704_1371_589_250_250_5.png" \
  --extractor "resnet","deit","dino_vit","dino_vit","dino_vit","byol_light","byol_light","byol_light","ibot_vits","ibot_vits","ibot_vits","ret_ccl","ctranspath","phikon","cdpath","uni","hoptim" \
  --num_features 128,128,384,384,384,256,256,256,384,384,384,2048,768,768,512,1024,1536 \
  --weights "$common_path/resnet/model2/last_epoch","$common_path/deit/model5/last_epoch","$common_path/Dino/model11/dino_deitsmall16_pretrain.pth","$common_path/Dino/model13/checkpoint0099_pretrained.pth","$common_path/Dino/model12/checkpoint0099_scratch.pth","$common_path/byol_light/model19/epoch=99-step=500400.ckpt","$common_path/byol_light/model18/epoch=99-step=164900.ckpt","$common_path/byol_light/model15/epoch=99-step=164900.ckpt","$common_path/ibot/model27/checkpoint_teacher.pth","$common_path/ibot/model30/checkpointv1_99.pth","$common_path/ibot/model31/checkpointV0_0085.pth","$common_path/retccl/model24/ret_pretrained.pth","$common_path/cdpath/model33/CAMELYON17.ckpt","$common_path/phikon/model26/placeholder.txt","$common_path/ctranspath/model29/ctranspath.pth","$common_path/uni/model32/uni","$common_path/hoptimus/model34/placeholder.txt" \
  --results_dir "$folder_name" \
  --path_index "$path_uliege_index"

  python database/retrieve_images.py \
  --path "$common_cam/normal/patch_patient_020_node_2_x_5600_y_16288.png","$common_cam/tumor/patch_patient_020_node_2_x_704_y_15232.png" \
  --extractor "resnet","deit","dino_vit","dino_vit","dino_vit","byol_light","byol_light","byol_light","ibot_vits","ibot_vits","ibot_vits","ret_ccl","ctranspath","phikon","cdpath","uni","hoptim" \
  --num_features 128,128,384,384,384,256,256,256,384,384,384,2048,768,768,512,1024,1536 \
  --weights "$common_path/resnet/model2/last_epoch","$common_path/deit/model5/last_epoch","$common_path/Dino/model11/dino_deitsmall16_pretrain.pth","$common_path/Dino/model13/checkpoint0099_pretrained.pth","$common_path/Dino/model12/checkpoint0099_scratch.pth","$common_path/byol_light/model19/epoch=99-step=500400.ckpt","$common_path/byol_light/model18/epoch=99-step=164900.ckpt","$common_path/byol_light/model15/epoch=99-step=164900.ckpt","$common_path/ibot/model27/checkpoint_teacher.pth","$common_path/ibot/model30/checkpointv1_99.pth","$common_path/ibot/model31/checkpointV0_0085.pth","$common_path/retccl/model24/ret_pretrained.pth","$common_path/cdpath/model33/CAMELYON17.ckpt","$common_path/phikon/model26/placeholder.txt","$common_path/ctranspath/model29/ctranspath.pth","$common_path/uni/model32/uni","$common_path/hoptimus/model34/placeholder.txt" \
  --results_dir $folder_name --path_index $path_camelyon_index

  python database/retrieve_images.py \
  --path  "$common_crc/BACK/BACK-TCGA-ADESWYYD.tif","$common_crc/LYM/LYM-TCGA-AAWGSCHH.tif","$common_crc/MUS/MUS-TCGA-AASRLCCT.tif","$common_crc/STR/STR-TCGA-AHKNISSH.tif" \
  --extractor "resnet","deit","dino_vit","dino_vit","dino_vit","byol_light","byol_light","byol_light","ibot_vits","ibot_vits","ibot_vits","ret_ccl","ctranspath","phikon","cdpath","uni","hoptim" \
  --num_features 128,128,384,384,384,256,256,256,384,384,384,2048,768,768,512,1024,1536 \
  --weights "$common_path/resnet/model2/last_epoch","$common_path/deit/model5/last_epoch","$common_path/Dino/model11/dino_deitsmall16_pretrain.pth","$common_path/Dino/model13/checkpoint0099_pretrained.pth","$common_path/Dino/model12/checkpoint0099_scratch.pth","$common_path/byol_light/model19/epoch=99-step=500400.ckpt","$common_path/byol_light/model18/epoch=99-step=164900.ckpt","$common_path/byol_light/model15/epoch=99-step=164900.ckpt","$common_path/ibot/model27/checkpoint_teacher.pth","$common_path/ibot/model30/checkpointv1_99.pth","$common_path/ibot/model31/checkpointV0_0085.pth","$common_path/retccl/model24/ret_pretrained.pth","$common_path/cdpath/model33/CAMELYON17.ckpt","$common_path/phikon/model26/placeholder.txt","$common_path/ctranspath/model29/ctranspath.pth","$common_path/uni/model32/uni","$common_path/hoptimus/model34/placeholder.txt" \
  --results_dir $folder_name --path_index $path_crc_index
done
  


for ((i=0; i<=0; i++))
do
  # Create folder with the iteration number as part of its name
  folder_name="results/cam"
  mkdir "$folder_name"
  
  python database/retrieve_images.py \
  --path "$common_cam/normal/patch_patient_020_node_2_x_5600_y_16288.png","$common_cam/tumor/patch_patient_020_node_2_x_704_y_15232.png" \
  --extractor "hoptim" \
  --num_features 1536 \
  --weights "$common_path/hoptimus/model34/placeholder.txt" \
  --results_dir $folder_name --path_index $path_camelyon_index

done


for ((i=0; i<=0; i++))
do
  # Create folder with the iteration number as part of its name
  folder_name="results/crc_$i"
  mkdir "$folder_name"
  python database/retrieve_images.py \
  --path  "$common_crc/BACK/BACK-TCGA-ADESWYYD.tif","$common_crc/LYM/LYM-TCGA-AAWGSCHH.tif","$common_crc/MUS/MUS-TCGA-AASRLCCT.tif","$common_crc/STR/STR-TCGA-AHKNISSH.tif" \
  --extractor "hoptim" \
  --num_features 1536 \
  --weights "$common_path/hoptimus/model34/placeholder.txt" \
  --results_dir $folder_name --path_index $path_crc_index

done