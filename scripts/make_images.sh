#!/usr/bin/bash
#--------------------------------------------------------------------------------------------------------
#                                    Script to generate qualitative results for each model
#--------------------------------------------------------------------------------------------------------
# aschyns



# path to data
path_uliege_index='/home/axelle/Documents/Doctorat/WP1/data/uliege/test'
path_camelyon_index='/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/test'
path_crc_index='/home/axelle/Documents/Doctorat/WP1/data/CrC/NCT-CRC-HE-100K'

# path to weights (each model)
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
        "$common_path/phikon/model26/placeholder.txt"
        "$common_path/cdpath/model33/CAMELYON17.ckpt"
          )

# Extractors
extractors=('resnet' 'deit' 'dino_vit' 'dino_vit' 'dino_vit' 'byol_light' 'byol_light' 'byol_light' 'ret_ccl' 'ibot_vits' 'ibot_vits' 'ibot_vits' 'ctranspath' 'phikon'  'cdpath' )

# Number of features
num_features=(128 128 384 384 384 256 256 256 2048 384 384 384 768 768 512 )

# Can be done query by query by changing 'path' arg to $queries_uliege[?]
queries_uliege=('/home/axelle/Documents/Doctorat/WP1/data/uliege/val/camelyon16_0/27665488_9984_129024_768_768.png' 
                '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/ulg_lbtd2_chimio_necrose_36362044/36360773_66312823.png'
                '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/patterns_no_aug_0/8124013_18091437.png' 
                '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/janowczyk6_1/8867_idx5_x501_y1051_class1.png'
                '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/mitos2014_1/A10_113799388_52_1503_323_323_0.png' 
                '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/tupac_mitosis_1/11104704_1371_589_250_250_5.png')

# Loop over the specified number of iterations
for ((i=1; i<=0; i++))
do
  # Create folder with the iteration number as part of its name
  folder_name="results/folder_$i"
  mkdir "$folder_name"
  
  python database/retrieve_images.py 
  \ --path '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/camelyon16_0/27665488_9984_129024_768_768.png','/home/axelle/Documents/Doctorat/WP1/data/uliege/val/ulg_lbtd2_chimio_necrose_36362044/36360773_66312823.png','/home/axelle/Documents/Doctorat/WP1/data/uliege/val/patterns_no_aug_0/8124013_18091437.png','/home/axelle/Documents/Doctorat/WP1/data/uliege/val/janowczyk6_1/8867_idx5_x501_y1051_class1.png','/home/axelle/Documents/Doctorat/WP1/data/uliege/val/mitos2014_1/A10_113799388_52_1503_323_323_0.png','/home/axelle/Documents/Doctorat/WP1/data/uliege/val/tupac_mitosis_1/11104704_1371_589_250_250_5.png'
  \ --extractor 'resnet','deit','dino_vit','dino_vit','dino_vit','byol_light','byol_light','byol_light','ret_ccl','ibot_vits','ibot_vits','ibot_vits','ctranspath','phikon','cdpath' 
  \ --num_features 128,128,384,384,384,256,256,256,2048,384,384,384,768,768,512 
  \ --weights '/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/resnet/model2/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/deit/model5/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model11/pretrained_vit_small_checkpoint.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model12/checkpoint0099_scratch.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model13/checkpoint0099_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model15/epoch=67-step=112132.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model18/epoch=45-step=75854.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model19/epoch=99-step=500400.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/retccl/model24/ret_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model27/checkpoint_teacher.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model30/checkpoint_new.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model31/checkpoint24.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ctranspath/model29/ctranspath.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/phikon/placeholder.txt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/cdpath/model33/CAMELYON17.ckpt' 
  \--results_dir $folder_name --path_index $path_uliege_index
done



queries_cam=('/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/normal/patch_patient_020_node_2_x_5600_y_16288.png'
            '/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/tumor/patch_patient_020_node_2_x_704_y_15232.png')
for ((i=1; i<=0; i++))
do
  # Create folder with the iteration number as part of its name
  folder_name="results/cam_$i"
  mkdir "$folder_name"
  
  python database/retrieve_images.py 
  \ --path '/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/normal/patch_patient_020_node_2_x_5600_y_16288.png','/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/tumor/patch_patient_020_node_2_x_704_y_15232.png'
  \ --extractor 'resnet','deit','dino_vit','dino_vit','dino_vit','byol_light','byol_light','byol_light','ret_ccl','ibot_vits','ibot_vits','ibot_vits','ctranspath','phikon','cdpath' 
  \ --num_features 128,128,384,384,384,256,256,256,2048,384,384,384,768,768,512 
  \ --weights '/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/resnet/model2/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/deit/model5/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model11/pretrained_vit_small_checkpoint.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model12/checkpoint0099_scratch.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model13/checkpoint0099_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model15/epoch=67-step=112132.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model18/epoch=45-step=75854.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model19/epoch=99-step=500400.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/retccl/model24/ret_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model27/checkpoint_teacher.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model30/checkpoint_new.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model31/checkpoint24.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ctranspath/model29/ctranspath.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/phikon/placeholder.txt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/cdpath/model33/CAMELYON17.ckpt' 
  \ --results_dir $folder_name --path_index $path_camelyon_index

done



queries_crc=('/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/BACK/BACK-TCGA-ADESWYYD.tif'
            '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/LYM/LYM-TCGA-AAWGSCHH.tif' 
            '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/MUS/MUS-TCGA-AASRLCCT.tif'
            '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/STR/STR-TCGA-AHKNISSH.tif' )
for ((i=0; i<=0; i++))
do
  # Create folder with the iteration number as part of its name
  folder_name="results/crc_$i"
  mkdir "$folder_name"
  python database/retrieve_images.py --path  '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/BACK/BACK-TCGA-ADESWYYD.tif','/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/LYM/LYM-TCGA-AAWGSCHH.tif','/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/MUS/MUS-TCGA-AASRLCCT.tif','/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/STR/STR-TCGA-AHKNISSH.tif' --extractor 'resnet','deit','dino_vit','dino_vit','dino_vit','byol_light','byol_light','byol_light','ret_ccl','ibot_vits','ibot_vits','ibot_vits','ctranspath','phikon','cdpath' --num_features 128,128,384,384,384,256,256,256,2048,384,384,384,768,768,512 --weights '/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/resnet/model2/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/deit/model5/last_epoch','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model11/pretrained_vit_small_checkpoint.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model12/checkpoint0099_scratch.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/Dino/model13/checkpoint0099_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model15/epoch=67-step=112132.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model18/epoch=45-step=75854.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/byol_light/model19/epoch=99-step=500400.ckpt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/retccl/model24/ret_pretrained.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model27/checkpoint_teacher.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model30/checkpoint_new.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ibot/model31/checkpoint24.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/ctranspath/model29/ctranspath.pth','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/phikon/placeholder.txt','/home/axelle/Documents/Doctorat/WP1/WP1/weights_folder/cdpath/model33/CAMELYON17.ckpt' --results_dir $folder_name --path_index $path_crc_index

done