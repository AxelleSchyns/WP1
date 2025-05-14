from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--dataset",
    )
    args = parser.parse_args()
    extractor_name_med = ['retccl', 'uni', 'uni v2', 'ctranspath', 'cdpath', 'Phikon v2', 'phikon', 'hoptimus v1', 'hoptimus', 'Virchow v2']
    search_strings_med=['ret_ccl', 'uni', 'ctranspath', 'cdpath','phikon','hoptim', 'virchow2' ]
    # Labels for ssl meds - in order of folders (Back, LYM, cam-normal, MUS, STR, cam-tumor)
    label0_med=['True',  'True',  'True',  'True',  'True',  'True',  'True',  'True', 'True',  'True']
    label1_med=['False', 'False', 'True',  'False', 'False', 'True',  'True',  'True', 'True',  'True']
    label2_med=['False', 'False', 'False', 'True',  'False', 'False', 'False', 'True', 'False', 'False']
    label3_med=['True',  'True',  'True',  'True',  'True',  'True',  'True',  'True', 'True',  'True']
    label4_med=['True',  'True',  'True',  'True',  'True',  'True',  'True',  'True', 'True',  'True']
    label5_med=['True',  'True',  'True',  'True',  'True',  'True',  'True',  'True',  'True', 'True',]
    labels_med = [label0_med, label1_med, label2_med, label3_med, label4_med, label5_med]

    extractor_name = ['dino_pretrained', 'dino_scratch', 'dino_finetuned', 'byol_pretrained', 'byol_scratch', 'byol_finetuned', 'ibot_pretrained', 'ibot_scratch', 'ibot_finetuned']
    search_strings=['dino_vit_2', 'dino_vit_3', "dino_vit_4","byol_light_7", "byol_light_5", "byol_light_6", "ibot_vits_8", "ibot_vits_9", "ibot_vits_10"]
    # Labels for ssl nat - in order of folders
    label0=['True','True','True','True','True','True','True','True','True']
    label1=['True', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'False']
    label2=['True', 'False', 'True', 'True', 'True', 'True', 'True', 'True', 'True']
    label3=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True']
    label4=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True']
    label5=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'False']
    labels = [label0, label1, label2, label3, label4, label5]

    crc_dir = os.path.join(args.input_dir, "crc")
    cam_dir = os.path.join(args.input_dir, "Camelyon17")
    directories = [os.path.join(crc_dir, 'BACK'), os.path.join(crc_dir, 'LYM'), os.path.join(cam_dir, "cam_0"), os.path.join(crc_dir, 'MUS'), os.path.join(crc_dir, 'STR'), os.path.join(cam_dir, "cam_1") ]
    if args.dataset == 'med' or args.dataset == "both":
        for j in range(6):
            dir_fold = directories[j]
            label_med = labels_med[j]
            # Walk through the directory
            inputs = []
            for search_string in search_strings_med:
                for files in os.listdir(dir_fold):
                    if search_string in files:
                        inputs.append(os.path.join(dir_fold, files))
                    if "query" in files:
                        input = os.path.join(dir_fold, files)
            print(len(inputs))
            plt.figure(figsize=(8,7))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.subplot(4,4,1)
            plt.axis('off')
            plt.imshow(Image.open(input))
            plt.title("Query", fontsize=12)

            for i in range(1, len(inputs)+1):
                if label_med[i-1]=='True':
                    color = 'green'
                else:
                    color = 'red'
                if i < 4:
                    plt.subplot(4,4,i+1)
                elif i<7:
                    plt.subplot(4,4,i+2)
                elif i<10:
                    plt.subplot(4,4,i+3)
                else:
                    plt.subplot(4,4, i+4)
                plt.axis('off')
                plt.imshow(Image.open(inputs[i-1]))
                plt.title(extractor_name_med[i-1], fontsize=12, color=color)
                if label_med[i-1]=='Wrong':
                    im = Image.open(inputs[i-1])
                    width, height = im.size
                    rect = patches.Rectangle((0, 0),  width, height,  linewidth=3, edgecolor='red', facecolor='none' )
                    plt.gca().add_patch(rect)
            plt.subplots_adjust(wspace=0, hspace=0)      
            plt.savefig(os.path.join(dir_fold, "ssl_med_"+str(j)+".png"))
            plt.savefig(os.path.join(args.input_dir, "ssl_med_bench_"+str(j)+".png"))
    
    if args.dataset == 'nat' or args.dataset == "both":
        for j in range(6):
            dir_fold = directories[j]
            label = labels[j]
            # Walk through the directory
            inputs = []
            for search_string in search_strings:
                for files in os.listdir(dir_fold):
                    if search_string in files:
                        inputs.append(os.path.join(dir_fold, files))
                    if "query" in files:
                        input = os.path.join(dir_fold, files)
            print(len(inputs))
            plt.figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.subplot(3,4,1)
            plt.axis('off')
            plt.imshow(Image.open(input))
            plt.title("Query", fontsize=12)

            for i in range(1, len(inputs)+1):
                if label[i-1]=='True':
                    color = 'green'
                else:
                    color = 'red'
                if i < 4:
                    plt.subplot(3,4,i+1)
                    plt.axis('off')
                    plt.imshow(Image.open(inputs[i-1]))
                    plt.title(extractor_name[i-1], fontsize=12, color=color)
                elif i<7:
                    plt.subplot(3,4,i+2)
                    plt.axis('off')
                    plt.imshow(Image.open(inputs[i-1]))
                    plt.title(extractor_name[i-1], fontsize=12, color=color)
                else:
                    plt.subplot(3,4,i+3)
                    plt.axis('off')
                    plt.imshow(Image.open(inputs[i-1]))
                    plt.title(extractor_name[i-1], fontsize=12, color=color)
                if label[i-1]=='Wrong':
                    im = Image.open(inputs[i-1])
                    width, height = im.size
                    rect = patches.Rectangle((0, 0),  width, height,  linewidth=3, edgecolor='red', facecolor='none' )
                    plt.gca().add_patch(rect)
            plt.subplots_adjust(wspace=0, hspace=0)      
            plt.savefig(os.path.join(dir_fold, "ssl_nat_"+str(j)+".png"))
            plt.savefig(os.path.join(args.input_dir, "ssl_nat_bench_"+str(j)+".png"))
