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
    extractor_name_med = ['retccl', 'uni',  'uni2', 'ctranspath', 'cdpath', 'Phikon v2', 'phikon', 'hoptimus', 'Virchow v2']
    search_strings_med=['ret_ccl', 'uni', 'ctranspath', 'cdpath','phikon', 'hoptim', 'virchow2']
    # Labels for ssl meds - in order of folders
    label0_med=['False','False','False', 'False','False','False','False','False','False']
    label1_med=['True','True','False','True','Wrong','False','True','False','False']
    label2_med=['True','True','True','True','True','True','True','True','True']
    label3_med=['True','True','True','True','False','True','False','True','True']
    label4_med=['True','True','False','True','True','True','True',  'True',  'True']
    label5_med=['True','False', 'False', 'False', 'False', 'False', 'False',  'True','False']
    labels_med = [label0_med, label1_med, label2_med, label3_med, label4_med, label5_med]

    extractor_name = ['dino_pretrained', 'dino_scratch', 'dino_finetuned', 'byol_pretrained', 'byol_scratch', 'byol_finetuned', 'ibot_pretrained', 'ibot_scratch', 'ibot_finetuned']
    search_strings=['dino_vit_2', 'dino_vit_3', "dino_vit_4","byol_light_7", "byol_light_5", "byol_light_6", "ibot_vits_9", "ibot_vits_11", "ibot_vits_10"]
    # Labels for ssl nat - in order of folders
    label0=['False', 'False', 'False', 'Wrong', 'True', 'False', 'False', 'False', 'False']
    label1=['Wrong', 'True', 'True', 'Wrong', 'False', 'False', 'False', 'False', 'False']
    label2=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True']
    label3=['True', 'True', 'True', 'Wrong', 'True', 'True', 'True', 'True', 'True']
    label4=['True', 'True', 'True', 'True', 'True', 'False', 'True', 'True', 'True']
    label5=['True', 'False', 'False', 'True', 'True', 'False', 'True', 'True', 'False']
    labels = [label0, label1, label2, label3, label4, label5]
    if args.dataset == 'med' or args.dataset == "both":
        for j in range(6):
            dir_fold = os.path.join(args.input_dir, "folder_"+str(j))
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
            print(inputs)
            plt.figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.subplot(3,4,1)
            plt.axis('off')
            plt.imshow(Image.open(input))
            plt.title("Query", fontsize=12)

            for i in range(1, len(inputs)+1):
                if label_med[i-1]=='True':
                    color = 'green'
                else:
                    color = 'red'
                if i < 4:
                    plt.subplot(3,4,i+1)
                    plt.axis('off')
                    plt.imshow(Image.open(inputs[i-1]))
                    plt.title(extractor_name_med[i-1], fontsize=12, color=color)
                elif i<7:
                    plt.subplot(3,4,i+2)
                    plt.axis('off')
                    plt.imshow(Image.open(inputs[i-1]))
                    plt.title(extractor_name_med[i-1], fontsize=12, color=color)
                else:
                    plt.subplot(3,4,i+3)
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
            plt.savefig(os.path.join(args.input_dir, "ssl_full",  "ssl_med_"+str(j)+".png"))
    if args.dataset == 'nat' or args.dataset == "both":
        for j in range(6):
            dir_fold = os.path.join(args.input_dir, "folder_"+str(j))
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
            plt.savefig(os.path.join(args.input_dir, "ssl_full",  "ssl_nat_"+str(j)+".png"))
