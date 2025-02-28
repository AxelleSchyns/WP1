from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from PIL import Image

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
    title_name_uliege = ['(a)','(b)','(c)','(d)','(e)','(f)']
    title_name_bench = ['BACK', 'LYM', 'MUS', 'STR', 'cam - normal', 'cam - tumor']
    search_strings = ['query', 'resnet', 'deit']
    labels_uliege=['False', 'False', 'True', 'True', 'False', 'False', 'True', 'True', 'True', 'True', 'True', 'False']
    labels_bench=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True']
    if args.dataset == 'uliege' or args.dataset == "both":
        inputs = []
        in_dir = os.path.join(args.input_dir, 'uliege')
        folds = os.listdir(in_dir)
        folds.sort()
        # Walk through the directory
        for search_string in search_strings:
            for fold in folds:
                if "folder" in fold:
                    for files in os.listdir(os.path.join(in_dir, fold)):
                        if search_string in files:
                            inputs.append(os.path.join(in_dir, fold, files))
        print(len(inputs))
        plt.figure()

        for i in range(1, len(inputs)+1):
                if labels_uliege[i-7] == 'True':
                    color = 'green'
                else:
                    color = 'red'
                plt.subplot(3,6,i)
                plt.axis('off')
                plt.imshow(Image.open(inputs[i-1]))
                if i < 7:
                    plt.title(title_name_uliege[i-1], fontsize=8)
                elif i < 13:
                    plt.title('ResNet', fontsize=8, color=color)
                else:
                    plt.title('DeiT', fontsize=8, color=color) 
        plt.subplots_adjust(wspace=0, hspace=0)      
        plt.savefig(os.path.join(args.input_dir, "sup_uliege.jpg"))

    if args.dataset == 'bench' or args.dataset == 'both':
        inputs = []
        in_dir_crc = os.path.join(args.input_dir, 'crc')
        folds_crc = os.listdir(in_dir_crc)
        folds_crc.sort()
        # Walk through the directory
        for search_string in search_strings:
            for fold in folds_crc:
                if  os.path.isdir(os.path.join(in_dir_crc, fold)):
                    for files in os.listdir(os.path.join(in_dir_crc, fold)):
                        if search_string in files:
                            inputs.append(os.path.join(in_dir_crc, fold, files))
        in_dir_cam = os.path.join(args.input_dir, 'Camelyon17')
        folds_cam = os.listdir(in_dir_cam)
        folds_cam.sort()
        # Walk through the directory
        for search_string in search_strings:
            for fold in folds_cam:
                if os.path.isdir(os.path.join(in_dir_cam, fold)):
                    for files in os.listdir(os.path.join(in_dir_cam, fold)):
                        if search_string in files:
                            inputs.append(os.path.join(in_dir_cam, fold, files))
        print(len(inputs))
        plt.figure()

        for i in range(1, len(inputs)+1):
                if labels_bench[i-7] == 'True':
                    color = 'green'
                else:
                    color = 'red'
                plt.subplot(3,6,i)
                plt.axis('off')
                plt.imshow(Image.open(inputs[i-1]))
                if i < 7:
                    plt.title(title_name_bench[i-1], fontsize=8)
                elif i < 13:
                    plt.title('ResNet', fontsize=8, color=color)
                else:
                    plt.title('DeiT', fontsize=8, color=color) 
        plt.subplots_adjust(wspace=0, hspace=0)      
        plt.savefig(os.path.join(args.input_dir, "sup_bench.jpg"))