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
    args = parser.parse_args()
    extractor_name = ['retccl', 'uni', 'ctranspath', 'cdpath','phikon']
    search_strings=['ret_ccl', 'uni', 'ctranspath', 'cdpath','phikon']
    #extractor_name = ['dino_pretrained', 'dino_scratch', 'dino_finetuned', 'byol_pretrained', 'byol_scratch', 'byol_finetuned', 'ibot_pretrained', 'ibot_scratch', 'ibot_finetuned']
    #search_strings=['dino_vit_2', 'dino_vit_3', "dino_vit_4","byol_light_7", "byol_light_5", "byol_light_6", "ibot_vits_9", "ibot_vits_11", "ibot_vits_10"]
    inputs = []
    # Walk through the directory
    for search_string in search_strings:
        for files in os.listdir(args.input_dir):
            if search_string in files:
                inputs.append(os.path.join(args.input_dir, files))
            if "query" in files:
                input = os.path.join(args.input_dir, files)
    print(len(inputs))
    plt.figure()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.subplot(3,4,1)
    plt.axis('off')
    plt.imshow(Image.open(input))
    plt.title("Query", fontsize=8)

    for i in range(1, len(inputs)+1):
        if i < 4:
            plt.subplot(3,4,i+1)
            plt.axis('off')
            plt.imshow(Image.open(inputs[i-1]))
            plt.title(extractor_name[i-1], fontsize=8)
        elif i<7:
            plt.subplot(3,4,i+2)
            plt.axis('off')
            plt.imshow(Image.open(inputs[i-1]))
            plt.title(extractor_name[i-1], fontsize=8)
        else:
            plt.subplot(3,4,i+3)
            plt.axis('off')
            plt.imshow(Image.open(inputs[i-1]))
            plt.title(extractor_name[i-1], fontsize=8)
    plt.subplots_adjust(wspace=0, hspace=0)      
    plt.savefig(os.path.join(args.input_dir, "ssl_med.png"))
