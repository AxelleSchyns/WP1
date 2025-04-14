from argparse import ArgumentParser
import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps
import matplotlib.patches as patches

def remove_blank_space(image, border=10):
    # Convert to grayscale and find bounding box
    gray = image.convert("L")  # Convert to grayscale
    inverted = ImageOps.invert(gray)  # Invert colors (so white areas become black)
    bbox = inverted.getbbox()  # Get bounding box of non-white areas

    if bbox:
        image = image.crop(bbox)  # Crop to content
        image = ImageOps.expand(image, border, fill="white")  # Optional: Add padding back
    return image

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
    )
    args = parser.parse_args()
    
    inputs = []
    # get the 6 images from folder
    for files in os.listdir(args.input_dir):
        inputs.append(files)
    inputs.sort() 

    img_list_med = []
    img_list_ssl = []
    for files in inputs:
        if files == "final_med.png" or files == "final_ssl.png":
            continue
        if 'med' in files:
            img = Image.open(os.path.join(args.input_dir, files))
            cropped_img = remove_blank_space(img)
            img_list_med.append(cropped_img)
        elif 'ssl' in files:
            img = Image.open(os.path.join(args.input_dir, files))
            cropped_img = remove_blank_space(img)
            img_list_ssl.append(cropped_img)
    # Med images
    fig = plt.figure(figsize=(20, 30))  # Adjust figure size as needed
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, wspace=0)  # hspace=0 removes vertical gap
    letters = ['(a)', '(b)', '(e)', '(c)', '(d)', '(f)']
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_list_med[i])  # Load images properly
        ax.axis('off')
        ax.text(0.1, 0.25, letters[i], fontsize=36, ha="center", va="center", color="black", transform=ax.transAxes)

    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(img_list_med[i+3])
        ax.axis('off')
        ax.text(0.1, 0.25, letters[i+3], fontsize=36, ha="center", va="center", color="black", transform=ax.transAxes)
    plt.savefig(os.path.join(args.input_dir,"final_med.png"), bbox_inches="tight", pad_inches=0)

    # SSL images
    fig = plt.figure(figsize=(20, 30))  # Adjust figure size as needed
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, wspace=0.1)  # hspace=0 removes vertical gap
    letters = ['(a)', '(b)', '(e))', '(c))', '(d))', '(f)']
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_list_ssl[i])  # Load images properly
        ax.axis('off')
        ax.text(0.1, 0.25, letters[i], fontsize=36, ha="center", va="center", color="black", transform=ax.transAxes)
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(img_list_ssl[i+3])
        ax.axis('off')
        ax.text(0.1, 0.25, letters[i+3], fontsize=36, ha="center", va="center", color="black", transform=ax.transAxes)  
    plt.savefig(os.path.join(args.input_dir,"final_ssl.png"), bbox_inches="tight", pad_inches=0)
    
