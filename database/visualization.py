

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import utils
import random

# Makes a figure of samples of each class
def summary_image(path):
    classes = os.listdir(path)
    classes.sort()
    plt.figure(figsize=(7, 10))
    for class_ in classes:
        # Get images
        images = os.listdir(os.path.join(path, class_))
        idx = np.random.randint(0, len(images))
        img = Image.open(os.path.join(path, class_, images[idx])).convert('RGB')

        # save image in subplot
        plt.subplot(10, 7, classes.index(class_)+1)

        # Adjust position and label
        # For resized images: 
        img = transforms.RandomResizedCrop(224, scale=(.7,1))(img)
        # Remove scale and ticks
        plt.xticks([])
        plt.yticks([])
        # stick images to one another
        plt.subplots_adjust(wspace=0, hspace=0)


        """# For normal images:
        # Select ticks font size
        plt.tick_params(axis='both', which='major', labelsize=6)

        # stick images to one another
        plt.subplots_adjust(wspace=1, hspace=1.5)

        # Alternate labels position height
        if classes.index(class_) % 2 == 0:
            plt.xlabel(class_, fontsize=6, rotation=0, labelpad=1)
        else:
            plt.xlabel(class_, fontsize=6, rotation=0, labelpad=7)"""
        
        plt.imshow(img)
    
    plt.show()

# Count the number of images in a class
def count_im_class(path, class_):
    images = os.listdir(os.path.join(path, class_))
    return len(images)

# Count the number of images in Camelyon_16_0 and janowczyk6_0
def count_maj(train, test, val):
    sets = [train, test, val]
    tot_cam = 0
    tot_jan = 0
    tot_im = 0
    for s in sets:
        set_name = s.split("/")[-1]
        nb_im_cam = count_im_class(s, "camelyon16_0")
        nb_im_jan = count_im_class(s, "janowczyk6_0")
        print("For set " + str(set_name) + ": Camelyon16_0 has "+ str(nb_im_cam) + " images")
        print("For set " + str(set_name) + ": janowczyk6_0 has "+ str(nb_im_jan) + " images")
        su = 0
        for c in os.listdir(s):
            su += count_im_class(s, c)
        print("For set " + str(set_name) + "the total number of images is "+str(su) + " images")
        print("The percentages are: "+str(nb_im_cam/su) + " for camelyon16_0 and "+str(nb_im_jan/su) + " for janowczyk6_0")
        print("So a percentage together of "+str(nb_im_cam/su + nb_im_jan/su) + " for a total of images together of " + str(nb_im_cam + nb_im_jan) + " images")

        tot_cam += nb_im_cam
        tot_jan += nb_im_jan
        tot_im += su
    print("For the whole database: Camelyon16_0 has "+ str(tot_cam) + " images, for a percentage of "+str(tot_cam/tot_im))
    print("For the whole database: janowczyk6_0 has "+ str(tot_jan) + " images, for a percentage of "+str(tot_jan/tot_im))
    print("For the whole database: the total number of images is "+str(tot_im) + " images")
    print("Together, the percentage is: "+str(tot_cam/tot_im +tot_jan/tot_im) + ", for a total of images together of " + str(tot_cam + tot_jan) + " images")

# Bar plot of the number of images per class  
def bar_plot(train, test, val):
    sets = [train, test, val]
    i = 0
    plt.figure(figsize=(15, 9)) # width - height
    for s in sets:
        i += 1
        set_name = s.split("/")[-1]
        nb_per_class = []
        classes = os.listdir(s)
        classes.sort()
        tot = 0
        for c in classes:
            tot += count_im_class(s, c)
            if c != "camelyon16_0" and c != "janowczyk6_0":
                nb_per_class.append(count_im_class(s, c))
        classes = utils.rename_classes(s)
        classes.remove("camelyon16_0")
        classes.remove("janowczyk6_0")

        # bar plot in subplots
        plt.subplot(3, 1, i)
        # Increasing bar widths
        plt.bar(classes, nb_per_class)
        # No ticks if not the last subplot
        # Add percentage of images for each class on top of the bar 
        for j in range(len(classes)):
            if classes[j] != "camelyon16_0" or classes[j] != "janowczyk6_0":
                # Computation of the value up to 4 decimals
                val = nb_per_class[j]/tot*100
                if val > 1:
                    val = round(val, 2)
                    val = "{:.2f}".format(val)
                else:
                    val = round(val, 3)
                    val = "{:.3f}".format(val)

                # Y position of text
                if i == 1:
                    if nb_per_class[j] > np.max(nb_per_class) -10000:
                        y = nb_per_class[j] - 7000
                        x = j+0.6
                    else:
                        y = nb_per_class[j] + 2000
                        x = j - 0.15
                else:
                    if nb_per_class[j] > np.max(nb_per_class)-1000:
                        y = nb_per_class[j] - 1000
                        x = j+0.6
                    else:
                        y = nb_per_class[j] + 250
                        x = j - 0.15
                plt.text(x = x, y = y, s = str(val)+"%", size = 7, rotation = 90)

        if i != 3:
            plt.xticks([])
        else:
            # Rotate labels
            plt.xticks(rotation=90)
            
            # change font size
            plt.tick_params(axis="x", which='major', labelsize=8)
        plt.yticks(rotation=180)
        plt.ylabel("Number of images in "+str(set_name))
        # add padding between subplots
        plt.subplots_adjust(hspace=0.1)
        # Remove vertical space on top of first subplot
        plt.subplots_adjust(top=0.95)
        # Add bottom space to see labels entierely
        plt.subplots_adjust(bottom=0.2)
    plt.show()

# Compute the mean width and height for each class, alongside other statistics 
def width_height(train, test, val):
    sets = [train, test, val]
    widths = np.zeros((67, 1))
    heights = np.zeros((67, 1))
    nb = np.zeros((67, 1))
    classes = os.listdir(sets[0])
    classes.sort()
    for c in range(len(classes)):
        widths_l = []
        heights_l = []
        for s in sets:
            images = os.listdir(os.path.join(s, classes[c]))
            for im in images:
                im = Image.open(os.path.join(s, classes[c], im))
                widths_l.append(im.size[0])
                heights_l.append(im.size[1])
                widths[c] += im.size[0]
                heights[c] += im.size[1]
                nb[c] += 1
        print(classes[c])
        print(widths[c]/nb[c])
        print(heights[c]/nb[c])
        print(nb[c])
        """print(np.max(np.array(widths_l)))
        print(np.max(np.array(heights_l)))
        print(np.median(np.array(widths_l)))
        print(np.median(np.array(heights_l)))
        print(np.min(np.array(widths_l)))
        print(np.min(np.array(heights_l)))"""
        print(np.std(np.array(widths_l)))
        print(np.std(np.array(heights_l)))

    widths = widths/nb
    heights = heights/nb
    
    print(np.mean(widths))
    print(np.mean(heights))

# Visualize the transformation
def vis_transf(path):
    # Load the image
    img = Image.open(path)
    # Custom transforms
    transform = transforms.Compose(
                [transforms.RandomVerticalFlip(.5),
                transforms.RandomHorizontalFlip(.5),
                transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.4),
                transforms.RandomResizedCrop(224, scale = (.8,1)),
                transforms.RandomApply([transforms.GaussianBlur(23)]),
                transforms.RandomRotation(random.randint(0,360)),
                transforms.RandomGrayscale(0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ]
            )
    # AMDIM transforms https://github.com/Lightning-Universe/lightning-bolts/blob/5669578aba733bd9a7f0403e43dd6cfdcfd91aac/src/pl_bolts/transforms/self_supervised/amdim_transforms.py
    transform1 = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],    
                                    std=[0.229, 0.224, 0.225])])
            
    #SimCLR transforms, pytorch lightning bolt 
    transform2 = transforms.Compose(
        [transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter()], 0.8),
        transforms.RandomGrayscale(0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
        transforms.transforms.ToTensor()])
    
    # Transform the image - one by one 
    """transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),])

    transform2 = transforms.Compose([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0.1),
        transforms.ToTensor(),])
    
    transform3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[200.69/255, 161.54/255, 189.47/255],
    #                        std=[30.69/255, 39.72/255, 29.7/255]),
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])"""
    # Plot the images (parameters to be changed depending on the situation)
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    transform_list = [transform, transform1, transform2]
    for t in transform_list:
        plt.subplot(1, 4, transform_list.index(t)+2)
        im = t(img)
        im = np.array(im.permute(1, 2, 0))
        plt.imshow(im)

    plt.show()
# Compute the mean and std of the pixel values of the images
def means_pixel(sets):
    means = np.zeros((3,1))
    stds = np.zeros((3,1))
    tot = 0
    for s in sets:
        classes = os.listdir(s)
        classes.sort()
        for c in classes:
            print(c)
            images = os.listdir(os.path.join(s, c))
            for im in images:
                tot += 1
                # Open in RGB
                im = Image.open(os.path.join(s, c, im)).convert('RGB')

                im = np.array(im)
                means[0] += np.mean(im[:,:,0]/255)
                means[1] += np.mean(im[:,:,1]/255)
                means[2] += np.mean(im[:,:,2]/255)
                stds[0] += np.std(im[:,:,0]/255)
                stds[1] += np.std(im[:,:,1]/255)
                stds[2] += np.std(im[:,:,2]/255)
    print("Mean of the pixel values: ", means/tot)
    print("Std of the pixel values: ", stds/tot)

# visualize the impatcof the resizing on different images resolutions
def resized_vis(paths):
    transformRes = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1)),
        transforms.ToTensor(),])
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=100, sharex=True, sharey=True)
    for p in paths:
        img = Image.open(p)
        im = transformRes(img)
        im = np.array(im.permute(1, 2, 0))
        ax[1, paths.index(p)].imshow(im)
        
        ax[0, paths.index(p)].imshow(img)
    # Remove vertical space on the bottom and top for the whole plot
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.05)

    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,7), dpi=100, sharex=True, sharey=True)
    
    img = Image.open(paths[1])
    ax[0].imshow(img)
    im = transformRes(img)
    im = np.array(im.permute(1, 2, 0))
    ax[1].imshow(im)
    plt.show()



def diversity():
    im1_cam = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/camelyon16_0/27664454_1536_188928_768_768.png'
    im2_cam = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/camelyon16_0/27664454_33792_118272_768_768.png'
    im3_cam = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/camelyon16_0/27671570_26880_79872_768_768.png'
    im4_cam = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/camelyon16_0/27698189_67584_69120_768_768.png'
    l1 = [im1_cam, im2_cam, im3_cam, im4_cam]

    im1_jan = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk6_0/8918_idx5_x151_y751_class0.png'
    im2_jan = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk6_0/8918_idx5_x251_y1401_class0.png'
    im3_jan = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk6_0/9181_idx5_x251_y1901_class0.png'
    im4_jan = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk6_0/14191_idx5_x2551_y601_class0.png'
    l2 = [im1_jan, im2_jan, im3_jan, im4_jan]

    im1_ici = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/iciar18_micro_113351608/108_118327654_512_512_512_512.png'
    im2_ici = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/iciar18_micro_113351608/111_118327082_0_512_512_512.png'
    im3_ici = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/iciar18_micro_113351608/112_118327012_1536_512_512_512.png'
    im4_ici = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/iciar18_micro_113351608/128_118325650_1536_1024_512_512.png'
    l3 = [im1_ici, im2_ici, im3_ici, im4_ici]

    im1_ulb_ana = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/ulb_anapath_lba_4720/377614_799253.png'
    im2_ulb_ana = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/ulb_anapath_lba_4720/2352_7986752.png'
    im3_ulb_ana = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/ulb_anapath_lba_4720/2248_7982522.png'
    im4_ulb_ana = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/ulb_anapath_lba_4720/377626_796526.png'
    l4 = [im1_ulb_ana, im2_ulb_ana, im3_ulb_ana, im4_ulb_ana]

    im1_jano7 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk7_113347673/114368428_384_384_384_384.png'
    im2_jano7 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk7_113347673/114370161_768_0_384_384.png'
    im3_jano7 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk7_113347673/114370869_384_384_384_384.png'
    im4_jano7 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/janowczyk7_113347673/114372826_384_384_384_384.png'
    l5 = [im1_jano7, im2_jano7, im3_jano7, im4_jano7]

    im1_umcm = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/umcm_colorectal_04_LYMPHO/02_001_1093E_Row_1_Col_1.tif'
    im2_umcm = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/umcm_colorectal_04_LYMPHO/02_002b_5B23_Row_301_Col_1.tif'
    im3_umcm = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/umcm_colorectal_04_LYMPHO/03_032_5971_Row_151_Col_1.tif'
    im4_umcm = '/home/labarvr4090/Documents/Axelle/cytomine/Data/test/umcm_colorectal_04_LYMPHO/10_c_5945_Row_1_Col_151.tif'
    l6 = [im1_umcm, im2_umcm, im3_umcm, im4_umcm]

    l = [l1, l2, l3, l4, l5, l6]

    for i in l:
        plt.figure(figsize=(5, 2))
        for j in range(4):
            plt.subplot(1, 4, j+1)
            plt.imshow(Image.open(i[j]))
            # change font size ticks
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
        plt.show()




# Run the wanted function
if __name__ == "__main__":
    val = "/home/labarvr4090/Documents/Axelle/cytomine/Data/validation"
    train = "/home/labarvr4090/Documents/Axelle/cytomine/Data/train"
    test = "/home/labarvr4090/Documents/Axelle/cytomine/Data/test"

    #summary_image(train)

    #count_maj(train, test, val)

    #bar_plot(train, test, val)
    
    #width_height(train, test, val)

    p = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk5_1/01_117238845_1502_393_250_250_2.png'
    #p2 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk6_0/8863_idx5_x651_y1551_class0.png'
    #p3 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/iciar18_micro_113351562/17_118316818_512_0_512_512.png'
    #paths = [p, p2, p3]
    vis_transf(p)

    #resized_vis(paths)

    #means_pixel([train, test, val])

    #diversity()
