

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

def prob_per_class(path, path_queries):      
    classes = os.listdir(path)
    classes.sort()
    count = 0
    counts = []
    for class_ in classes:
        images = os.listdir(os.path.join(path, class_))
        count += len(images)
        counts.append(len(images))

    queries_classes = os.listdir(path_queries)
    queries_classes.sort()
    props = []
    tot_queries = 0
    for query_c in queries_classes:
        prp_c = []
        print("Current class is",query_c)
        queries = os.listdir(os.path.join(path_queries, query_c))
        tot_queries += len(queries)
        for query in queries:
            prop = []
            class_im = query_c
            retrieved = random_draw(count, path, classes, counts)
            # Fill prop with the proportion of correct images at each step 
            for i in range(len(retrieved)):
                class_retr = utils.get_class(retrieved[i])
                # The image is correct, have to add 1 
                if class_retr == class_im:  
                    if len(prop) == 0:
                        prop.append(1) 
                    else:
                        prop.append(prop[-1] + 1)   
                else:
                    if len(prop) == 0:
                        prop.append(0)
                    else:
                        prop.append(prop[-1])  

            for i in range(len(retrieved)):
                prop[i] = prop[i] / (i+1) 
            if len(prp_c) == 0:
                prp_c = prop
            else:
                for el in range(len(prop)):
                    prp_c[el] += prop[el] 
        for el in range(len(prp_c)):
            prp_c[el] = prp_c[el]/len(queries)
        props.append(prp_c)
    
    plt.figure(figsize=(12, 8))
    c = ["camelyon16_0", "janowczyk6_0", "janowczyk6_1", "tupac_mitosis_0", "mitos2014_2", "umcm_colorectal_04_LYMPHO", "ulg_lbtd_lba_4762", "ulg_bonemarrow_0"]
    for i in range(len(props)) :
        if queries_classes[i] in c:
            plt.plot(props[i], label=queries_classes[i], marker='o')
        #plt.plot(props[i], label=queries_classes[i], marker='o')
    
    plt.title('Proportion of Correct Images per Model')
    plt.xlabel('Retrieval Step')
    plt.ylabel('Proportion of Correct Images')
    plt.xticks(np.arange(10), labels=np.arange(1, 11))  # Label steps 1 to 10
    plt.legend()
    plt.grid()
    plt.show()

def prob_vs_size(path, path_queries):
    classes = os.listdir(path)
    classes.sort()
    count = 0
    counts = []
    for class_ in classes:
        images = os.listdir(os.path.join(path, class_))
        count += len(images)
        counts.append(len(images))

    queries_classes = os.listdir(path_queries)
    queries_classes.sort()
    props_stat = []
    tot_queries = 0
    for i in range(50):
        props = []
        for query_c in queries_classes:
            prop = 0
            print("Current class is",query_c)
            queries = os.listdir(os.path.join(path_queries, query_c))
            tot_queries += len(queries)
            for query in queries:
                class_im = query_c
                retrieved = random_draw(count, path, classes, counts, n = 1)
                class_retr = utils.get_class(retrieved[0])
                # The image is correct, have to add 1 
                if class_retr == class_im:  
                    prop += 1 
            prop = prop/len(queries)
            props.append(prop)
        if len(props_stat) == 0:
            props_stat = props
        else:
            for el in range(len(props)):
                props_stat[el] += props[el]
    for el in range(len(props_stat)):
        props_stat[el] = props_stat[el]/50
        
    plt.figure(figsize=(12, 8))
    
    plt.scatter(counts, props_stat)
    
    plt.title('Proportion of Correct Images per size of the class')
    plt.xlabel('Number of images in the indexing Set of the class')
    plt.ylabel('Proportion of Correct Images') # Label steps 1 to 10
    plt.grid()
    plt.show()
def plot_im_curve(path, path_queries):
    # Model 1 Props
    model_1_props = [
        0.81835405, 0.81886411, 0.81869062, 0.81832802, 0.81773156,
        0.81731657, 0.81703501, 0.81682645, 0.81672323, 0.81618679
    ]

    # Model 2 Props
    model_2_props = [
        0.79779527, 0.7977224, 0.79769811, 0.79744134, 0.79733308,
        0.79706487, 0.79679, 0.79661118, 0.79624309, 0.79595382
    ]

    # Model 3 Props
    model_3_props = [
        0.7590823, 0.75592822, 0.75302743, 0.75145213, 0.74987196,
        0.74838132, 0.74714259, 0.74604699, 0.74486638, 0.74384486
    ]

    # Model 4 Props
    model_4_props = [
        0.77414486, 0.77171944, 0.77169168, 0.77022568, 0.76930027,
        0.76861047, 0.76808801,  0.76735656, 0.76656558, 0.76615348
    ]

    # Model 5 Props
    model_5_props = [
        0.76858618, 0.76640018, 0.76478671, 0.76403983, 0.76293173,
        0.76225893, 0.76173226, 0.76118372, 0.76070387, 0.76017842
    ]
    
    # Model 6 Props
    model_6_props = [
        0.74470676, 0.74472758, 0.74504334, 0.74433462, 0.74407387,
        0.74328934, 0.74273937, 0.742228, 0.74161861, 0.74106552
    ]

    # Model 7 Props
    model_7_props = [
        0.74495659, 0.74407699, 0.74331536, 0.74289811, 0.7425978,
        0.74252424, 0.74225161, 0.74184155, 0.74088069, 0.74046281
    ]

    # Model 8 Props
    model_8_props = [
        0.63808215, 0.63216955, 0.62810637, 0.62494795, 0.6220182,
        0.61943525, 0.61750255, 0.61553125, 0.61375281, 0.61211355
    ]

    # Model 9 Props
    model_9_props = [
        0.73684758, 0.7322726, 0.72914455, 0.7271277, 0.72565736,
        0.723801, 0.72246313, 0.72118257, 0.72005242, 0.71911186
    ]

    # Model 10 Props
    model_10_props = [
        0.75780193, 0.75546499, 0.7527325, 0.75037474, 0.74866862,
        0.74689623, 0.74550384, 0.74432682, 0.74326678, 0.74230633
    ]

    model_11_props = [0.76778465, 0.76523952, 0.76405804, 0.76308736, 0.76263819, 
                      0.76177836, 0.76097088, 0.76066064, 0.76003303, 0.75963192]

    model_12_props = [0.75903025, 0.75691712, 0.75586576, 0.75480139, 0.75377345,
                        0.75293028, 0.75217484, 0.75147815, 0.75074081, 0.75015406]

    model_13_props = [0.76420378, 0.76188766, 0.76009202, 0.75831720, 0.75717944,
                        0.75589178, 0.75491552, 0.75409614, 0.75331196, 0.75254513]

    model_14_props = [ 0.78047384, 0.77723128, 0.77580344, 0.77432182, 0.77318927, 0.77228329, 0.77130901, 0.77078129, 0.77017537, 0.76955426]

    model_15_props = [ 0.58391106, 0.58067370, 0.57777292, 0.57559386, 0.57398663, 0.57233395, 0.57088728, 0.56943403, 0.56823550, 0.56716112]
   
    model_16_props = [ 0.79346491, 0.79146108, 0.79071333, 0.78950149, 0.78865363, 
                      0.7879843, 0.78728166, 0.78669743, 0.78607647, 0.7854829]
    model_17_props = [0.75852018, 0.75883247, 0.76045635, 0.75952991, 0.75847855, 
                      0.75757292, 0.75708367, 0.75379427, 0.75473112, 0.75465826]
   

    props = [11113.0, 11136.0, 11191.333333334209, 11155.0, 11172.399999999667, 11147.333333333503, 11149.571428567484, 11143.375, 11153.22222223129, 11148.100000003062]


    """
    # Count number of images of the test dataset
    classes = os.listdir(path)
    classes.sort()
    count = 0
    counts = []
    for class_ in classes:
        images = os.listdir(os.path.join(path, class_))
        count += len(images)
        counts.append(len(images))

    queries_classes = os.listdir(path_queries)
    props = []
    tot_queries = 0
    for query_c in queries_classes:
        print("Current class is",query_c)
        queries = os.listdir(os.path.join(path_queries, query_c))
        tot_queries += len(queries)
        for query in queries:
            prop = []
            class_im = query_c
            retrieved = random_draw(count, path, classes, counts)
            # Fill prop with the proportion of correct images at each step 
            for i in range(len(retrieved)):
                class_retr = utils.get_class(retrieved[i])
                # The image is correct, have to add 1 
                if class_retr == class_im:  
                    if len(prop) == 0:
                        prop.append(1) 
                    else:
                        prop.append(prop[-1] + 1)   
                else:
                    if len(prop) == 0:
                        prop.append(0)
                    else:
                        prop.append(prop[-1])  

            for i in range(len(retrieved)):
                prop[i] = prop[i] / (i+1) 
            if len(props) == 0:
                props = prop
            else:
                for el in range(len(prop)):
                    props[el] += prop[el] """
    print(props)
    # Normalize the accumulated props
    for el in range(len(props)):
        props[el] = props[el]/96066 
    extractors = ["resnet", "deit", "dino_p", "dino_s", "dino_f", "byol_s", "byol_f", "byol_p", "ret_ccl", "ibot_p", "ibot_f", "ibot_s", "ctranspath", "phikon", "cdpath", "uni", "hoptim", "random"]   
    model_prop = [model_1_props, model_2_props, model_3_props, model_4_props, model_5_props, model_6_props, model_7_props, model_8_props, model_9_props, model_10_props, model_11_props, model_12_props, model_13_props, model_14_props, model_15_props, model_16_props, model_17_props, props]
    colors = [
        "blue", "navy", "olive", "green", "lime", "orange", "coral",  "gold",  "red", "maroon", "purple", "brown", "pink", "gray", "cyan",
        "teal", "yellow", "magenta"
    ]

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for i in range(len(extractors)) :
        plt.plot(model_prop[i], label=extractors[i], marker='o', color=colors[i])
    
    plt.title('Proportion of Correct Images per Model')
    plt.xlabel('Retrieval Step')
    plt.ylabel('Proportion of Correct Images')
    plt.xticks(np.arange(10), labels=np.arange(1, 11))  # Label steps 1 to 10
    plt.legend()
    plt.grid()
    plt.show()
    
def random_draw(count, path, classes, counts , n = 10):
    
    retrieved = []
    for i in range(0, n):
        idx = random.randint(0, count-1)
        i = 0
        for class_ in classes:
            if idx < counts[i]:
                images = os.listdir(os.path.join(path, class_))
                retrieved.append(os.path.join(path, class_, images[idx]))
                break
            else:
                idx -= counts[i]
            i += 1
    return retrieved
# Count the number of images in a class
def count_im_class(path, class_):
    images = os.listdir(os.path.join(path, class_))
    return len(images)
def count_all_classes(path):
    min = -1
    min_name = ""
    max = 0
    max_name = ""
    tot = 0
    for c in os.listdir(path):
        count = count_im_class(path, c)
        tot += count
        if count < min or min == -1:
            min = count
            min_name = c
        if count > max:
            max = count
            max_name = c

        print("For class "+str(c)+": "+str(count)+" images")
    print("The class with the minimum number of images is "+str(min_name)+" with "+str(min)+" images")
    print("The class with the maximum number of images is "+str(max_name)+" with "+str(max)+" images")
    print("The total number of images is "+str(tot))
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
    val = "/home/labsig//Documents/Axelle/cytomine/Data/our/validation"
    train = "/home/labsig//Documents/Axelle/cytomine/Data/our/train"
    test = '/home/labsig/Documents/Axelle/cytomine/Data/our/test'

    plot_im_curve(test, val)
    #prob_per_class(test, val)
    #prob_vs_size(test, val)
    #summary_image(train)

    #count_all_classes(test)
    #count_maj(train, test, val)

    #bar_plot(train, test, val)
    
    #width_height(train, test, val)

    p = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk5_1/01_117238845_1502_393_250_250_2.png'
    #p2 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk6_0/8863_idx5_x651_y1551_class0.png'
    #p3 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/iciar18_micro_113351562/17_118316818_512_0_512_512.png'
    #paths = [p, p2, p3]
    #vis_transf(p)

    #resized_vis(paths)

    #means_pixel([train, test, val])

    #diversity()
