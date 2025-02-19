from matplotlib import pyplot as plt
import torch
import os

import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from collections import defaultdict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ----------------------------------- Training ----------------------------------- #

# Class for the rest of the methods for training 
class TrainingDataset(Dataset):
    def __init__(self, root, model_name, samples_per_class, need_val=0, informative_samp = True):

        # 1. Load the dataset + generalise it if needed
        self.classes = os.listdir(root)
        self.classes.sort()
        
        
        # 2. Create a dictionary to convert class name to number and vice versa
        self.conversion = {x: i for i, x in enumerate(self.classes)} # Number given the class
        self.conv_inv = {i: x for i, x in enumerate(self.classes)} # class given the number
        self.image_dict = {}
        self.image_list = defaultdict(list)


        print("================================")
        print("Loading dataset")
        print("================================")
        
        # 3. Create a dictionary of images and their class
        i = 0
        for c in self.classes:
            for dir, subdirs, files in os.walk(os.path.join(root, c)):
                nb_im_c = len(files)
                cpt_c = 0
                files.sort()
                for file in files:
                    if need_val != 1:
                        if need_val == 0 or cpt_c <= 0.98 * nb_im_c:
                            img = os.path.join(dir, file)
                            cls = dir[dir.rfind("/") + 1:]
                            self.image_dict[i] = (img, self.conversion[cls])
                            self.image_list[self.conversion[cls]].append(img)
                            i += 1
                    else:
                        if cpt_c > 0.98 * nb_im_c:
                            img = os.path.join(dir, file)
                            cls = dir[dir.rfind("/") + 1:]
                            self.image_dict[i] = (img, self.conversion[cls])
                            self.image_list[self.conversion[cls]].append(img)
                            i += 1
                    cpt_c += 1
        
            
        self.transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                    transforms.RandomResizedCrop(224, scale=(.7,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        # 5. Set remaining variables
        self.samples_per_class = samples_per_class
        self.current_class = np.random.choice(self.classes)
        self.classes_visited = [self.current_class, self.current_class] 
        self.n_samples_drawn = 0
        self.model_name = model_name
        self.is_init = True
        self.needs_val = need_val
        self.informative_samp = informative_samp



    def __len__(self):
        return len(self.image_dict)

    # https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/datasets.py#L428
    def __getitem__(self, idx):
        if self.informative_samp == "False" or self.needs_val == 1:
            img = Image.open(self.image_dict[idx][0]).convert('RGB')
            return self.image_dict[idx][1], self.transform(img), 
        else:
            # no image drawn and thus no class visited yet
            if self.is_init:
                self.current_class = self.classes[idx % len(self.classes)]
                self.classes_visited = [self.current_class]
                self.is_init = False
                # Select the class to draw from till we have drawn samples_per_class images

            if self.samples_per_class == 1:
                img = Image.open(self.image_dict[idx][0]).convert('RGB')
                return self.image_dict[idx][0], self.transform(img)

            if self.n_samples_drawn == self.samples_per_class:
                counter = [cls for cls in self.classes if cls not in self.classes_visited]
                if len(counter) == 0:
                    self.current_class = self.classes[idx % len(self.classes)]
                    self.classes_visited = [self.current_class]
                    self.n_samples_drawn = 0
                else:
                    self.current_class = counter[idx % len(counter)]
                    self.classes_visited.append(self.current_class)
                    self.n_samples_drawn = 0

            # Find the index corresponding to the class we want to draw from and then
            # retrieve an image from all the images belonging to that class.
            class_nbr = self.conversion[self.current_class]
            class_sample_idx = idx % len(self.image_list[class_nbr])
            self.n_samples_drawn += 1


            img = Image.open(self.image_list[class_nbr][class_sample_idx]).convert('RGB')

            #if self.model_name == 'deit':
            #    return class_nbr, self.transform(images=img, return_tensors='pt')['pixel_values']
            
            return class_nbr, self.transform(img)

# ----------------------------------- Indexing ----------------------------------- #
class AddDataset(Dataset):
    def __init__(self, root, model_name, generalise=0):
        self.root = root
        self.model_name = model_name
        self.list_img = []
        if model_name == "hoptim":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.707223, 0.578729, 0.703617], 
                    std=[0.211883, 0.230117, 0.177517]
                    ),
            ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.classes = os.listdir(root)
        for c in self.classes:
                for dir, subdir, files in os.walk(os.path.join(root, c)):
                    for f in files:
                        self.list_img.append(os.path.join(dir, f))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img = Image.open(self.list_img[idx]).convert('RGB') 
        
        return self.transform(img), self.list_img[idx]

   