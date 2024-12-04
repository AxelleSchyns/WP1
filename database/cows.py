import argparse
import os
import time
import cv2

from matplotlib import pyplot as plt
import numpy as np
import torch
import models
import db
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class AddCowsDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.list_img = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    
        for file in os.listdir(root):
            if os.path.isdir(os.path.join(root, file)) is False:
                self.list_img.append(os.path.join(root, file))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img = Image.open(self.list_img[idx]).convert('RGB') 
        
        return self.transform(img), self.list_img[idx]


def create_mosaic(path):
    try:
        os.mkdir(path + '/mosaic')
    except FileExistsError:
        print('Mosaic folder already exists')
        exit(1)

    for files in os.listdir(path):
        if os.path.isdir(path+'/'+files) is False:
            im = Image.open(path+'/'+files)
            width, height = im.size
            new_width = width // 4
            new_height = height // 4
            for i in range(4):
                for j in range(4):
                    im.crop((i*new_width, j*new_height, (i+1)*new_width, (j+1)*new_height)).save(path + '/mosaic/' + str(i) + str(j) + '_' + files)


def turn_gray(path):
    try:
        os.mkdir(path + '/gray')
    except FileExistsError:
        print('Gray folder already exists')
        exit(1)

    for files in os.listdir(path):
        if os.path.isdir(path+'/'+files) is False:
            im = Image.open(path+'/'+files)
            im = im.convert('L')
            im = im.convert('RGB')
            im.save(path + '/gray/' + files)

def visualize_transform(path):
    try:
        os.mkdir(path + '/transformed')
    except FileExistsError:
        print('Transformed folder already exists')
        exit(1)

    transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    for files in os.listdir(path):
        if os.path.isdir(path+'/'+files) is False:
            im = Image.open(path+'/'+files)
            im = transform(im)

            # save transformed image
            im = transforms.ToPILImage()(im)
            im.save(path + '/transformed/' + files)

@torch.no_grad()       
def add_cows_dataset( data_root, database):
    # Create a dataset from a directory root
    data = AddCowsDataset(data_root)
    loader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=12, pin_memory=True, shuffle = True)
    t_model = 0
    t_indexing = 0
    t_transfer = 0
    for i, (images, filenames) in enumerate(loader):
        images = images.view(-1, 3, 224, 224).to(device=next(database.model.parameters()).device)
        t = time.time()
        out = database.model(images)
        t_im = time.time() - t 

        t = time.time()
        out = out.cpu()
        t_transfer = t_transfer + time.time() - t

        t = time.time()
        
        # check if out contains nan
        if np.isnan(out.numpy()).any():
            print("Nan in output")

        database.add(out.numpy(), list(filenames))
        t_im_ind = time.time() - t
        t_indexing = t_indexing + t_im_ind
        t_model = t_model + t_im
    print("Time of the model: "+str(t_model))
    print("Time of the transfer: "+str(t_transfer))
    print("Time of the indexing: "+str(t_indexing))
    database.save()

def extract_from_vids(path_vid, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video_capture = cv2.VideoCapture(path_vid)
    success, frame = video_capture.read()
    count = 0

    # Loop through frames
    while success:
        # Construct the filename for each frame
        frame_filename = os.path.join(output_folder, f"frame_{count:05d}.jpg")
        
        # Save the current frame as a JPG file
        if count % 20 == 0:
            cv2.imwrite(frame_filename, frame)
        #cv2.imwrite(frame_filename, frame)

        # Read the next frame
        success, frame = video_capture.read()
        count += 1
    print(count)
    # Release the video capture object
    video_capture.release()
    print("Frame extraction completed.")

def index_from_vids(path_vid, output_folder, database):
    
    extract_from_vids(path_vid, output_folder)
    add_cows_dataset(output_folder, database)

def retrieve_im(database, transform, query, extractor, dir, path_indexing):
    ret_values = database.search(transform(Image.open(query).convert('RGB')), extractor,nrt_neigh= 10) 

    # Retrieve the most similar images
    names = ret_values[0]
    dist = ret_values[1]

    class_names = []
    classement = 0 
    Image.open(query).convert('RGB').save(os.path.join(dir, "3_query_image.png"))

    # Retrieve the class of each retrieved image
    for n in names:
        classement += 1
        img = Image.open(n).convert('RGB')

        img.save(os.path.join(dir,str(classement) + "_" +n[n.rfind('/')+1:] ))
        
        small_n = n[n.rfind('/')+1:]
        if args.exp_number == 0:
            if small_n in os.listdir(path_indexing):
                class_names.append(os.path.join(path_indexing, small_n))
        else:
            if small_n in os.listdir(path_indexing+'/mosaic'):
                class_names.append(os.path.join(path_indexing+'/mosaic', small_n))
    
    # Subplot of the image and the nearest images
    plt.figure(figsize=(7,3))
    plt.subplot(2,6,1)
    plt.imshow(Image.open(query).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.axis('off')
    print(class_names)
    for i in range(2,12):
        plt.subplot(2,6,i)
        plt.imshow(Image.open(class_names[i-2]).convert('RGB'))

        plt.axis('off')
    plt.savefig(os.path.join(dir, "10_nearest_images.png"))
    

if __name__ == "__main__":
    use = 3
    if use == 0:
        create_mosaic('/home/axelle/Documents/Doctorat/cows_images/test/class1')
    elif use == 1:
        turn_gray('/home/axelle/Documents/Doctorat/cows_images/test/class1')
    elif use == 2:
        visualize_transform('/home/axelle/Documents/Doctorat/cows_images/test/class1/mosaic/gray')
        #turn_gray('/home/axelle/Documents/Doctorat/cows_images/test/class1/mosaic')
    elif use == 3:
        extract_from_vids('/home/axelle/Documents/Doctorat/Images et vidéos test CBIR/Vidéos/Post-ébourgeonnage/2037 Post-ébourgeonnage AP.mp4', '/home/axelle/Documents/Doctorat/cows_images/extraction_from_vids')
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--extractor',
            default='densenet'
        )

        parser.add_argument(
            '--weights'
        )

        parser.add_argument(
            '--db_name',
            default = 'db'
        )

        parser.add_argument(
            '--num_features',
            default = 128,
            type=int
        )

        parser.add_argument(
            '--path_indexing'
        )

        parser.add_argument(
            '--exp_number',
            default = 0,
            type = int
        )

        parser.add_argument(
            '--dir',
        )

        parser.add_argument(
            '--query',
        )

        args = parser.parse_args()
        
        # Retrieve the pretrained model 
        model = models.Model(num_features=args.num_features, model=args.extractor, weight=args.weights)

        # Create the database
        database = db.Database(args.db_name, model, load=False)

        model = models.Model(model=args.extractor, num_features=args.num_features, 
                            weight=args.weights)
        
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # index the images
        if args.exp_number == 0: # if the experiment number is 0, add the dataset as is
            add_cows_dataset(args.path_indexing+'/gray/', database)
            retrieve_im(database, transform, args.query, args.extractor, args.dir, args.path_indexing)
        elif args.exp_number == 1: # if the experiment number is not 0, add the dataset after splitting each image in sub images
            add_cows_dataset(args.path_indexing+'/mosaic/gray/', database)
            retrieve_im(database, transform, args.query, args.extractor, args.dir, args.path_indexing)
        else: 
            index_from_vids(args.path_indexing, '/extraction_from_vids', database)

            for files in os.listdir(args.query):
                ret_values = database.search(transform(Image.open(files).convert('RGB')), args.extractor, nrt_neigh= 10) 

                # Retrieve the most similar images
                names = ret_values[0]
                dist = ret_values[1]

                # Retrieve the class of each retrieved image
                for n in names:
                    