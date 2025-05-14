import torch
from database.db import Database
import dataset
import time
import timm

import numpy as np
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import resnet_ret as ResNet_ret
import arch

from argparse import ArgumentParser
from PIL import Image

if __name__ == "__main__":

    #1 Parse the arguments
    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--indexing',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--path_indexing',
        help='path to the folder that contains the images to add',
        default=None
    )

    parser.add_argument(
        '--query',
        help='path to the folder that contains the images to add',
        default=None
    )
    args = parser.parse_args()
    # Load the feature extractor 
    model = models.Model(model="resnet", num_features=128, weight='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder/resnet/model1/last_epoch')
    
    if model is None:
        exit(-1)

    if args.indexing: 
        # Initialize the database
        print("database starting")
        database = Database(args.db_name, model, load = False)
        # Indexed the images in the database
        t = time.time()
        database.add_dataset(args.path_indexing, "resnet")
        print("T_indexing = "+str(time.time() - t))

    else:
        # Load the database
        print("Loading the database")
        database = Database(args.db_name, model, load = True)

    # 1. Retrieve the result corresponding to the query
    img = Image.open(args.query).convert('RGB')
    names, distance, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(img, "resnet", nrt_neigh=1)
    
         
    