from argparse import ArgumentParser
import os
import numpy as np

import torch

import torchvision.transforms as transforms
from explanations import SBSM

from PIL import Image
from matplotlib import pyplot as plt
import redis
import torch
import db
import faiss
import time
import models
import utils


from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
        


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Put model loading here#1 Parse the arguments
    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--weight',
        help='path to the folder that contains the weights',
        default='/home/labsig/Documents/Axelle/cytomine/WP1/weights_folder/resnet/model1/last_epoch'
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

    parser.add_argument(
        '--nb_masks',
        help='number of masks to generate',
        default=10000
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor to use',
        default='resnet'
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------------------
    #                                     1. Load of items
    # ----------------------------------------------------------------------------------------------------------

    # Load the feature extractor 
    model = models.Model(model=args.extractor, weight=args.weight, device ='cuda:0')
    nb_features = model.num_features
    if model is None:
        exit(-1)

    # Load the databse 
    if args.indexing: 
        print("database starting")
        database = db.Database(args.db_name, model, load = False)
        t = time.time()
        database.add_dataset(args.path_indexing, "resnet")
        print("T_indexing = "+str(time.time() - t))
    else:
        print("Loading the database")
        database = db.Database(args.db_name, model, load = True)

    # Load the query
    query_im = Image.open(args.query).convert('RGB')
    feat_extract = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
    query_im = feat_extract(query_im)
    cl_query = utils.get_class(args.query)
    vec_query = model.get_vector(query_im)


    # ----------------------------------------------------------------------------------------------------------
    #                                     2. Find the closest vector
    # ----------------------------------------------------------------------------------------------------------
    # Search closest vector and retrieve it from db
    names_og, distances, _, _, _ = database.search(query_im, args.extractor, nrt_neigh=10)
    top1_result = names_og[0]
    """index = faiss.read_index(args.db_name)
    r = redis.Redis(host='localhost', port='6379', db=0)
    id_top1 = int(r.get(str(top1_result)).decode('utf-8'))
    vector = index.index.reconstruct(id_top1)
    vector = vector.reshape(1, nb_features)"""


    # prepare the result
    result_im = Image.open(top1_result).convert('RGB')
    result_im = feat_extract(result_im)
    vector = model.get_vector(result_im)
    cl_result = utils.get_class(top1_result)

   
    explainer = SBSM(model, input_size=(224, 224))

    maskspath = 'masks.npy'
    explainer.generate_masks(window_size=24, stride=5, savepath=maskspath)
    explainer.to(device)
                      
    salmaps = explainer(query_im, result_im)
    # convert to numpy
    salmaps = salmaps.permute(1, 2, 0).cpu().numpy()

    print(salmaps.shape)
    # Display the image and the saliency map on one 
    plt.imshow(query_im.permute(1, 2, 0).cpu().numpy())
    #plt.imshow(salmaps, cmap='jet', alpha=0.5)
    plt.imshow(np.flipud(np.fliplr(salmaps)), cmap='jet', alpha=0.5)

    plt.show()
    # save the image
    plt.savefig("test" + ".png")

if __name__ == '__main__':
    #args = parse_args()
    main()
