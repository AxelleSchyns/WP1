from matplotlib import pyplot as plt
import redis
import torch
import db
import faiss
import time
import models
import utils

import torch.nn.functional as F

from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np

nb_masks = 10000

def generate_masks_torch(N, s, p1, input_size, device):
    H, W = input_size
    
    cell_size = torch.ceil(torch.tensor([H / s, W / s]))
    up_size = ((s + 1) * cell_size).int()  # (up_H, up_W)
    print("cell_size = ", cell_size)
    print("up_size = ", up_size)

    # Step 1: Create binary grid
    grid = (torch.rand(N, 1, s, s) < p1).float().to(device)

    # Step 2: Upsample
    upsampled = F.interpolate(grid, size=tuple(up_size.tolist()), mode='bilinear', align_corners=False)

    # Step 3: Random crop
    masks = torch.empty((N, 1, H, W), device=device)
    for i in tqdm(range(N), desc='Generating masks'):
        x = torch.randint(0, int(cell_size[0]), (1,)).item()
        y = torch.randint(0, int(cell_size[1]), (1,)).item()
        masks[i] = upsampled[i, :, x:x + H, y:y + W]

    return masks  # shape: (N, 1, H, W)

if __name__ == "__main__":

    #1 Parse the arguments
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
    # Load the feature extractor 
    model = models.Model(model=args.extractor, weight=args.weight, device ='cuda:0')
    nb_features = model.num_features
    if model is None:
        exit(-1)
    if args.indexing: 
        # Initialize the database
        print("database starting")
        database = db.Database(args.db_name, model, load = False)
        # Indexed the images in the database
        t = time.time()
        database.add_dataset(args.path_indexing, "resnet")
        print("T_indexing = "+str(time.time() - t))

    else:
        # Load the database
        print("Loading the database")
        database = db.Database(args.db_name, model, load = True)

    # 1. Retrieve the result corresponding to the query
    img = Image.open(args.query).convert('RGB')
    feat_extract = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
    img = feat_extract(img)
    names, distances, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(img, args.extractor, nrt_neigh=1)
    # retrieve the feature vector correpsonding to top_1 result
    top1_result = names[0]
    index = faiss.read_index(args.db_name)
    r = redis.Redis(host='localhost', port='6379', db=0)
    id_top1 = int(r.get(str(top1_result)).decode('utf-8'))
    # Retrieve the vectors from the index
    vector = index.index.reconstruct(id_top1)

    """ Checking of correct distance used
    out = img.to(device="cuda:0", non_blocking=True).reshape(-1, 3, 224, 224)
    out = model(out)
    print(out.shape)
    print(faiss.pairwise_distances(out.cpu().detach().numpy(), vector.reshape(1, 128)))"""

    masks = generate_masks_torch(nb_masks, 7, 0.5, (224, 224), device="cuda:0") #values from article
    saliency_map = torch.zeros((224, 224)) # Initialize the saliency map
    #2 Apply RISE  
    img = img.unsqueeze(0).cuda()
    for i in range(nb_masks):
        print(i)
        masked_query = img * masks[i] # mask value 0 = pixels erased
        masked_vec = model(masked_query)[0]
        masked_vec = masked_vec.unsqueeze(0)
        vector = vector.reshape(1, nb_features)
        # Compute similarity with the vector of top1 results
        sim_score =  faiss.pairwise_distances(masked_vec.cpu().detach().numpy(), vector)
        # Store the results - the difference between original similarity score and new one is placed in the saliency map for each pixel not masked, normalized by the original distance
        saliency_map += (1 - masks[i].squeeze(0).cpu()) * (abs((distances[0] - sim_score)) / distances[0])
    
    E_p =  0.5
    saliency_map = saliency_map / nb_masks / E_p
    

    cl = utils.get_class(args.query)
    #3 Save the saliency map
    saliency_map = saliency_map.cpu().numpy()
    # save as image and display
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255
    saliency_map = saliency_map.astype('uint8')

    saliency_map = Image.fromarray(saliency_map)
    saliency_map.save('sal_'+args.extractor+'_'+cl+'.png')
    saliency_map.show()

    plt.imshow(np.array(img.squeeze(0).cpu().permute(1, 2, 0)))
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    # save the image
    plt.savefig(args.extractor+'_'+cl+'.png')
    
