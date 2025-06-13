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
result_folder = "./results/explainability/"
score_match = {"distance":0, "rank_top":1, "rank_all":2, "change":3}


def generate_masks_torch(N, s, p1, input_size, device):
    H, W = input_size
    
    cell_size = torch.ceil(torch.tensor([H / s, W / s]))
    up_size = ((s + 1) * cell_size).int()  # (up_H, up_W)

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

def parse():
    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--weight',
        help='path to the folder that contains the weights',
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

    # Can be a list of string
    parser.add_argument(
        '--score',
        help='score to use for the explanation',
        default='distance,rank_top'
    )
    args = parser.parse_args()
    return args

def compute_scores(i, exp, args, scores, masked_img, top1_result, masks, maps, names_og, distances_og, vec_list, database, model): 
    if exp == 0: 
        if 'rank_top' or 'rank_all' or 'change' in scores:
            names, distances,_, _, _ = database.search(masked_img, args.extractor, nrt_neigh=50)
            if 'rank_top' in scores:
                rank_top = 100
                for l in range(len(names)):
                    if names[l] == top1_result:
                        rank_top = l
                        break
                id_rt = scores.index('rank_top')
                maps[id_rt] += (1 - masks[i].squeeze(0).cpu()) * (rank_top-1)
            if 'change' in scores:
                changes = 0
                for l in range(10):
                    if names[l] not in names_og:
                        changes += 1
                changes = changes/10
                id_ch = scores.index('change')
                maps[id_ch] += (1 - masks[i].squeeze(0).cpu()) * changes
            if 'rank_all' in scores: 
                score = 0
                for l in range(10):
                    try:
                        temp_rank = names.index(names_og[l])
                    except:
                        temp_rank = 100
                    diff = abs(temp_rank - l)
                    score += (10-l)*diff
                score = score/10
                id_ra = scores.index('rank_all')
                maps[id_ra] += (1 - masks[i].squeeze(0).cpu()) * score
    if 'distance' in scores:
        # Get the vector of the masked image
        masked_vec = model.get_vector(masked_img)
        sim_score =  faiss.pairwise_distances(masked_vec.cpu().detach().numpy(), vec_list[exp]) 
        if exp == 1:
            id_d = 0
        else:
            id_d = scores.index('distance')
        maps[id_d] += (1 - masks[i].squeeze(0).cpu()) * (abs((distances_og[0] - sim_score)) / distances_og[0])
    return maps

def save_ind_map(maps, scores, exp, args, og_im_list, cl_list, status, saliency_maps):
    new_scores = scores if exp == 0 else ['distance']
    for i, score in enumerate(new_scores): 
        score_id = score_match[score]
        folder_sal = result_folder+'/score_'+str(score_id)+'/saliency/'
        folder_sup = result_folder+'/score_'+str(score_id)+'/superposition/'
        temp_path = status+'_'+args.extractor+'_'+cl_list[exp]+'.png'


        saliency_map = maps[i]
        # Divide by the expectation
        E_p =  0.5
        saliency_map = saliency_map / nb_masks / E_p
        

        
        # Normalize and save saliency map
        saliency_map = saliency_map.cpu().numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255
        saliency_map = saliency_map.astype('uint8')
        saliency_map = Image.fromarray(saliency_map)

        saliency_map.save(folder_sal+temp_path)

        # Display the image and the saliency map on one 
        plt.imshow(og_im_list[exp])
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        # save the image
        plt.savefig(folder_sup+temp_path)

        saliency_maps.append(saliency_map)

    return saliency_maps

def save_comp_map(scores, og_im_list):
    for i, score in enumerate(scores): 
        score_id = score_match[score]
        folder_comp = result_folder+'/score_'+str(score_id)+'/comparison/'+args.extractor+'_'+cl_list[0]+'.png'
        if score != "distance":
            # Display in one image the query, the top 1 result and both saliency maps
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 2, 1)
            plt.imshow(og_im_list[0])
            plt.title('Query')
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(og_im_list[1])
            plt.title('Result')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(saliency_maps[i])
            plt.title('Saliency Map Query')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(og_im_list[0])
            plt.imshow(saliency_maps[i], cmap='jet', alpha=0.5)
            plt.title('Superposition for Query')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(folder_comp)
        else:
            # Query 
            plt.figure(figsize=(10, 5))
            plt.subplot(3, 2, 1)
            plt.imshow(og_im_list[0])
            plt.title('Query')
            plt.axis('off')

            # Top1 result 
            plt.subplot(3, 2, 2)
            plt.imshow(og_im_list[1])
            plt.title('Result')
            plt.axis('off')

            # Sal map of query 
            plt.subplot(3, 2, 3)
            plt.imshow(saliency_maps[i])
            plt.title('Saliency Map Query')
            plt.axis('off')

            # Sal map of result
            plt.subplot(3, 2, 4)
            plt.imshow(saliency_maps[len(scores)-1])
            plt.title('Saliency Map Result')
            plt.axis('off')

            # Superposition for query
            plt.subplot(3, 2, 5)
            plt.imshow(og_im_list[0])
            plt.imshow(saliency_maps[i], cmap='jet', alpha=0.5)
            plt.title('Superposition for Query')
            plt.axis('off')

            # Superposition for result
            plt.subplot(3, 2, 6)
            plt.imshow(og_im_list[1])
            plt.imshow(saliency_maps[len(scores)-1], cmap='jet', alpha=0.5)
            plt.title('Superposition for Result')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(folder_comp)

    nb_row = int(np.ceil(len(scores) /2 + 1))

    # Display in one image the query, the top 1 result and saliency maps for each score
    plt.figure(figsize=(5, 5))
    plt.subplot(nb_row, 2, 1)
    plt.imshow(Image.open(args.query).convert('RGB'))
    plt.title('Query')
    plt.axis('off')
    plt.subplot(nb_row, 2, 2)
    plt.imshow(Image.open(top1_result).convert('RGB'))
    plt.title('Result')
    plt.axis('off')

    for i, score in enumerate(scores):
        id_map = score_match[score]
        plt.subplot(nb_row, 2, i + 3)
        plt.imshow(og_im_list[0])
        plt.imshow(saliency_maps[id_map], cmap='jet', alpha=0.5)
        plt.title(score + ' - Query')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(result_folder+args.extractor+'_'+cl_list[0]+'_all_scores'+'.png') 
   
if __name__ == "__main__":

    #1 Parse the arguments
    args = parse()

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

    # Load args
    scores = args.score.split(',')
    scores.sort()

    # ----------------------------------------------------------------------------------------------------------
    #                                     2. Find the closest vector
    # ----------------------------------------------------------------------------------------------------------
    # Search closest vector and retrieve it from db
    names_og, distances_og, _, _, _ = database.search(query_im, args.extractor, nrt_neigh=10)
    top1_result = names_og[0]
    index = faiss.read_index(args.db_name)
    r = redis.Redis(host='localhost', port='6379', db=0)
    id_top1 = int(r.get(str(top1_result)).decode('utf-8'))
    vector = index.index.reconstruct(id_top1)
    vector = vector.reshape(1, nb_features)


    # prepare the result
    result_im = Image.open(top1_result).convert('RGB')
    result_im = feat_extract(result_im)
    cl_result = utils.get_class(top1_result)

    # ----------------------------------------------------------------------------------------------------------
    #                                     3. Apply RISE
    # ----------------------------------------------------------------------------------------------------------
    im_list = [query_im, result_im] # tensors - transforms applied 
    og_im_list = [Image.open(args.query).convert('RGB').resize((224,224)), Image.open(top1_result).convert('RGB').resize((224,224))]
    cl_list = [cl_query, cl_result]
    vec_list = [vector, vec_query.cpu().detach().numpy()]
    
    saliency_maps=[] 
    # Run on the query then on the result 
    for exp in range(len(im_list)):
        maps = []
        if exp == 0:
            for nb_score in range(len(scores)):
                maps.append(torch.zeros((224, 224))) # Initialize the saliency maps for each score
        else:
            if "distance" not in scores:
                break 
            else:
                maps.append(torch.zeros((224, 224))) 

        status = 'Query' if exp == 0 else 'Result'
        img = im_list[exp]

        # Generates the masks 
        masks = generate_masks_torch(nb_masks, 7, 0.5, (224, 224), device="cuda:0") #values from article7
        
        # Apply RISE  
        img = img.unsqueeze(0).cuda()
        for i in range(nb_masks):
            if i % 2000==0:
                print(i)
            
            masked_img = img * masks[i] # mask value 0 = pixels erased

            maps = compute_scores(i, exp, args, scores, masked_img, top1_result, masks, maps, names_og, distances_og, vec_list, database, model)
        
            
        # Save the map
        saliency_maps = save_ind_map(maps, scores, exp, args, og_im_list, cl_list, status, saliency_maps)
    save_comp_map(scores, og_im_list)