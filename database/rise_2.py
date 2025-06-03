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

nb_masks = 8000
result_folder = "./results/explainability/"


def generate_masks_torch(N, s, p1, input_size, device):
    H, W = input_size
    
    cell_size = torch.ceil(torch.tensor([H / s, W / s]))
    up_size = ((s + 1) * cell_size).int()  # (up_H, up_W)
    #print("cell_size = ", cell_size)
    #print("up_size = ", up_size)

    # Step 1: Create binary grid
    grid = (torch.rand(N, 1, s, s) < p1).float().to(device)
    """
    grid_dis = Image.fromarray((grid[0].squeeze(0).cpu().numpy() * 255).astype('uint8'))
    grid_dis.save('grid.png')
    exit()"""
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
    names_og, distances, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(query_im, args.extractor, nrt_neigh=10)
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
    im_list = [query_im, result_im]
    og_im_list = [Image.open(args.query).convert('RGB').resize((224,224)), Image.open(top1_result).convert('RGB').resize((224,224))]
    cl_list = [cl_query, cl_result]
    vec_list = [vector, vec_query.cpu().detach().numpy()]
    saliency_maps = []
    maps = []
    for j in range(len(im_list)):
        status = 'Query' if j == 0 else 'Result'
        img = im_list[j]

        # Generates the masks 
        masks = generate_masks_torch(nb_masks, 7, 0.5, (224, 224), device="cuda:0") #values from article
        saliency_map_1 = torch.zeros((224, 224)) # Initialize the saliency map for the change of rank of top1
        saliency_map_2 = torch.zeros((224, 224)) # Initialize the saliency map for percentage of changes in top10 results
        saliency_map_3 = torch.zeros((224, 224))
        maps = []
        
        # Apply RISE  
        img = img.unsqueeze(0).cuda()
        for i in range(nb_masks):
            if i % 2000==0:
                print(i)
            
            masked_query = img * masks[i] # mask value 0 = pixels erased
            names, distances,_, _, _ = database.search(masked_query, args.extractor, nrt_neigh=50)
            rank_top = 100
            # first score
            for l in range(len(names)):
                if names[l] == top1_result:
                    rank_top = l
                    break
            changes = 0
            for l in range(10):
                if names[l] not in names_og:
                    changes += 1
            changes = changes/10

            score = 0
            for l in range(10):
                try:
                    temp_rank = names.index(names_og[l])
                except:
                    temp_rank = 100
                diff = abs(temp_rank - l)
                score += (10-l)*diff
            score = score/10
    
            masked_vec = model.get_vector(masked_query)
            # Compute confidence score
            sim_score =  faiss.pairwise_distances(masked_vec.cpu().detach().numpy(), vec_list[j]) 
            """print(abs((distances[0] - sim_score)))
            print("rank_top = ", rank_top)
            print("changes = ", changes)
            print("score = ", score)"""
            saliency_map_1 += (1 - masks[i].squeeze(0).cpu()) * (rank_top-1)
            saliency_map_2 += (1 - masks[i].squeeze(0).cpu()) * changes
            saliency_map_3 += (1 - masks[i].squeeze(0).cpu()) * score
        
        score_id = 1
        folders_sal = [result_folder+'/score_1/saliency/', result_folder+'/score_2/saliency/',
                   result_folder+'/score_3/saliency/']
        folders_sup = [result_folder+'/score_1/superposition/', result_folder+'/score_2/superposition/',
                   result_folder+'/score_3/superposition/']
        maps.append(saliency_map_1)
        maps.append(saliency_map_2)
        maps.append(saliency_map_3)
        for saliency_map in maps:
            temp_path = status+'_'+args.extractor+'_'+cl_list[j]+'_'+str(score_id)+'.png'
            # Divide by the expectation
            E_p =  0.5
            saliency_map = saliency_map / nb_masks / E_p
            

            #- ----------------------------------------------------------------------------------------------------------
            #                                    4. Save and display the results
            # ----------------------------------------------------------------------------------------------------------
            # Normalize and save saliency map
            saliency_map = saliency_map.cpu().numpy()
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255
            saliency_map = saliency_map.astype('uint8')
            saliency_map = Image.fromarray(saliency_map)
            saliency_map.save(folders_sal[score_id-1]+temp_path)
            saliency_maps.append(saliency_map)

            # Display the image and the saliency map on one 
            plt.imshow(og_im_list[j])
            plt.imshow(saliency_map, cmap='jet', alpha=0.5)
            # save the image
            plt.savefig(folders_sup[score_id-1]+temp_path)
            score_id += 1

    folders_comp = [result_folder+'/score_1/', result_folder+'/score_2/',
                   result_folder+'/score_3/']
    for l in range(3):
        temp_path = 'comparison/'+args.extractor+'_'+cl_list[0]+'_'+str(l+1)+'.png'
        # Display in one image the query, the top 1 result and both saliency maps
        plt.figure(figsize=(10, 5))
        plt.subplot(3, 2, 1)
        plt.imshow(Image.open(args.query).convert('RGB'))
        plt.title('Query')
        plt.axis('off')
        plt.subplot(3, 2, 2)
        plt.imshow(Image.open(top1_result).convert('RGB'))
        plt.title('Result')
        plt.axis('off')
        plt.subplot(3, 2, 3)
        plt.imshow(saliency_maps[0+l])
        plt.title('Saliency Map Query')
        plt.axis('off')
        plt.subplot(3, 2, 4)
        plt.imshow(saliency_maps[3+l])
        plt.title('Saliency Map Result')
        plt.axis('off')
        plt.subplot(3, 2, 5)
        plt.imshow(og_im_list[0])
        plt.imshow(saliency_maps[0+l], cmap='jet', alpha=0.5)
        plt.title('Superposition for Query')
        plt.axis('off')
        plt.subplot(3, 2, 6)
        plt.imshow(og_im_list[1])
        plt.imshow(saliency_maps[3+l], cmap='jet', alpha=0.5)
        plt.title('Superposition for Result')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(folders_comp[l]+temp_path)

    
    # Display in one image the query, the top 1 result and saliency maps from each score
    plt.figure(figsize=(5, 5))
    plt.subplot(4, 2, 1)
    plt.imshow(Image.open(args.query).convert('RGB'))
    plt.title('Query')
    plt.axis('off')
    plt.subplot(4, 2, 2)
    plt.imshow(Image.open(top1_result).convert('RGB'))
    plt.title('Result')
    plt.axis('off')
    plt.subplot(4, 2, 3)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[0], cmap='jet', alpha=0.5)
    plt.title('Score 1 - Query')
    plt.axis('off')
    plt.subplot(4, 2, 4)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[3], cmap='jet', alpha=0.5)
    plt.title('Score 1 - Result')
    plt.axis('off')
    plt.subplot(4, 2, 5)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[1], cmap='jet', alpha=0.5)
    plt.title('Score 2 - Query')
    plt.axis('off')
    plt.subplot(4, 2, 6)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[4], cmap='jet', alpha=0.5)
    plt.title('Score 2 - Result')
    plt.axis('off')
    plt.subplot(4, 2, 7)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[2], cmap='jet', alpha=0.5)
    plt.title('Score 3 - Query')
    plt.axis('off')
    plt.subplot(4, 2, 8)
    plt.imshow(og_im_list[0])
    plt.imshow(saliency_maps[5], cmap='jet', alpha=0.5)
    plt.title('Score 3 - Result')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_folder+args.extractor+'_'+cl_list[0]+'_all_scores'+'.png')
