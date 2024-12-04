import random
import time
import models
import os
import utils

import matplotlib.pyplot as plt

from db import Database
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms

def list_of_strings(string):
    return string.split(',')

def list_of_ints(string):
    return [int(i) for i in string.split(',')]

# File to retrieve images from the database
class ImageRetriever:

    def __init__(self, db_name, model, load=False):
        self.db = Database(db_name, model, load)

    def retrieve(self, image, extractor, nrt_neigh=10):
        return self.db.search(image,extractor,nrt_neigh= nrt_neigh)

def all_models(args ):
    args.path = args.path[0]
    class_names = []
    names_only = []
    names = []
    class_names = []

    plt.figure()
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.axis('off')
    plt.savefig(os.path.join(args.results_dir, "1_query_image.png"))
    for i in range(len(args.extractor)):
        extractor = args.extractor[i]
        weights = args.weights[i]
        num_features = args.num_features[i]
        print(num_features)

        model = models.Model(model=extractor, num_features=num_features, 
                         weight=weights, device=device)
        retriever = ImageRetriever(args.db_name, model)

        # Indexed the images in the database
        t = time.time()
        retriever.db.add_dataset(args.path_index, extractor)
        print("T_indexing = "+str(time.time() - t))

        # Retrieve the most similar images
        ret_values = retriever.retrieve(feat_extract(Image.open(args.path).convert('RGB')), extractor, 1)
        dir = args.results_dir

        name = ret_values[0][0]
        dist = ret_values[1]

        names_only = []

        class_names.append(utils.get_class(name))
        names_only.append(name[name.rfind('/')+1:])
        names.append(name)

        plt.figure()
        plt.imshow(Image.open(name).convert('RGB'))
        plt.axis('off')
        plt.savefig(os.path.join(dir, utils.get_class(name)+"_"+extractor+"_"+str(i)+".png"))
        
    # Subplot of the image and the nearest images
    plt.figure(figsize=(7,4))
    plt.subplot(3,6,1)
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.axis('off')
    for i in range(2,17):
        class_name = class_names[i-2]
        plt.subplot(3,6,i)
        plt.imshow(Image.open(names[i-2]).convert('RGB'))
        plt.title(class_name+"\n"+args.extractor[i-2],fontsize=4)

        plt.axis('off')
    plt.savefig(os.path.join(dir, "3_nearest_images.png"))

def oneQueryOneModel(args):
    args.path = args.path[0]
    model = models.Model(model=args.extractor, num_features=args.num_features, 
                         weight=args.weights, device=device)
    
            
    retriever = ImageRetriever(args.db_name, model, True)

    # Retrieve the most similar images
    ret_values = retriever.retrieve(feat_extract(Image.open(args.path).convert('RGB')), args.extractor, args.nrt_neigh)
    dir = args.results_dir
    print(ret_values)
    names = ret_values[0]
    print(names)
    dist = ret_values[1]

    names_only = []
    class_names = []
    classement = 0 
    Image.open(args.path).convert('RGB').save(os.path.join(dir, "3_query_image.png"))

    # Retrieve the class of each retrieved image
    for n in names:
        classement += 1
        class_names.append(utils.get_class(n))
        names_only.append(n[n.rfind('/')+1:])
        img = Image.open(n)
        img.save(os.path.join(dir,str(classement) + "_" + utils.get_class(n)+'_'+n[n.rfind('/')+1:] ))
    
    # Subplot of the image and the nearest images
    plt.figure(figsize=(7,3))
    plt.subplot(2,6,1)
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.axis('off')
    for i in range(2,12):
        #class_name = utils.get_new_name(class_names[i-2]) # to use if uliege dataset 
        #class_name = class_names[i-2]
        plt.subplot(2,6,i)
        plt.imshow(Image.open(names[i-2]).convert('RGB'))
        # Write the distance and rank right below each of the image
        #plt.text(2, 45, str(dist[i-2])+ str(i-1), fontsize=8)
        # Change title font size
        #plt.title(class_name,fontsize=8)

        plt.axis('off')
    plt.savefig(os.path.join(dir, "3_nearest_images.png"))
    
    plt.subplot(1,2,1)
    plt.imshow(Image.open(args.path).convert('RGB'))
    plt.title("Query image", fontsize=8)
    plt.subplot(1,2,2)
    plt.imshow(Image.open(names[0]).convert('RGB'))
    plt.title("Nearest image", fontsize=8)
    plt.savefig(os.path.join(dir, "3_nearest_image.png"))
    print("The names of the nearest images are: "+str(class_names))

def mulQueriesMulModels(args):
    names_tot = []
    class_names_tot = []

    q_classes = []
    for path in args.path:
        plt.figure()
        plt.imshow(Image.open(path).convert('RGB'))
        plt.axis('off')
        plt.savefig(os.path.join(args.results_dir, "query_"+utils.get_class(path)+".png"))
        q_classes.append(utils.get_class(path))
    for i in range(len(args.extractor)):
        extractor = args.extractor[i]
        weights = args.weights[i]
        num_features = args.num_features[i]

        model = models.Model(model=extractor, num_features=num_features, 
                         weight=weights, device=device)
        retriever = ImageRetriever(args.db_name, model)

        # Indexed the images in the database
        t = time.time()
        retriever.db.add_dataset(args.path_index, extractor)
        print("T_indexing = "+str(time.time() - t))

        class_names = []
        names = []
        for j in range(len(args.path)):
            # Retrieve the most similar images
            ret_values = retriever.retrieve(feat_extract(Image.open(args.path[j]).convert('RGB')), extractor, 1)
            dir = args.results_dir

            name = ret_values[0][0]
            class_names.append(utils.get_class(name))
            names.append(name)

            plt.figure()
            plt.imshow(Image.open(name).convert('RGB'))
            plt.axis('off')
            plt.savefig(os.path.join(dir, q_classes[j]+"_"+utils.get_class(name)+"_"+extractor+"_"+str(i)+".png"))
        names_tot.append(names)
        class_names_tot.append(class_names)

    for j in range(len(args.path)):
        # Subplot of the image and the nearest images
        plt.figure(figsize=(7,4))
        plt.subplot(3,6,1)
        plt.imshow(Image.open(args.path[j]).convert('RGB'))
        plt.title("Query image", fontsize=8)
        plt.axis('off')
        for i in range(2,17):
            class_name = class_names_tot[i-2][j]
            plt.subplot(3,6,i)
            plt.imshow(Image.open(names_tot[i-2][j]).convert('RGB'))
            plt.title(class_name+"\n"+args.extractor[i-2],fontsize=4)

            plt.axis('off')
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--path',
        help='path to the image used as query. None if drawn at random',
        type=list_of_strings,
        default = "",
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used. Can be a list',
        type=list_of_strings,
        default='resnet'
    )

    parser.add_argument(
        '--db_name',
        help='name of the database',
        default='db'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract. Must match the number of extractors',
        default=[128],
        type=list_of_ints
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network. Must match the number of extractors',
        default='weights',
        type=list_of_strings
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--nrt_neigh',
        default=10,
        type=int
    )
    parser.add_argument(
        '--results_dir'
    )

    parser.add_argument(
        '--path_index',
        help='path to the folder that contains the images to add',
        default = "",
    )
    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'


    feat_extract = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
    

    if len(args.path)== 1 and os.path.isfile(args.path[0]): # one single image as query 
        if len(args.extractor) == 1:
            args.extractor = args.extractor[0]
            args.num_features = args.num_features[0]
            args.weights = args.weights[0]
            oneQueryOneModel(args)
        else:
            all_models(args) 
    elif len(args.path) >1:
        mulQueriesMulModels(args)
            
    else:
        print("all okay")
        model = models.Model(model=args.extractor[0], num_features=args.num_features[0], 
                         weight=args.weights, device=device)
    
        args.extractor = args.extractor[0]
        args.num_features = args.num_features[0]
        retriever = ImageRetriever(args.db_name, model)
        # Indexed the images in the database
        t = time.time()
        retriever.db.add_dataset(args.path_index, args.extractor)
        print("T_indexing = "+str(time.time() - t))
        # Queries for the report 
        #list_im = ['/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_4/0212.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_06_MUCOSA/02_010_1109C_Row_1_Col_151.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4720/2288_8122631.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd2_chimio_necrose_36362044/36360773_94317504.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_3/0273.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk6_0/9076_idx5_x1801_y51_class0.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/mitos2014_1/H04_113660152_2280_2023_323_323_0.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/warwick_crc_0/116153472_400_400_100_100.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_02_STROMA/02_011_10AEA_Row_1_Col_751.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/lbpstroma_113349434/114114866_0_60570_Oncotex_str_valid_0.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/13687_116863030_875_1625_250_250.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_485565/2288_8122394.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk5_0/03_117258525_0_1750_250_250.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/iciar18_micro_113351588/103_118328229_0_512_512_512.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/mitos2014_2/H12_113598371_4845_1292_323_323.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4715/58215_926216.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/lbpstroma_113349448/114115182_1_60414_Oncotex_str_valid.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_5/0600.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4768/710601_797946.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4712/58215_69961.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/iciar18_micro_113351608/107_118327798_1536_0_512_512.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/cells_no_aug_1/728783_776327.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_01_TUMOR/02_003_9B4B_Row_301_Col_301.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_672444/1916_2190056.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4763/76162_79587.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk7_113347715/114400491_384_384_384_384.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/cells_no_aug_0/720965_8010562.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_0/0520.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_1/0611.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/glomeruli_no_aug_1/2_21236_12295222.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk7_113347673/114373689_384_0_384_384.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_08_EMPTY/10_026_10C6_Row_3301_Col_601.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_406558/786011_799757.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_04_LYMPHO/02_001_17A3C_Row_1_Col_301.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/iciar18_micro_113351628/280_118309298_512_0_512_512.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4762/76162_78909.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_05_DEBRIS/02_020_126E5_Row_1_Col_1.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk6_1/15473_idx5_x951_y1951_class1.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4766/710601_778075.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4765/785913_802563.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/glomeruli_no_aug_0/9_29049_13654319.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/tupac_mitosis_0/11123619_1000_750_250_250.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk7_113347697/114392253_384_0_384_384.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd2_chimio_necrose_36362022/36360773_94320907.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/warwick_crc_1/116272667_200_100_100_100.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/camelyon16_1/27704331_21504_119040_768_768.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk1_2/12875_116857417_1250_375_250_250.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/patterns_no_aug_0/8124013_18092382.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/tupac_mitosis_1/11134154_577_973_250_250_4.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_7/0185.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4713/1916_71503.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4764/710601_762046.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_2/0307.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_03_COMPLEX/02_006_4F37_Row_451_Col_1.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_lbtd_lba_4767/786011_799531.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk2_1/10278_116933062_100_300_200_200.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_68567/2288_932461.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/patterns_no_aug_1/728066_790631.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4711/58219_61780.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulg_bonemarrow_6/0516.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk5_1/07_117248119_257_37_250_250_7.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/ulb_anapath_lba_4714/58215_926058.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/iciar18_micro_113351562/243_118314352_0_512_512_512.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/janowczyk2_0/12626_116925376_800_800_200_200.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/mitos2014_0/H15_113547969_4199_2584_323_323.png', '/home/labsig/Documents/Axelle/cytomine/Data/validation/umcm_colorectal_07_ADIPOSE/03_012_10B34_Row_301_Col_1501.tif', '/home/labsig/Documents/Axelle/cytomine/Data/validation/camelyon16_0/27710864_50688_31488_768_768.png']
        #list_im = []
        #list_im = ['/home/axelle/Documents/Doctorat/WP1/data/uliege/val/camelyon16_0/27665488_9984_129024_768_768.png',
        #        '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/ulg_lbtd2_chimio_necrose_36362044/36360773_66312823.png',
        #        '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/patterns_no_aug_0/8124013_18091437.png' ,
        #        '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/janowczyk6_1/8867_idx5_x501_y1051_class1.png',
        #        '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/mitos2014_1/A10_113799388_52_1503_323_323_0.png' ,
        #        '/home/axelle/Documents/Doctorat/WP1/data/uliege/val/tupac_mitosis_1/11104704_1371_589_250_250_5.png']
        #list_im = ['/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/BACK/BACK-TCGA-ADESWYYD.tif',
        #    '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/LYM/LYM-TCGA-AAWGSCHH.tif' ,
        #    '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/MUS/MUS-TCGA-AASRLCCT.tif',
        #    '/home/axelle/Documents/Doctorat/WP1/data/CrC/CRC-VAL-HE-7K/STR/STR-TCGA-AHKNISSH.tif']
        list_im = ['/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/normal/patch_patient_020_node_2_x_5600_y_16288.png',
            '/home/axelle/Documents/Doctorat/WP1/data/Camelyon17/cm17_changed/validation/tumor/patch_patient_020_node_2_x_704_y_15232.png']
        #for c in os.listdir(os.path.join(args.path)):
        for i in range(len(list_im)):
            print(i)
            # takes one image at random per class
            #print(c)
            #img = random.choice(os.listdir(os.path.join(args.path,c)))
            #img = os.path.join(args.path,c,img)
            #list_im.append(img)
            img = list_im[i]
            c = utils.get_class(img)
            ret_value = retriever.retrieve(feat_extract(Image.open(img).convert('RGB')), args.extractor, 1)
            dir = args.results_dir
            print(ret_value)
            name = ret_value[0][0]
            dist = ret_value[1][0]

            
            # Retrieve the class of the retrieved image
            names_only = name[name.rfind('/')+1:]
            class_name = utils.get_class(name)
            
            ret_img = Image.open(name)
            ret_img.save(os.path.join(dir, '1_' + c + "_"+ class_name +".png"))

            plt.figure()
            plt.imshow(Image.open(name).convert('RGB'))
            plt.axis('off')
            plt.savefig(os.path.join(dir, utils.get_class(name)+"_"+"uni"+".png"))

            plt.subplot(1,2,1)
            plt.imshow(Image.open(img).convert('RGB'))
            plt.title("Query image", fontsize=8)
            plt.subplot(1,2,2)
            plt.imshow(ret_img)
            plt.title("Nearest image", fontsize=8)
            plt.savefig(os.path.join(dir,  c+ "_"+"nearest_image.png"))
            


            
