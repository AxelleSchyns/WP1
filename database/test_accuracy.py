import torch
import db
import os
import sklearn
import time
import sklearn.metrics
import utils

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from models import Model
from PIL import Image
from argparse import ArgumentParser
from collections import Counter, defaultdict
from torch.utils.data import Dataset
from torchvision import transforms



# This function computes the results given the search results and the query
# - names = list of the most similar images in the database
# - data = dataset object containing the data
# - predictions = complete list of the label of the top-1 result for each query
# - class_im = class of the query
# - proj_im = project of the query
# - top_1_acc = list containing the top-1 accuracy at current
# - top_5_acc = list containing the top-5 accuracy at current
# - maj_acc = list containing the maj accuracy at current
# - predictions_maj = complete list of the label of the maj result for each query
# - weights = list containing the weights of each class for the weighted protocol
def compute_results(names, data, predictions, class_im, proj_im, top_1_acc, top_5_acc, maj_acc, predictions_maj, weights):
    similar = names[:5]
    temp = []
    already_found_5 = 0
    already_found_5_proj = 0
    already_found_5_sim = 0
    prop = []
    ev_prop = []
    if len(data.classes) == 1:
        idx_class = 0
    else:
        idx_class = data.conversion[class_im]

    # Proportion of correct image at each step independantly 
    for i in range(len(names)):
        class_retr = utils.get_class(names[i])
        if class_retr == class_im:  
            prop.append(1) 
        else:
            prop.append(0)

    # Evolution of proportion of correct image at each step independantly 
    for i in range(len(names)): # additive prop 
        if i == 0:
            ev_prop.append(prop[i])
        else:
            ev_prop.append( (prop[i] + (ev_prop[i-1] * i)) / (i+1) )
        
    for j in range(len(similar)):
        # Gets the class and project of the retrieved image
        class_retr = utils.get_class(similar[j])
        temp.append(class_retr)
        proj_retr = utils.get_proj(similar[j])
        
        # Retrieves label of top 1 result for confusion matrix
        if j == 0:
            # if class retrieved is in class of data given
            if class_retr in data.conversion:
                predictions.append(data.conversion[class_retr])
            else:
                print(class_retr)
                print(data.conversion)
                print(len(data.conversion))
                predictions.append(len(data.conversion)+1) # to keep a trace of the data for the cm
        
        # Class retrieved is same as query
        if class_retr == class_im: 
            if already_found_5 == 0:
                top_5_acc[0] += weights[idx_class]
                if already_found_5_proj == 0:
                    top_5_acc[1] += weights[idx_class]
                if already_found_5_sim == 0:
                    top_5_acc[2] += weights[idx_class]
        
                if j == 0:
                    top_1_acc[0] += weights[idx_class]
                    if already_found_5_proj == 0:
                        top_1_acc[1] += weights[idx_class]  
                    if already_found_5_sim == 0:
                        top_1_acc[2] += weights[idx_class]
            already_found_5 += 1 # One of the 5best results matches the label of the query ->> no need to check further
            already_found_5_proj += 1
            already_found_5_sim +=1
            
        # Class retrieved is in the same project as query
        elif proj_retr == proj_im:
            if already_found_5_proj == 0:
                top_5_acc[1] += weights[idx_class]
                if already_found_5_sim == 0:
                    top_5_acc[2] += weights[idx_class]
                if j == 0:
                    if already_found_5_sim == 0:
                        top_1_acc[2] += weights[idx_class]
                    top_1_acc[1] += weights[idx_class]
            already_found_5_sim += 1
            already_found_5_proj += 1
        
        
        # Class retrieved is in a project whose content is similar to the query ->> check
        else:       
            # 'janowczyk'
            if proj_im[0:len(proj_im)-2] == proj_retr[0:len(proj_retr)-2]:
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_im == "cells_no_aug" and proj_retr == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_retr == "cells_no_aug" and proj_im == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_retr == "mitos2014" and proj_im == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_im == "mitos2014" and proj_retr == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
        
    if already_found_5 > 2:
        maj_acc[0] += weights[idx_class]
    if already_found_5_proj > 2:
        maj_acc[1] += weights[idx_class]
    if already_found_5_sim > 2:
        maj_acc[2] += weights[idx_class]
    # Get label of the majority class for confusion matrix
    #predictions_maj.append(data.conversion[max(set(temp), key = temp.count)])

    return predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc, prop

        

# Creates and displays the confusion matrix relative to top-1 and maj accuracy 
def display_cm(weight, measure, ground_truth, data, predictions, predictions_maj):
    rows_lab = []
    rows = []
    for el in ground_truth:
        if el not in rows:
            rows.append(el)
            rows_lab.append(list(data.conversion.keys())[el])
    rows = sorted(rows)
    rows_lab = sorted(rows_lab)
    
    # Confusion matrix based on top 1 accuracy
    columns = []
    columns_lab = []
    for el in predictions:
        if el not in columns:
            columns.append(el)
            columns_lab.append(list(data.conversion.keys())[el])
    columns = sorted(columns)
    columns_lab=sorted(columns_lab)
        
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    #plt.show()
    # save the confusion matrix
    fold_path = weight[0:weight.rfind("/")]
    plt.savefig(fold_path + '/uliege_confusion_matrix_top1_'+measure+ '.png')
    # Confusion matrix based on maj_class accuracy:
    columns = []
    columns_lab = []
    for el in predictions_maj:
        if el not in columns:
            columns.append(el)
            columns_lab.append(list(data.conversion.keys())[el])
    columns = sorted(columns)
    columns_lab = sorted(columns_lab)
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions_maj, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes)
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    #plt.show()
    # save the confusion matrix
    plt.savefig(fold_path + '/uliege_confusion_matrix_maj_'+measure+'.png')

def display_precision_recall(weight, measure, ground_truth, predictions):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import PrecisionRecallDisplay

    classes = np.unique(ground_truth)
    Y_test = label_binarize(ground_truth, classes=classes)
    y_score = label_binarize(predictions, classes=classes)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(67):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    from collections import Counter

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
        prevalence_pos_label=Counter(Y_test.ravel())[1] / Y_test.size,
    )
    display.plot(plot_chance_level=True)
    _ = display.ax_.set_title("Micro-averaged over all classes")
    fold_path = weight[0:weight.rfind("/")]
    plt.savefig(fold_path + '/uliege_prec_recall_curve_'+measure+'.png')
    
def display_prec_im(weight, props, data, measure):
    plt.figure()    
    props = props / data.__len__()
    plt.plot(props)
    plt.xlabel('Number of images')
    plt.ylabel('Proportion of correct images')
    fold_path = weight[0:weight.rfind("/")]
    plt.savefig(fold_path + '/uliege_prec_im_'+measure+'.png')


# Compute the metrics per class of the dataset 
# - model = python object containing the feature extractor
# - dataset = path to the dataset from which to get the queries
# - db_name = name of the database in which the data is indexed
# - extractor = name of the feature extractor used
# - measure = protocol for the results (random, remove, all, weighted)
# - name = name to give to the excel sheet in which to write the results
# - excel_path = path to the excel file in which to write the results
def test_each_class(model, dataset, db_name, extractor, measure, name, excel_path):
    classes = sorted(os.listdir(dataset))
    res = np.zeros((len(classes), 13))

    # Compute the results for each class
    i = 0
    for c in classes:
        print("Class: ", c)
        r = test(model, dataset, db_name, extractor, measure, project_name = False, class_name= c, see_cms= False, stat = True)
        res[i][:] = r
        i += 1
    df = pd.DataFrame(res, columns=["top_1_acc", "top_5_acc", "top_1_proj", "top_5_proj", "top_1_sim", "top_5_sim", "maj_acc_class", "maj_acc_proj", "maj_acc_sim", "t_tot", "t_model", "t_search", "t_transfer"])
    df.index = classes


    writer = pd.ExcelWriter(excel_path, engine="openpyxl", mode="a")
    if name is None:
        name = "Sheet ?" 
    df.to_excel(writer, sheet_name = name)
    writer.close()

# Dataset class for the test
class TestDataset(Dataset):
    # - model = python object containing the feature extractor
    # - root = path to the dataset from which to get the queries
    # - measure = protocol for the results (random, remove, separated, all, weighted)
    # - name = name of the project to compute the results of if only one project is wanted
    # - class_name = name of the class to compute the results of if only one class is wanted
    def __init__(self, model, root, measure, class_name =None, name = None):
        
        self.root = root
        self.feat_extract = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
        
        self.dic_img = defaultdict(list)
        self.img_list = []

        self.classes = os.listdir(root)
        self.classes = sorted(self.classes)
        
        self.conversion = {x: i for i, x in enumerate(self.classes)}
        # User has specify the classe whose results he wants to compute
        if class_name is not None:
            
            for c in self.classes:
                if c == class_name:
                    self.classes = [c]
                    break
        # User has specify the project he wants to compute the results of
        elif name is not None:
            start = False
            new_c = []
            for c in self.classes:
                end_class = c.rfind('_')
                if name == c[0:end_class]:
                    start = True
                    new_c.append(c)
                elif start == True:
                    break
            self.classes = new_c
                    
        elif measure == 'remove':
            self.classes.remove('camelyon16_0')
            self.classes.remove('janowczyk6_0')

        # Register images in list and compute weights for weighted protocol
        if measure != 'random':
            if measure == "weighted":
                weights = np.zeros(len(self.classes))
                for i in self.classes:
                    weights[self.conversion[i]] = 1 / len(os.listdir(os.path.join(root, str(i))))
                    for img in os.listdir(os.path.join(root, str(i))):
                        self.img_list.append(os.path.join(root, str(i), img))
                self.weights = weights
            else:
                for i in self.classes:
                    for img in os.listdir(os.path.join(root, str(i))):
                        self.img_list.append(os.path.join(root, str(i), img))
        # Selection of the 1020 images for the random protocol
        else:
            for i in self.classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.dic_img[i].append(os.path.join(root, str(i), img))

            to_delete = []

            while True:
                for key in self.dic_img:
                    if (not self.dic_img[key]) is False:
                        img = np.random.choice(self.dic_img[key]) # Take an image of the class at random
                        self.dic_img[key].remove(img)
                        self.img_list.append(img)
                    else: # no images in the class?
                        to_delete.append(key)

                for key in to_delete:
                    self.dic_img.pop(key, None)

                to_delete.clear()
                if len(self.img_list) > 1000 or len(self.dic_img) == 0:
                    break


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        img = Image.open(self.img_list[idx]).convert('RGB')

        return self.feat_extract(img), self.img_list[idx]


# Main function of the file, calls the different computation functions and display functions given the value of the parameters
# - model = python object containing the feature extractor 
# - dataset = path to the dataset from which to get the queries
# - db_name = name of the database in which the data is indexed
# - extractor = name of the feature extractor used
# - measure = protocol for the results (random, remove, separated, all, weighted)
# - generalise = 0 if no generalisation, 1 -2 for half the classes, 3 if generalisation on kmeans
# - project_name = name of the project to compute the results of if only one project is wanted
# - class_name = name of the class to compute the results of if only one class is wanted
# - see_cms = True if the graphics of the results must be displayed
# - label = if the database to search into is the labelled, unlabelled or mixed one ("True", "False", "mixed")
# - stat = True if the protocol is stat, False otherwise. Controls the display of the results in terminal 
def test(model, model_weight, dataset, db_name, extractor, measure, project_name, class_name, see_cms, stat = False):

    # Load database
    database = db.Database(db_name, model, True)

    # Load data 
    data = TestDataset(model, dataset, measure, project_name, class_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)
    
    # Load weights 
    if measure == 'weighted':
        weights = data.weights
    elif measure == 'remove':
        weights = np.ones(len(data.conversion))
    else:
        weights = np.ones(len(data.classes))
    

    top_1_acc = np.zeros((3,1)) # in order: class - project - similarity 
    top_5_acc = np.zeros((3,1))
    maj_acc = np.zeros((3,1))
    props = np.zeros((10,1))

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []
    predictions_maj = []
    
    t_search = 0
    t_model = 0
    t_transfer = 0
    t_tot = 0
    print(model.num_features)
    # For each image in the dataset, search for the 5 most similar images in the database and compute the accuracy
    for i, (image, filename) in enumerate(loader):
            
        if (i+1) % 1000000 == 0:
            print(i)
        # Search for the 5 most similar images in the database
        t = time.time()
        names, _, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(image, extractor)
        image = filename
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_transfer += t_transfer_tmp
        t_search += t_search_tmp

        # Retrieve class and project of query 
        class_im = utils.get_class(image[0])
        proj_im = utils.get_proj(image[0])
        nbr_per_class[class_im] += 1
        ground_truth.append(data.conversion[class_im])


        # Compute accuracy 
        predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc, prop = compute_results(names, data, predictions, class_im, proj_im, top_1_acc,top_5_acc,maj_acc,predictions_maj, weights)

        for el in range(len(prop)):
            props[el] += prop[el]
    
    if measure == 'weighted':
        f1 = sklearn.metrics.f1_score(ground_truth, predictions, average = "weighted")
    else:
        f1 = sklearn.metrics.f1_score(ground_truth, predictions, average = "micro")

    if measure == 'weighted':
        s = len(data.classes) # In weighted, each result was already divided by the length of the class
    else:
        s = data.__len__()
    
    # Display results in terminal
    if not stat:
        print("top-1 accuracy : ", top_1_acc[0] / s)
        print("top-5 accuracy : ", top_5_acc[0] / s)
        print("top-1 accuracy proj : ", top_1_acc[1] / s)
        print("top-5 accuracy proj : ", top_5_acc[1] / s)
        print("top-1 accuracy sim : ", top_1_acc[2]/ s)
        print("top-5 accuracy sim : ", top_5_acc[2] / s)
        print("maj accuracy class : ", maj_acc[0] / s)
        print("maj accuracy proj : ", maj_acc[1] / s)
        print("maj accuracy sim : ", maj_acc[2] / s)
        print("Props: ", props / s)
        print("f1 score : ", f1)

        print('t_tot:', t_tot)
        print('t_model:', t_model)
        print('t_transfer:', t_transfer)
        print('t_search:', t_search)

    # Display results in graphics
    if see_cms:
        #display_cm(model_weight, measure, ground_truth, data, predictions, predictions_maj)
        #display_precision_recall(model_weight, measure, ground_truth, predictions)
        display_prec_im(model_weight, props, data, measure)
    
    return [top_1_acc[0]/ s, top_5_acc[0]/ s, top_1_acc[1]/ s, top_5_acc[1]/ s, top_1_acc[2]/ s, top_5_acc[2]/ s, maj_acc[0]/ s, maj_acc[1]/ s, maj_acc[2]/ s, t_tot, t_model, t_search, t_transfer]
    
# This function executes the random protocol 50 times for stability of the results
def stat_results(model, model_weight, dataset, db_name, extractor, project_name, class_name, protocol):
    nb_exp = 5

    if protocol == 'random':
        nb_images = 1020
    else:
        nb_images = len(os.listdir(dataset))

    top_1_acc = np.zeros((3, nb_exp))
    top_5_acc = np.zeros((3, nb_exp))
    maj_acc = np.zeros((3, nb_exp))

    ts = np.zeros((4,nb_exp))
    for i in range(nb_exp):
        top_1_acc[0][i], top_5_acc[0][i], top_1_acc[1][i], top_5_acc[1][i], top_1_acc[2][i], top_5_acc[2][i], maj_acc[0][i], maj_acc[1][i], maj_acc[2][i], ts[0][i], ts[1][i], ts[2][i], ts[3][i] =  test(model, model_weight,  dataset, db_name, extractor, protocol, project_name, class_name, False, stat = True)

    print("Top 1 accuracy: ", np.mean(top_1_acc[0]), " +- ", np.std(top_1_acc[0]))
    print("Top 5 accuracy: ", np.mean(top_5_acc[0]), " +- ", np.std(top_5_acc[0]))
    print("Maj accuracy: ", np.mean(maj_acc[0]), " +- ", np.std(maj_acc[0]))
    print("Top 1 accuracy on project: ", np.mean(top_1_acc[1]), " +- ", np.std(top_1_acc[1]))
    print("Top 5 accuracy on project: ", np.mean(top_5_acc[1]), " +- ", np.std(top_5_acc[1]))
    print("Maj accuracy on project: ", np.mean(maj_acc[1]), " +- ", np.std(maj_acc[1]))
    print("Top 1 accuracy on sim: ", np.mean(top_1_acc[2]), " +- ", np.std(top_1_acc[2]))
    print("Top 5 accuracy on sim: ", np.mean(top_5_acc[2]), " +- ", np.std(top_5_acc[2]))
    print("Maj accuracy on sim: ", np.mean(maj_acc[2]), " +- ", np.std(maj_acc[2]))
    print('t_tot:', np.mean(ts[0]/nb_images), "+-", np.std(ts[0]/nb_images))
    print('t_model:', np.mean(ts[1]/nb_images), "+-", np.std(ts[1]/nb_images))
    print('t_transfer:', np.mean(ts[3]/nb_images), "+-", np.std(ts[3]/nb_images))
    print('t model complet:', np.mean((ts[1]+ts[3])/nb_images), "+-", np.std((ts[1]+ts[3])/nb_images))
    print('t_search:', np.mean(ts[2]/nb_images), "+-", np.std(ts[2]/nb_images))

    return 0

def stat_process(model, model_weight, dataset, index_dataset, db_name, extractor, project_name, class_name, protocol):
    nb_exp = 5

    if protocol == 'random':
        nb_images = 1020
    else:
        nb_images = len(os.listdir(dataset))

    top_1_acc = np.zeros((3, nb_exp))
    top_5_acc = np.zeros((3, nb_exp))
    maj_acc = np.zeros((3, nb_exp))

    ts = np.zeros((4,nb_exp))
    for i in range(nb_exp):
        print("Experiment number: ", i)
        database = db.Database(db_name, model, load = False)
        database.add_dataset(index_dataset, extractor)

        top_1_acc[0][i], top_5_acc[0][i], top_1_acc[1][i], top_5_acc[1][i], top_1_acc[2][i], top_5_acc[2][i], maj_acc[0][i], maj_acc[1][i], maj_acc[2][i], ts[0][i], ts[1][i], ts[2][i], ts[3][i] =  test(model, model_weight,  dataset, db_name, extractor, protocol, project_name, class_name, False, stat = True)


    print("Top 1 accuracy: ", np.mean(top_1_acc[0]), " +- ", np.std(top_1_acc[0]))
    print("Top 5 accuracy: ", np.mean(top_5_acc[0]), " +- ", np.std(top_5_acc[0]))
    print("Maj accuracy: ", np.mean(maj_acc[0]), " +- ", np.std(maj_acc[0]))
    print("Top 1 accuracy on project: ", np.mean(top_1_acc[1]), " +- ", np.std(top_1_acc[1]))
    print("Top 5 accuracy on project: ", np.mean(top_5_acc[1]), " +- ", np.std(top_5_acc[1]))
    print("Maj accuracy on project: ", np.mean(maj_acc[1]), " +- ", np.std(maj_acc[1]))
    print("Top 1 accuracy on sim: ", np.mean(top_1_acc[2]), " +- ", np.std(top_1_acc[2]))
    print("Top 5 accuracy on sim: ", np.mean(top_5_acc[2]), " +- ", np.std(top_5_acc[2]))
    print("Maj accuracy on sim: ", np.mean(maj_acc[2]), " +- ", np.std(maj_acc[2]))
    print('t_tot:', np.mean(ts[0]/nb_images), "+-", np.std(ts[0]/nb_images))
    print('t_model:', np.mean(ts[1]/nb_images), "+-", np.std(ts[1]/nb_images))
    print('t_transfer:', np.mean(ts[3]/nb_images), "+-", np.std(ts[3]/nb_images))
    print('t model complet:', np.mean((ts[1]+ts[3])/nb_images), "+-", np.std((ts[1]+ts[3])/nb_images))
    print('t_search:', np.mean(ts[2]/nb_images), "+-", np.std(ts[2]/nb_images))

    return 0
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--path',
        default='patch/val'
    )

    parser.add_argument(
        '--path_indexed',   
    )
    

    parser.add_argument(
        '--extractor',
        default='resnet'
    )
    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--measure',
        help='random samples from validation set <random>, remove camelyon16_0 and janowczyk6_0 <remove>, all in separated class <separated>, all <all>, weighted <weighted> ',
        default = 'random'
    )

    parser.add_argument(
        '--stat',
        action='store_true'
    )
    
    parser.add_argument(
        '--project_name',
        help='name of the project of the dataset onto which to test the accuracy',
        default=None
    )
    
    
    parser.add_argument(
        '--class_name',
         help='name of the class of the dataset onto which to test the accuracy',
         default=None
    )
    
    parser.add_argument(
        '--excel_path',
        help='path to the excel file where the results must be saved',
        default = None
    )
    
    parser.add_argument(
        '--name',
        help='name to give to the sheet of the excel file',
        default = None
    )


    parser.add_argument(
        '--parallel',
        action = 'store_true'
    )

    args = parser.parse_args()
    
    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print('Path mentionned is not a folder')
        exit(-1)
    if args.class_name is not None and args.class_name not in os.listdir(args.path):
        print("Class name does not exist")
        exit(-1)
    
    model = Model(num_features=args.num_features, weight=args.weights, model=args.extractor,
                  device=device, parallel=args.parallel) # eval est par defaut true
        
    # Compute results per class and save them in given excel file 
    if args.measure == "separated":
        if args.excel_path is None:
            print("Please give the path to the excel file where to save the results")
            exit(-1)
        test_each_class(model, args.path, args.db_name, args.extractor, args.measure, args.name, args.excel_path)
    else:
        # Random protocol realised 50 times to make stat
        if args.stat == True:
            stat_process(model, args.weights, args.path, args.path_indexed, args.db_name, args.extractor,args.project_name, args.class_name, args.measure)
        else: # Other protocols: weighted - default - remove - random 
            r = test(model, args.weights, args.path, args.db_name, args.extractor, args.measure, args.project_name, args.class_name, True)
