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
def compute_results(names, data, predictions, class_im, top_1_acc, top_5_acc, maj_acc, predictions_maj, weights):
    similar = names[:5]
    temp = []
    already_found_5 = 0
    if len(data.classes) == 1:
        idx_class = 0
    else:
        idx_class = data.conversion[class_im]
    for j in range(len(similar)):
        # Gets the class and project of the retrieved image
        class_retr = utils.get_class(similar[j])
        temp.append(class_retr)
        
        # Retrieves label of top 1 result for confusion matrix
        if j == 0:
            # if class retrieved is in class of data given
            if class_retr in data.conversion:
                predictions.append(data.conversion[class_retr])
            else:
                predictions.append("other") # to keep a trace of the data for the cm
        
        # Class retrieved is same as query
        if class_retr == class_im: 
            if already_found_5 == 0:
                top_5_acc += weights[idx_class]
        
                if j == 0:
                    top_1_acc += weights[idx_class]
                    
            already_found_5 += 1 # One of the 5best results matches the label of the query ->> no need to check further
        
    if already_found_5 > 2:
        maj_acc += weights[idx_class]
    
    predictions_maj.append(data.conversion[max(set(temp), key = temp.count)])

    return predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc


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
    plt.savefig(fold_path + '/crc_confusion_matrix_top1_'+measure+ '.png')
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
    plt.savefig(fold_path + '/crc_confusion_matrix_maj_'+measure+'.png')

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
    for i in range(len(classes)):
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
    plt.savefig(fold_path + '/crc_prec_recall_curve_'+measure+'.png')
    


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
    def __init__(self, model, root, measure, class_name = None):
        
        self.root = root
        self.feat_extract = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
        
        # Define the transforms 
        if model.model_name == 'deit':
            self.transformer = True
        else:
            self.transformer = False

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
def test(model, model_weight, dataset, db_name, extractor, measure, class_name, see_cms, stat = False):

    # Load database
    database = db.Database(db_name, model, True)

    # Load data 
    data = TestDataset(model, dataset, measure, class_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)
    
    # Load weights 
    if measure == 'weighted':
        weights = data.weights
    else:
        weights = np.ones(len(data.classes))
    

    top_1_acc = 0 
    top_5_acc = 0
    maj_acc = 0

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []
    predictions_maj = []
    
    t_search = 0
    t_model = 0
    t_transfer = 0
    t_tot = 0

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
        nbr_per_class[class_im] += 1
        ground_truth.append(data.conversion[class_im])


        # Compute accuracy 
        predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc = compute_results(names, data, predictions, class_im, top_1_acc,top_5_acc,maj_acc,predictions_maj, weights)

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
        print("top-1 accuracy : ", top_1_acc / s)
        print("top-5 accuracy : ", top_5_acc / s)
        print("maj accuracy class : ", maj_acc / s)
        print("f1 score : ", f1)

        print('t_tot:', t_tot)
        print('t_model:', t_model)
        print('t_transfer:', t_transfer)
        print('t_search:', t_search)

    # Display results in graphics
    if see_cms:
        display_cm(model_weight, measure, ground_truth, data, predictions, predictions_maj)
        display_precision_recall(model_weight, measure, ground_truth, predictions)
    
    return [top_1_acc/ s, top_5_acc/ s, maj_acc/ s, t_tot, t_model, t_search, t_transfer]

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
        help='all in separated class <separated>, all <all>, weighted <weighted>',
        default = 'all'
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
        r = test(model, args.weights, args.path, args.db_name, args.extractor, args.measure, args.class_name, True)
