import numpy as np
from PIL import Image
import torch 
import os
import matplotlib.pyplot as plt

# given the filename, return the image object 
def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        image = image.resize((224,224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return image.reshape(-1)

# Define a function to generate batches of image paths
def batch_image_paths(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i+batch_size]

# For the AE model, retrieve the latent representation of an image
def encode(model, img):
    exp = 0
    if exp == 7:
        code = model.module.encode_7(img).cpu()
    else:
        with torch.no_grad():
            if exp=="3b":
                code = model.model.encoder(img).cpu()
            elif exp==4:
                code = model.model.encoder(img).cpu()
            else:
                code = model.module.encoder(img).cpu()
    return code

# Get the class of an image from its path
# ! Only works when the path is in the form: /home/.../project/class/image_name
def get_class(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1
    if end_c == -1 or begin_c == -1:
        return -1
    return path[begin_c:end_c]

# Get the project name from the path
def get_proj(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1
    end_proj = path[begin_c:end_c].rfind("_")
    proj_name = path[begin_c:begin_c+end_proj]
    if proj_name.rfind("_") != -1:
        rest = proj_name[proj_name.rfind("_")+1: len(proj_name)]
        if rest.isdigit():
                proj_name = proj_name[0:proj_name.rfind("_")]
    return proj_name

# Change class names to have all class number starting from 0
def rename_classes(class_list):
    classes = os.listdir(class_list)
    classes.sort()
    cpt_c = 0
    old_name = ""
    new_classes = []
    for c in classes:
        idx = c.rfind("_")
        c_project = c[0:idx]
        if c_project.rfind("_") != -1:
            rest = c_project[c_project.rfind("_")+1: len(c_project)]
            if rest.isdigit():
                c_project = c_project[0:c_project.rfind("_")]
        if c_project == old_name:
            cpt_c += 1
        else: 
            cpt_c = 0

        new_classes.append(c_project + "_" + str(cpt_c))
        old_name = c_project
    return new_classes

# Get new class name of a class
def get_new_name(class_name, path=None):
    if path == None:
        classes = os.listdir("/home/labarvr4090/Documents/Axelle/cytomine/Data/validation")
    else:
        classes = os.listdir(path)
    classes.sort()
    proj = get_proj(class_name)
    cpt_c = 0
    for c in classes:
        if c == class_name:
            return proj + "_" + str(cpt_c)
        elif proj in c:
            cpt_c += 1
    
    return -1 

def create_weights_folder(model_name, starting_weights = None):
    try:
        os.mkdir("weights_folder")
    except FileExistsError:
        pass
    try:
        os.mkdir("weights_folder/"+model_name)
    except FileExistsError:
        pass
    if starting_weights != None:
        id_start = starting_weights.rfind("version")
        if id_start == -1:
            print("Issue with the format of the folder containing the weight, please check that it is in the form: weights_folder/model_name/version_x")
            exit(-1)
        id_end = starting_weights[id_start:].rfind("/") + id_start
        weight_path = starting_weights[0:id_end]
    else:
        versions = []
        for file in os.listdir("weights_folder/"+model_name):
            id_ = file.find("_")
            if id_ != 7:
                continue
            versions.append(int(file[8:]))
        versions.sort()
        
        count = 0
        for nb in versions:
            if nb != count:
                break
            count += 1
        weight_path = "weights_folder/"+model_name+"/version_"+str(count)
        try:
            os.mkdir(weight_path)
        except FileExistsError:
            print("Issue with the creation of the folder, risk of overwriting existing files.")
    
    return weight_path
    
def model_saving(model, epoch, epochs,  epoch_freq, weight_path, optimizer, scheduler, loss, loss_function, loss_list, loss_mean, loss_stds):
    try:
        model = model.module
    except AttributeError:
        pass

    state = np.isnan(loss_list).any() # Indicate if the model has diverged

    if state == False:
        # Save separate file every epoch_freq epoch
        if epoch == 0 or (epoch+1) % epoch_freq == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'loss_function': loss_function,
            }, weight_path + "/epoch_"+str(epoch))
        
        # Save the last epoch (overwrite the file)
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'loss_function': loss_function,
        }, weight_path + "/last_epoch")

        # Save the loss (overwrite the file)
        torch.save({
            'loss_means': loss_mean,
            'loss_stds': loss_stds,
            }, weight_path + "/loss")
        
        # Save the model corresponding to the lowest validation loss
        if len(loss_mean) > 1:
            if len(loss_mean[1]) == 1 or loss_mean[1][-1] < min(loss_mean[1][:-1]):
                if len(loss_mean[1]) > 1:
                    print("Loss mean: ", loss_mean[1][-1], " min: ", min(loss_mean[1][:-1]))
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'loss_function': loss_function,
                }, weight_path + "/best_model")
    else:
        print("Loss is nan, not saving the model and stopping the training")
        plt.errorbar(range(epoch-1), loss_mean[:-1], yerr=loss_stds[:-1], fmt='o--k',
                ecolor='lightblue', elinewidth=3)
        plt.savefig(weight_path+"/training_loss.png")
        exit(-1)