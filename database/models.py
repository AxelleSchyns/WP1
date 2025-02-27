import torch
import dataset
import time
import timm

import numpy as np
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import resnet_ret as ResNet_ret

from transformers import DeiTForImageClassification
from loss import MarginLoss, ProxyNCA_prob, NormSoftmax, SoftTriple
from transformers import  ViTModel
from argparse import ArgumentParser
from arch import DINO, cdpath, iBot, ctranspath, BYOL
from utils import create_weights_folder, model_saving, write_info

 
archs_weighted = {"resnet": [models.resnet50(weights='ResNet50_Weights.DEFAULT'), 128], 
                  "deit": [DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224'),128], 
                  "dino_vit": [DINO("vit_small"), 384], "dino_resnet": [DINO("resnet50"), 384], 
                  "dino_tiny": [DINO("vit_tiny"), 384], # 'vit_tiny', 'vit_small', 'vit_base', n'importe lequel des CNNs de torchvision
                  "ret_ccl": [ResNet_ret.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear = True), 2048],
                  "cdpath": [cdpath(), 512], "phikon": [ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False), 768],
                  "ibot_vits": [iBot("vit_small"), 384], "ibot_vitb": [iBot("vit_base"), 384], "byol_light": [BYOL(67), 256], } 
try:
    archs_weighted["uni"] = [timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True), 1024]
    archs_weighted["hoptim"] = [timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False), 1536]
except Exception as e:
   archs_weighted["ctranspath"] =  [ctranspath(), 768]

class Model(nn.Module):
    def __init__(self, model='resnet', eval=True, batch_size=32, num_features=128,
                 weight='weights', device='cuda:0', freeze=False, parallel = True):
        super(Model, self).__init__()
        self.parallel = parallel
        self.norm = nn.functional.normalize
        self.weight = weight
        self.model_name = model
        self.device = device

        if model == 'deit':
            self.transformer = True
        else:
            self.transformer = False
        
        
        #--------------------------------------------------------------------------------------------------------------
        #                              Settings of the model
        #--------------------------------------------------------------------------------------------------------------
        if model in ['resnet', 'deit']:
            self.forward_function = self.forward_model
            self.num_features = num_features
        
        self.model = archs_weighted[model][0].to(device=device)
        self.num_features = archs_weighted[model][1]
        self.model = self.model.to(device=device)
        #----------------------------------------------------------------------------------------------------------------
        #                                  Freeze of model parameters
        #----------------------------------------------------------------------------------------------------------------
        if freeze and not self.transformer:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        #----------------------------------------------------------------------------------------------------------------
        #                                   Modification of the model's last layer
        #----------------------------------------------------------------------------------------------------------------
        # in features parameters = the nb of features from the models last layer
        # Given the model, this layer has a different name: classifier densenet, .fc resnet, ._fc effnet,... 
        # Localisation can be found by displaying the model: print(self.model) and taking the last layer name
        
        if model == 'resnet':
            self.model.fc = nn.Linear(self.model.fc.in_features, num_features).to(device=device)
        elif model == 'deit':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_features).to(device=device)
            if freeze: 
                for module in filter(lambda m: type(m) == nn.LayerNorm, self.model.modules()):
                    module.eval()
                    module.train = lambda _: None

        if eval == True:
            if model in ['dino_vit', 'dino_resnet','dino_tiny', "cdpath", "ibot_vits" , "ibot_vitb"]:
                self.model.load_weights(weight)
                self.model = self.model.model
            elif model == "byol_light" :
                try:
                    self.model.load_state_dict(torch.load(weight)["state_dict"])
                except:
                    self.model = BYOL(1000).to(device=device)
                    self.model.load_state_dict(torch.load(weight)["state_dict"])
            elif model == "ret_ccl":
                pretext_model = torch.load(weight)
                self.model.fc = nn.Identity()
                self.model.load_state_dict(pretext_model, strict=True)
            elif model == "phikon" or model == "hoptim": 
                pass
            elif model == "ctranspath":
                self.model.head = nn.Identity()
                self.model.load_state_dict(torch.load(weight)['model'])
            elif model == "uni":
                self.model.load_state_dict(torch.load(weight))
            else:
                try:
                    self.load_state_dict(torch.load(weight))
                except Exception as e:
                    try:
                        checkpoint = torch.load(weight)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        #self.model.load_state_dict(checkpoint)
                    except Exception as e:
                        print("Error with the loading of the model's weights: ", e) 
                        print("Exiting...")
                        exit(-1)
            self.forward_function = self.model.forward
            self.eval()
            self.eval = True
        else:
            if parallel:
                self.model = nn.DataParallel(self.model)
            self.train()
            self.eval = False
            self.batch_size = batch_size

    def forward_model(self, input):
        tensor1 = self.model(input)
        if self.model_name == 'deit':
            tensor1 = tensor1.logits
        tensor1 = self.norm(tensor1)

        return tensor1

    def forward(self, input):
        return self.forward_function(input)
    
    def get_optim(self, data, loss, lr, decay, beta_lr, lr_proxies):
        if loss == 'margin':
            loss_function = MarginLoss(self.device, n_classes=len(data.classes))

            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay},
                        {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'proxy_nca_pp':
            loss_function = ProxyNCA_prob(len(data.classes), self.num_features, 3, device)

            to_optim = [
                {'params':self.parameters(), 'weight_decay':0},
                {'params':loss_function.parameters(), 'lr': lr_proxies}, # Allows to update automaticcaly the proxies vectors when doing a step of the optimizer
            ]

            optimizer = torch.optim.Adam(to_optim, lr=lr, eps=1)
        elif loss == 'softmax':
            loss_function = NormSoftmax(0.05, len(data.classes), self.num_features, lr_proxies, self.device)

            to_optim = [
                {'params':self.parameters(),'lr':lr,'weight_decay':decay},
                {'params':loss_function.parameters(),'lr':lr_proxies,'weight_decay':decay}
            ]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'softtriple':
            # Official - paper implementation
            loss_function = SoftTriple(self.device)
            to_optim = [{"params": self.parameters(), "lr": 0.0001},
                                  {"params": loss_function.parameters(), "lr": 0.01}] 
            optimizer = torch.optim.Adam(to_optim, eps=0.01, weight_decay=0.0001)
        else:
            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay}] # For byol: lr = 3e-4
            optimizer = torch.optim.Adam(to_optim)
            loss_function = None
        return optimizer, loss_function

    def train_model(self, loss_name, epochs, training_dir, lr, decay, beta_lr, lr_proxies, sched, gamma, informative_samp = True, starting_weights = None, epoch_freq = 20, need_val = True):
        
        # download the dataset
        if need_val:
            data = dataset.TrainingDataset(root = training_dir, model_name = self.model_name, samples_per_class= 2,  informative_samp = informative_samp, need_val=2)
            data_val = dataset.TrainingDataset(root = training_dir, model_name = self.model_name, samples_per_class= 2, informative_samp = informative_samp, need_val=1)
            loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                                shuffle=True, num_workers=12,
                                                pin_memory=True)
            loader_val = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size,
                                                shuffle=True, num_workers=12,
                                                pin_memory=True)
            loaders = [loader, loader_val]
            print('Size of the training dataset:', data.__len__(), '|  Size of the validation dataset: ', data_val.__len__() )

            losses_mean = [[],[]]
            losses_std = [[],[]]
        else:   
            data = dataset.TrainingDataset(root = training_dir, model_name = self.model_name, samples_per_class= 2, informative_samp = informative_samp, need_val=0)
            print('Size of dataset', data.__len__())
            loaders = [torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                                shuffle=True, num_workers=12,
                                                pin_memory=True)]
            losses_mean = [[]]
            losses_std = [[]]
        
        # Creation of the optimizer and the scheduler
        optimizer, loss_function = self.get_optim(data, loss_name, lr, decay, beta_lr, lr_proxies)
        starting_epoch = 0
        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                            gamma=gamma)
        
        range_plot = range(epochs)

        # Creation of the folder to save the weight
        weight_path = create_weights_folder(self.model_name, starting_weights)
        write_info(self, weight_path, lr, decay, beta_lr, lr_proxies, loss_name, epochs, informative_samp, need_val, sched, gamma)
        # Downloading of the pretrained weights and parameters 
        if starting_weights is not None:
            checkpoint = torch.load(starting_weights)
            if self.parallel:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            loss_function = checkpoint['loss_function']
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            try:
                checkpoint_loss = torch.load(weight_path + '/loss')
                losses_mean = checkpoint_loss['loss_means']
                losses_std = checkpoint_loss['loss_stds']

            except:
                print("Issue with the loss file, it will be started from scratch")
                range_plot = range(starting_epoch, epochs)


        try:
            for epoch in range(starting_epoch, epochs):
                start_time = time.time()
                if need_val:
                    loss_list_val = []
                    loss_list = []
                    loss_lists = [loss_list, loss_list_val]
                else:
                    loss_list = []
                    loss_lists = [loss_list]
                for j in range(len(loaders)):
                    loader = loaders[j]
                    for i, (labels, images) in enumerate(loader):
                        if i%1000 == 0 and j ==0:
                            print(i, flush=True)
                        
                        images_gpu = images.to(self.device)
                        labels_gpu = labels.to(self.device)
                        if self.transformer:
                            out = self.forward(images_gpu.view(-1, 3, 224, 224))
                        else:
                            out = self.forward(images_gpu)
                        if loss_function is None:
                            print("This model requires a specific loss. Please specifiy one. ")
                            exit(-1)
                        loss = loss_function(out, labels_gpu)

                        if j == 0:
                            # Update
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()

                        loss_lists[j].append(loss.item())
                
                if need_val:
                    print("epoch {}, loss = {}, loss_val = {}, time {}".format(epoch, np.mean(loss_lists[0]),
                                                            np.mean(loss_lists[1]), time.time() - start_time))
                    losses_mean[1].append(np.mean(loss_lists[1]))
                    losses_std[1].append(np.std(loss_lists[1]))
                else:
                    print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_lists[0]),
                                                            time.time() - start_time))
                
                print("\n----------------------------------------------------------------\n")
                losses_mean[0].append(np.mean(loss_lists[0]))
                losses_std[0].append(np.std(loss_lists[0]))
                if sched != None:
                    scheduler.step()

                # Saving of the model
                model_saving(self.model, epoch, epochs, epoch_freq, weight_path, optimizer, scheduler, loss, loss_function, loss_list, losses_mean, losses_std)

            if need_val:
                plt.figure()
                plt.errorbar(range_plot, losses_mean[1], yerr=losses_std[1], fmt='o--k',
                         ecolor='lightblue', elinewidth=3)
                plt.savefig(weight_path+"/validation_loss.png")
            plt.figure()
            plt.errorbar(range_plot, losses_mean[0], yerr=losses_std[0], fmt='o--k',
                         ecolor='lightblue', elinewidth=3)
            plt.savefig(weight_path+"/training_loss.png")
                    
        
        except KeyboardInterrupt:
            print("Interrupted")
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        type=int,
        help='number of features to use if supervised models - Default 128',
        default=128
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--model',
        help='feature extractor to use',
        default='resnet'
    )

    parser.add_argument(
        '--weights',
        default='weights'
    )

    parser.add_argument(
        '--training_data',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10
    )

    parser.add_argument(
        '--scheduler',
        default='exponential',
        help='<exponential, step>'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--loss',
        default=None,
        help='<margin, proxy_nca_pp, softmax, softtriple>'
    )

    parser.add_argument(
        '--starting_weights',
        default=None,
        help='path to the weights of the model to continue the training with'
    )

    parser.add_argument(
        '--freeze',
        action='store_true'
    )

    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '--decay',
        default=0.0004,
        type=float
    )

    parser.add_argument(
        '--beta_lr',
        default=0.0005,
        type=float
    )

    parser.add_argument(
        '--gamma',
        default=0.3,
        type=float
    )

    parser.add_argument(
        '--lr_proxies',
        default=0.00001,
        type=float
    )

    parser.add_argument(
        '--parallel',
        action = 'store_true'
    )

    parser.add_argument(
        '--i_sampling',
        default = None,
        help='whether or not ot use informative (label-based) sampling',
    )
    parser.add_argument(
        '--epoch_freq',
        default = 20,
        type = int,
        help='frequency of saving the model'
    )

    parser.add_argument(
        '--remove_val',
        action='store_true',
        help='whether or not to use validation set'
    )



    args = parser.parse_args()
    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.i_sampling is None:
        args.i_sampling = "True"

    m = Model(model=args.model, eval=False, batch_size=args.batch_size,
              num_features=args.num_features, weight=args.weights,
              device=device, freeze=args.freeze, parallel=args.parallel)
    
    m.train_model(loss_name = args.loss, epochs = args.num_epochs, training_dir=args.training_data, sched = args.scheduler, lr = args.lr, decay = args.decay, gamma = args.gamma, beta_lr = args.beta_lr, lr_proxies = args.lr_proxies, informative_samp = args.i_sampling, starting_weights=args.starting_weights, epoch_freq = args.epoch_freq, need_val = not args.remove_val)
    
    