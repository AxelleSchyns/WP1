
import os
import sys
import torch
import utils_dino 
import timm
import copy 

import torch.nn as nn
import cdpath_models as cdpath_models 
import vision_transformer as vits
import torch.nn.functional as F

from torchvision import models as torchvision_models
from vision_transformer import DINOHead
from pytorch_lightning import LightningModule
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.loss import NegativeCosineSimilarity
from timm.models.layers import to_2tuple
from torch import Tensor

class BYOL(LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        resnet = torchvision_models.resnet50()
        resnet.fc = nn.Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = BYOLProjectionHead()
        self.prediction_head = BYOLPredictionHead()
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = BYOLProjectionHead()
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.projection_head(x)


class DINO():
    def __init__(self, model_name):
        super().__init__()
        if model_name in vits.__dict__.keys():
            self.model = vits.__dict__[model_name](patch_size=16,)
        elif model_name in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[model_name]()

        self.model = utils_dino.MultiCropWrapper(self.model, DINOHead(
            2048,
            65536,
            use_bn=False,
            norm_last_layer=True
        ))
        self.model = self.model.backbone
    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path)
        if 'student' in state_dict.keys():
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict['student'].items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k: v for k, v  in state_dict.items() if k.startswith("backbone.")}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        
        print('Pretrained weights found at {} and loaded with msg: {}'.format(weight_path, msg))
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

class cdpath():
    def __init__(self) -> None:
        super().__init__()
        self.generator = cdpath_models.Generator(64, 112, 512, backbone=True)
    
    def load_weights(self, weight_path):
        self.model = None
        checkpoint = torch.load(weight_path)
        self.model = checkpoint['generator']
    
    def to(self, device):
        self.model = self.generator.to(device)
        return self
def scale_generator(x_batch, size, alpha, x_dim, rescale_size=224):

    x_batch = F.interpolate(x_batch, size=rescale_size, mode='bilinear',
                            align_corners=True)

    margin = (rescale_size - x_dim) // 2
    x_crop = x_batch[:, :, margin:rescale_size-margin, margin:rescale_size-margin]


    return x_crop


class iBot():
    def __init__(self, model_name):
        super().__init__()   
        # ============ building network ... ============
        if "vit" in model_name:
            model = vits.__dict__[model_name](
                patch_size=16, 
                num_classes=0,
                use_mean_pooling=True) # true if using beit model
            print(f"Model {model_name} {16}x{16} built.")
        elif model_name in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[model_name](num_classes=0)
        else:
            print(f"Architecture {model_name} non supported")
            sys.exit(1)
        self.model = model
        # TODO check the difference with the addition of multicrop wrapper
        self.model_name = model_name
        self.patch_size = 16
    def load_weights(self, weight_path):
        # load pretrained weights
        if os.path.isfile(weight_path):
            state_dict = torch.load(weight_path)
            
            # Uncomment this line if using the pre-trained weights
            try:
                state_dict = state_dict["state_dict"] # when using pre-trained wieghts
            except:
                pass
            if "teacher" in state_dict:
                print(f"Take key teacher in provided checkpoint dict")
                state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(weight_path, msg))
        else:
            url = None
            if self.model_name == "vit_small" and  self.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif self.model_name == "vit_small" and self.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif self.model_name == "vit_base" and self.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif self.model_name == "vit_base" and self.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.model.load_state_dict(state_dict, strict=True)
                # print("Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2.")
                # model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))
            else:
                print("There is no reference weights available for this model => We use random weights.")


    def to(self, device):
        self.model = self.model.to(device)
        return self

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model =  timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model