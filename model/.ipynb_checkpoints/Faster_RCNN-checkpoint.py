import torch
from torch import nn
from utils import *
from torchvision import models
from RPN import RegionProposalNetwork
from ROI_Detector import ROI_Detector

##### Global Variables #####
IMAGE_MIN_DIM = 600
IMAGE_MAX_DIM = 1000
############################


vgg16 = models.vgg16_bn(weights="IMAGENET1K_V1", progress=False)

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.image_min_size = IMAGE_MIN_DIM
        self.image_max_size = IMAGE_MAX_DIM
        
        ''' Shared Backbone Conv Layers '''
        self.shared_backbone = vgg16.features[:-1]
        ''' Region Proposal Network '''
        self.rpn = RegionProposalNetwork()
        ''' ROI_Detector Layers '''
        self.roi_detector = ROI_Detector()

        ''' Freezing Early Layers '''
        for layer in self.shared_backbone[:10]:
            for param in layer.parameters():
                param.requires_grad = False



    def forward(self, image, target=None):
        