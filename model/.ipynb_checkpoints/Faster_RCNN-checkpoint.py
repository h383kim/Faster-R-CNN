import torch
from torch import nn
from model.utils import *
from torchvision import models
from model.RPN import RegionProposalNetwork
from model.ROI_Detector import ROI_Detector

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
        old_shape = image.shape[-2:]
        if self.training:
            target['labels'] = target['labels'].squeeze(dim=0)
            # image from (N, W, C, H) -> (N, C, H, W)
            image = image.permute(0, 2, 3, 1)
            # Normalize and resize image and boxes
            image, bboxes = normalize_resize_image_and_boxes(image, target['bboxes'], self.image_mean, self.image_std)
            target['bboxes'] = bboxes.squeeze(dim=0)
        else:
            # image from (C, H, W) -> (N, C, H, W)
            image = image.unsqueeze(dim=0)
            # Normalize and resize image only when inferencing
            image, _ = normalize_resize_image_and_boxes(image, None, self.image_mean, self.image_std)
            
        # Feed forward shared backbone
        feat_map = self.shared_backbone(image)
        # Feed forward RPN and get proposals
        rpn_output = self.rpn(feat_map, image, target)
        proposals = rpn_output['proposals']
        # Feed forward ROI_Detector
        roi_detector_output = self.roi_detector(feat_map, proposals, image.shape[-2:], target)
        
        ''' If Inferencing, find predicted boxes in original image size '''
        if not self.training:
            roi_detector_output['bboxes'] = transform_boxes_to_original_size(roi_detector_output['bboxes'],
                                                                             image.shape[-2:],
                                                                             old_shape)
        return rpn_output, roi_detector_output