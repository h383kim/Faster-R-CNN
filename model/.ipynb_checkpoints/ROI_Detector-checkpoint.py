import torch
from torch import nn
from utils import *

##### Global Variables #####
ROI_POOL_SIZE = 7
FCL_DIM = 2048
############################

class ROI_Detector(nn.Module):
    def __init__(self, in_channels=512, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.roi_pool_size = ROI_POOL_SIZE
        self.fc_dim = FCL_DIM

        '''
        ROI Pooling layer is defined and done in forward() method as proposals is not ready at this stage
        '''
        # Following FC layers after ROI pooling is done
        self.fc6 = nn.Linear(in_channels * (self.roi_pool_size ** 2), self.fc_dim) 
        self.fc7 = nn.Linear(self.fc_dim, self.fc_dim)
        # Two sibling layers (cls, bbox_regressor) after FC Layers
        self.cls_layer = nn.Lienar(self.fc_dim, self.num_classes)
        self.bbox_regressor = nn.Linear(self.fc_dim, self.num_classes * 4)

        '''
        Initializing sibling layers' weights/biases
        '''
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)
        torch.nn.init.normal_(self.bbox_regressor.weight, std=0.01)
        torch.nn.init.constant_(self.bbox_regressor.bias, 0)

    def forward(self, feat_map, proposals, image_shape, target):
        if self.training and target is not None:
            # The proposals will be that resulted from RPN, PLUS the ground truth bboxes which is are good samples for training Detector
            proposals = torch.cat([proposals, target['bboxes']], dim = 0) # Shape: [num_proposals_from_RPN + num_gt_boxes, 4]
            gt_boxes = target['bboxes']
            gt_labels = target['labels']
            ''' Step 1: tagging proposals '''
            labels_for_proposals, gt_boxes_for_proposals = assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
            ''' Step 2: sample positive and negative '''
            sampled_pos_mask, sampled_neg_mask = sample_positive_negative(labels_for_proposals, positive_ct=32, total_ct=128)
            sampled_total = torch.where(sampled_pos_mask | sampled_neg_mask)[0]

            proposals = proposals[sampled_total] # [num_sampled_total, 4]
            labels_for_proposals = labels_for_proposals[sampled_total]     # [num_sampled_total,]
            gt_boxes_for_proposals = gt_boxes_for_proposals[sampled_total] # [num_sampled_total, 4]
            target_box_offsets = compute_bbox_transformation_targets(gt_boxes_for_proposals, proposals) # [num_sampled_total, 4]

        
        ''' Compute Scaling '''

        ''' ROI_Pooling layer '''
        roi_pool_feat = torchvision.ops.roi_pool(feat_map, 
                                                 [proposals], 
                                                 output_size=self.roi_pool_size, 
                                                 spatial_scale=)
        