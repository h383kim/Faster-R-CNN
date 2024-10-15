import numpy as np
import torch
from torch import nn
from model.utils import *



class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512,
                 scales=[128, 256, 512], ratios=[0.5, 1, 2]):
        super().__init__()
        self.scales = scales
        self.ratios = ratios
        self.num_anchors = len(scales) * len(ratios) # 3 x 3 = 9 anchors per each (x,y)
        ''' 
        3x3 conv before 2 siblings(cls, loc)
        [B, 512, H, W] -> [B, 512, H, W]
        '''
        self.rpn_conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3), stride=1, padding=1)
        '''
        1x1 classification layer (Score whether it's foreground)
        [B, 512, H, W] -> [B, num_anchors, H, W]
        The resulting feature maps will have spatial dimension remained H x W
        and each feature map at (x,y) will represent the "Predicted back/foreground scores for each anchor"
        '''
        self.cls_layer = nn.Conv2d(in_channels=mid_channels, out_channels=self.num_anchors, kernel_size=(1, 1), stride=1)
        '''
        1x1 conv layer to predict bbox regression offset
        [B, 512, H, W] -> [B, num_anchors * 4, H, W]
        The resulting feature maps will have spatial dimension remained H x W
        and each feature map at (x,y) will represent the "Predicted bounding box offsets and scales for each anchor"
        '''
        self.bbox_layer = nn.Conv2d(in_channels=mid_channels, out_channels=self.num_anchors*4, kernel_size=(1, 1), stride=1)


    def forward(self, feat_map, image, target=None):
        ''' Step 1: Feed forward to get class / box_transform prediction feature maps'''
        rpn_feat = nn.ReLU()(self.rpn_conv(feat_map))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_layer(rpn_feat)

        ''' Step 2: Generate base anchors for each location on feature map'''
        '''
        :base_anchors: [feat_H * feat_W, num_anchors, 4]
        '''
        base_anchors = generate_anchor_base(image, feat_map)

        ''' Step 3: Reshaping cls_scores and box_transform_pred feature maps '''
        '''
        :before: [B, num_anchors, feat_H, feat_W]
        :after:  [B * feat_H * feat_W * num_anchors, 2]
        '''
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        '''
        :before: [B, num_anchors * 4, feat_H, feat_W]
        :after:  [B * feat_H * feat_2 * num_anchors, 4]
        '''
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            self.num_anchors,
            4,
            box_transform_pred.shape[-2],
            box_transform_pred.shape[-1]
        )
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        ''' Step 4: Transform the base anchors by applying the predicted box_transform (i.e. box_transform_pred)'''
        proposals = apply_transform_to_baseAnchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4), # Adding a new dimension num_classes=1 to fit the method requirement 
            base_anchors
        )
        proposals = proposals.reshape(proposals.size(0), 4) # Converting back to original [num_anchors, 4] shape. [num_anchors=B*H*W*num_anchors]
        ''' Step 5: Sample Proposals and corresponding objectness scores '''
        proposals, scores = sample_proposals(proposals, cls_scores.detach(), image.shape)
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }
        
        # During the Inference step(Not training), skip below
        if not self.training or target is None:
            return rpn_output
        # During Training
        else:
            ''' Step (1): Assign label and target gt_box to each anchor '''
            labels_of_anchors, gt_boxes_for_anchors = assign_targets_to_anchors(target['bboxes'], base_anchors)
            ''' Step (2): Based on the assigned labels / bboxes, compute bbox regression targets '''
            target_box_offsets = compute_bbox_transformation_targets(gt_boxes_for_anchors, base_anchors)
            ''' Step (3): Sample positive / negative samples '''
            sampled_pos_mask, sampled_neg_mask = sample_positive_and_negative(labels_of_anchors, positive_ct=128, total_ct=256)
            sampled_total = torch.where(sampled_pos_mask | sampled_neg_mask)[0]
            ''' Step (4): Compute Localization Loss '''
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_mask], 
                    target_box_offsets[sampled_pos_mask],
                    beta=1/9,
                    reduction="sum"
                ) / sampled_total.numel())
            ''' Step (5): Compute objectiveness Loss '''
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_total].flatten(),
                labels_of_anchors[sampled_total].flatten()
            )

            rpn_output['rpn_localization_loss'] = localization_loss
            rpn_output['rpn_cls_loss'] = cls_loss
            return rpn_output