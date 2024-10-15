import torch
from torch import nn
from model.utils import *


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
        self.fc6 = nn.Sequential(
            nn.Linear(in_channels * (self.roi_pool_size ** 2), self.fc_dim),
            nn.ReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.ReLU()
        )
        # Two sibling layers (cls, bbox_regressor) after FC Layers
        self.cls_layer = nn.Linear(self.fc_dim, self.num_classes + 1)
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
            sampled_pos_mask, sampled_neg_mask = sample_positive_and_negative(labels_for_proposals, positive_ct=32, total_ct=128)
            sampled_total = torch.where(sampled_pos_mask | sampled_neg_mask)[0]

            proposals = proposals[sampled_total] # [num_sampled_total, 4] = (128, 4) in our case
            labels_for_proposals = labels_for_proposals[sampled_total]     # [num_sampled_total,]
            gt_boxes_for_proposals = gt_boxes_for_proposals[sampled_total] # [num_sampled_total, 4]
            target_box_offsets = compute_bbox_transformation_targets(gt_boxes_for_proposals, proposals) # [num_sampled_total, 4]

        
        ''' Compute Scaling '''
        size = feat_map.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, image_shape):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        ''' ROI_Pooling layer '''
        roi_pool_feat = torchvision.ops.roi_pool(feat_map, 
                                                 [proposals], 
                                                 output_size=self.roi_pool_size, 
                                                 spatial_scale=possible_scales[0])
        # Flattening roi_pooled feature with batch reserved (N, C, H, W) -> (N, C * H * W)
        roi_pool_feat = roi_pool_feat.flatten(start_dim=1)
        fc_out = self.fc7(self.fc6(roi_pool_feat))
        ''' Sibling Layers: cls_layer / bbox_regressor '''
        cls_scores = self.cls_layer(fc_out) 
        box_transform_pred = self.bbox_regressor(fc_out)
        num_proposals = cls_scores.shape[0]
        box_transform_pred = box_transform_pred.reshape(num_proposals, self.num_classes, 4) #shape: [num_proposals, num_classes, 4]=[128, 20, 4]

        frcnn_output = {}
        
        ''' Compute Loss if Training stage '''
        if self.training and target is not None:
            ''' Classification Loss '''
            # cls_scores: [128, 21]
            # labels_for_proposals: [128,]
            # ignore_index doesn't take 0 from labels_for_proposals as it is background
            cls_loss = torch.nn.functional.cross_entropy(cls_scores, labels_for_proposals.long())
            ''' Localization Loss '''
            # Step 1: get the proposals that are foreground
            fg_proposals = torch.where(labels_for_proposals > 0)[0]
            # Step 2: get the class idx for each foreground proposals' class so that we select only those from box_transform_pred
            #         Pick [num_foreground, 4] from the entire box_transform_pred of shape [128, 20, 4]
            # Subtract 1 from all because there are 20 fg_classes while classification includes background = 21
            fg_cls_idx = labels_for_proposals[fg_proposals] - 1
            # Step 3: compute loss
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[fg_proposals, fg_cls_idx],
                    target_box_offsets[fg_proposals],
                    beta = (1/9),
                    reduction="sum"
                ) / fg_proposals.numel()
            )
            ''' Saving Ouptut '''
            frcnn_output['cls_loss'] = cls_loss
            frcnn_output['localization_loss'] = localization_loss


        ''' If training, output result now '''
        if self.training:
            return frcnn_output
        
        else:
            ''' 
            If Inferencing
            :cls_scores: [num_proposals, 21] where num_proposals is around ~2000
            :box_transform_pred: [num_proposals, 4] 
            '''
            device = cls_scores.device
            
            ''' predicted bounding boxes transformed and clamped '''
            pred_boxes_transformed = apply_transform_to_baseAnchors_or_proposals(box_transform_pred, proposals)
            pred_boxes_transformed = clamp_boxes(pred_boxes_transformed, image_shape) # [num_proposals, num_classes, 4]
            
            ''' predicted scores '''
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1) # [num_proposals. 21]
            
            ''' labels '''
            pred_labels = torch.argmax(pred_scores, dim=1) # [num_proposals,]
            
            ''' Filter out background predictions '''
            fg_proposals = torch.where(pred_labels > 0)[0]
            pred_boxes_transformed = pred_boxes_transformed[fg_proposals] # [num_fg_proposals, num_classes, 4]
            pred_scores = pred_scores[fg_proposals, 1:] # [num_fg_proposals, 20]
            pred_labels = pred_labels[fg_proposals]     # [num_fg_proposals,]
            
            ''' Apply class-wise NMS '''
            pred_boxes, pred_scores, pred_labels = apply_nms(pred_boxes_transformed.detach(), pred_scores.detach(), pred_labels.detach())
            frcnn_output['bboxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            
            return frcnn_output