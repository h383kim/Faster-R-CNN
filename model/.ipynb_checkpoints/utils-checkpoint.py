import torch
import torchvision
from torchvision.ops import nms
import numpy as np

##### Global Variables #####
PRE_NMS_TOPK = 12000
FINAL_TOPK = 2000
LOW_IOU_THRESHOLD = 0.3
HIHG_IOU_THRESHOLD = 0.7
ROI_LOW_IOU_THRESHOLD = 0.1
ROI_HIGH_IOU_THRESHOLD = 0.5
############################



##############################################
########## RPN & ROI_Detector Utils ##########
##############################################

def compute_iou(boxes1, boxes2):
    '''
    :boxes1: Tensor of shape [N, 4]
    :boxes2: Tensor of shape [M, 4]
    :returns: IOU matrix of shape [N, M]
    '''
    # areas of each boxes from boxes1 and boxes2
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # (M,)

    # Top-left coordinates of intersection boxes
    tl_x = torch.max(boxes1[:, None, 0], boxes2[:, 0]) # (N, M)
    tl_y = torch.max(boxes1[:, None, 1], boxes2[:, 1]) # (N, M)
    # Bottom-right coordinates of intersection boxes
    br_x = torch.max(boxes1[:, None, 2], boxes2[:, 2]) # (N, M)
    br_y = torch.max(boxes1[:, None, 3], boxes2[:, 3]) # (N, M)

    # Intersections areas
    intersect = (br_x - tl_x).clamp(min=0) * (br_y - tl_y).clamp(min=0) # (N, M)

    # area1 + area2 - intersect
    union = area1[:, None] + area2 - intersect # (N, M)
    iou = intersect / union # (N, M)
    return iou


def apply_transform_to_baseAnchors_or_prposals(box_transform_pred, baseAnchors_or_proposals):
    '''
    This method is used for both:
    1. Creating proposals in RPN (In which case num_classes=1)
    2. Adjusting predicted bounding box proposals (In which case num_classes=21 in PASCAL VOC)
    In case 1,
        :box_transform_pred: [num_anchors, num_classes, 4]
        :baseAnchors_or_proposals : [num_anchors, 4]
        where num_anchors = Batch_size(1) * feat_H * feat_W * anchors_per_location
        and num_classes = 1
    In case 2,
        :box_transform_pred:
        :baseAnchors_or_proposals: [num_proposals, 4]
        where
        and num_classes = 21
    '''
    box_transform_pred = box_transform_pred.reshape(
        box_trasnform_pred.size(0), -1, 4
    )

    ws = baseAnchors_or_proposals[:, 2] - baseAnchors_or_proposals[:, 0] # [num_anchors_or_proposals]
    hs = baseAnchors_or_proposals[:, 3] - baseAnchors_or_proposals[:, 1]
    ctr_x = baseAnchors_or_proposals[:, 0] + (0.5 * ws) # [num_anchors_or_proposals]
    ctr_y = baseAnchors_or_proposals[:, 1] + (0.5 * hs)

    # t* shape : [num_anchors_or_proposals, num_classes]
    t_x, t_y, t_w, t_h = box_transform_pred[..., 0], box_transform_pred[..., 1], box_transform_pred[..., 2], box_transform_pred[..., 3]
    
    # Prevent sending too large values into torch.exp()
    t_w = torch.clamp(t_w, max=math.log(1000.0 / 16))
    t_h = torch.clamp(t_h, max=math.log(1000.0 / 16))
    
    # pred_* shape: [num_anchors_or_proposals, num_clases]
    pred_ctr_x = t_x * ws[:, None] + ctr_x[:, None]
    pred_ctr_y = t_y * hs[:, None] + ctr_y[:, None]
    pred_w = torch.exp(t_w) * ws[:, None]
    pred_h = torch.exp(t_h) * hs[:, None]

    pred_box_xmin = pred_ctr_x - (0.5 * pred_w)
    pred_box_ymin = pred_ctr_y - (0.5 * pred_h)
    pred_box_xmax = pred_ctr_x + (0.5 * pred_w)
    pred_box_ymax = pred_ctr_y + (0.5 * pred_h)

    # Final transformed pred_boxes: [num_anchors_or_proposals, num_classes, 4]
    pred_boxes = torch.stack((pred_box_xmin, pred_box_ymin, pred_box_xmax, pred_box_ymax), dim=2)
    return pred_boxes


def clamp_boxes(boxes, image_shape):
    '''
    :boxes: [num_boxes, ..., 4]
    returns:
        :clamped_boxes: [num_boxes, ..., 4]
    '''
    xmin, ymin, xmax, ymax = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    width, height = image_shape[-2:]
    # clamp
    xmin = xmin.clamp(min=0, max=width) # [num_boxes, ...] last dimension removed by slicing in the first last
    ymin = ymin.clamp(min=0, max=height)
    xmax = xmax.clamp(min=0, max=width)
    ymax = ymax.clamp(min=0, max=height)
    # Reconstructing boxes
    clamped_boxes = torch.cat((
        xmin[:, None], # Adding new dimensions to make [num_boxes, ..., 1]
        ymin[:, None],
        xmax[:, None],
        ymax[:, None]),
        dim=-1)
    # Concat result in clamped_boxes : [num_boxes, ..., 4]
    return clamped_boxes


def assign_targets_to_anchors(gt_boxes, anchors):
    '''
    This method assigns labels(-1, 0, 1) to each anchor as well as ground truth box from the gt_boxes list.
    labels:
        -1 : Ignore (low_threshold < iou < high_threshold)
         0 : Background (iou < low_threshold)
         1 : Foreground (high_threshold < iou)
    paramenters:
        :gt_boxes: [num_gt_boxes_of_an_image, 4] #(N, 4)
        :anchors: [num_anchors_of_an_image, 4]   #(M, 4)
    returns:
        :matched_labels: [num_anchors_of_an_image, ]    #(M,)
        :matched_gt_boxes: [num_anchors_of_an_image, 4] #(M, 4)
    *Note: Returned matched_gt_boxes also assigns gt_box to background/ignore anchors which will be filtered later
    '''
    # Step 1: Calculate IoU matrix
    iou_matrix = compute_iou(gt_boxes, anchors) # (N, M)
    # Step 2: For each anchor, get the GT box index with the highest IoU
    best_iou, best_gt_idx = iou_matrix.max(dim=0) # [num_anchors_of_an_iamge,] = (M,)
    # Step 3: Initialize labels as Ignore (-1)
    labels = torch.full_like(best_iou, fill_value=-1, dtype=torch.float32)
    # Step 4: Apply thresholds to assign foreground (1) and background (0)
    labels[best_iou >= HIGH_IOU_THRESHOLD] = 1
    labels[best_iou < LOW_IOU_THRESHOLD] = 0
    # Step 5: For each GT box, get the maximum IoU value amongst all anchors
    max_iou_for_gt, _ = iou_matrix.max(dim=1) # [num_gt_boxes_of_an_image,] = (N,)
    # Step 6: Identify anchors with the highest IoU for each GT box
    gt_pair_with_highest_iou = torch.where(iou_matrix == max_iou_for_gt[:, None]) # [num_gt_boxes_of_an_image, num_anchors_of_an_image] = (N, M)
    anchor_idx_to_update = gt_pair_with_highest_iou[1]
    # Step 7: Ensure the highest IoU anchors are labeled as foreground
    labels[anchor_idx_to_update] = 1
    # Step 8: Matched GT boxes 
    matched_gt_boxes = gt_boxes[best_gt_idx]

    return labels, matched_gt_boxes


def compute_bbox_transformation_targets(gt_boxes, anchors_or_proposals):
    '''
    This method computes the bounding box regression targets (tx, ty, tw, th)
    provided all anchors/proposals and corresponding target gt_boxes
    variables:
        G_*: gt_boxes
        P_*: anchors_or_proposals
        t_*: target offsets
    parameters:
        :gt_boxes: [num_anchors_or_proposals_in_an_image, 4] = (N, 4) (We assign one gt_box to each anchor/proposal)
        :anchors_or_proposals: [num_anchors_or_proposals_in_an_image, 4] = (N, 4)
    returns:
        :target_offsets: [num_anchors_or_proposals_in_an_image, 4] = (N, 4)
    '''
    # anchors_or_proposals info Shape: (N,)
    P_w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    P_h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    P_x = anchors_or_proposals[:, 0] + (0.5 * P_w)
    P_y = anchors_or_proposals[:, 1] + (0.5 * P_h)

    # gt_boxes info Shape: (N,)
    G_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    G_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    G_x = gt_boxes[:, 0] + (0.5 * G_w)
    G_y = gt_boxes[:, 1] + (0.5 * G_h)

    # Target offsets Shape: (N, 4)
    t_x = (G_x - P_x) / P_w
    t_y = (G_y - P_y) / P_h
    t_w = torch.log(G_w / P_w)
    t_h = torch.log(G_h / P_h)
    
    target_offsets = torch.stack((t_x, t_y, t_w, t_h), dim=1)
    
    return target_offsets


def sample_positive_and_negative(labels, positive_ct, total_ct):
    pos_idx = torch.where(labels >= 1)[0]
    neg_idx = torch.where(labels == 0)[0]
    actual_positive_ct = min(pos_idx.numel(), positive_ct)
    negative_ct = total_ct - actual_positive_ct
    actual_negative_ct = min(negative_ct, neg_idx.numel())

    # Random sampling
    random_idx_pos = torch.randperm(pos_idx.size(0), device=pos_idx.device)[:actual_positive_ct]
    random_idx_neg = torch.randperm(neg_idx.size(0), device=neg_idx.device)[:actual_negative_ct]
    pos_idx = pos_idx[random_idx_pos]
    neg_idx = neg_idx[random_idx_pos]

    # Creating a mask
    pos_mask = torch.zeros_like(labels, dtype=torch.bool)
    neg_mask = torch.zeros_like(labels, dtype=torch.bool)
    pos_mask[pos_idx] = True
    neg_mask[neg_idx] = True
    return pos_mask, neg_mask






###############################
########## RPN Utils ##########
###############################

def generate_anchor_base(image, feature, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
    '''
    :image: [N, C, H, W] where N = 1
    :feature: [N, C, H, W] where N = 1
    Example aspect ratios assuming anchors of scale 128 sq pixels:
    1:1 would be (128, 128)      with area=16384
    2:1 would be (181.02, 90.51) with area=16384
    1:2 would be (90.51, 181.02) with area=16384
    '''
    feat_h, feat_w = feature.shape[-2:]
    image_h, image_w = image.shape[-2:]
    scales = torch.as_tensor(scales, dtype=feature.dtype, device=feature.device)
    ratios = torch.as_tensor(ratios, dtype=feature.dtype, device=feature.device)
    
    # Find the stride(downsampling ratio from input image to feature map)
    stride_h = torch.tensor(image_h // feat_h, dtype=torch.int64, device=feature.device)
    stride_w = torch.tensor(image_w // feat_w, dtype=torch.int64, device=feature.device)
    
    # Find the ratios the widths and heights will be stretched by
    ratios_w = torch.sqrt(ratios)
    ratios_h = 1 / ratios_w
    
    # Rescaling the widths and heights w.r.t ratios
    '''
    [3, 1] * [1, 3] -> [3, 3] -> [9,]
    '''
    widths = (ratios_w[:, None] * scales[None, :]).view(-1)
    heights = (ratios_h[:, None] * scales[None, :]).view(-1)

    # Make anchors zero centered
    base_anchors = torch.stack([-widths, -heights, widths, heights], dim=1) / 2 # [9, 4]
    base_anchors = base_anchors.round()

    # Get the center coordinates where base_anchors will be replicated on (i.e every featuremap location on the image)
    shift_x = torch.arange(0, feat_w, dtype=torch.int32, device=feature.device) * stride_w # [feat_w,]
    shift_y = torhc.arange(0, feat_h, dtype=torch.int32, device=feature.device) * stride_h # [feat_h,]
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    '''
    :shift_x: [feat_H, feat_W] *Note that we do want height comes first for reason (recall PyTorch's tensors follow N,C,H,W)
    :shift_y: [feat_H, feat_W]
    '''
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
    '''
    :shift_x: [feat_H * feat_W]
    :shift_y: [feat_H * feat_W]
    '''
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    ''' :shifts: [feat_H * feat_W, 4]'''

    # Adding anchor variations to each featuremap location
    ''' [feat_H * feat_W, 1, 4] + [1, num_anchors, 4] = [feat_H * feat_W, num_anchors, 4]'''
    anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
    anchors = anchors.reshape(-1, 4) # [feat_H * feat_W * num_anchors, 4]
    
    return anchors
                   

def sample_proposals(proposals, cls_scores, image_shape):
    '''
    This method samples positive / negative proposals from transformed anchor boxes
    1. Pre NMS topK filtering
    2. Make proposals valid by clamping coordinates(0, width/height)
    3. Small Boxes filtering based on width and height
    4. NMS
    5. Post NMS topK filtering
    params:
        :proposals: [num_anchors, 4]
        :cls_scores: [num_anchors, 1]
    return:
        :sampled_proposals, sampled_scores: [num_sampled_proposals, 4], [num_sampled_proposals]
    '''
    # Step 1: Pre NMS topK
    cls_scores = cls_scores.reshape(-1)    # [num_anchors, ]
    cls_scores = torch.sigmoid(cls_scores) # All scores are distributed between 0 and 1
    _, topk_idx = torch.topk(cls_scores, min(PRE_NMS_TOPK, len(cls_scores)))
    proposals = proposals[topk_idx]
    cls_scores[topk_idx]
    
    # Step 2: Clamp boxes to fit in image
    proposals = clamp_boxes(proposals, image_shape)

    # Step 3: Filter out small boxes based on min size width/height
    min_size = 16
    ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    keep = (ws >= min_size) & (hs >= min_size) # 0s and 1s
    keep = torch.where(keep)[0] # Trues and Falses
    proposals = proposals[keep]
    cls_scores = cls_scores[keep]

    # Step 4: Apply NMS based on the objectness score
    keep_idx = nms(proposals, cls_scores, 0.7)
    # Sort
    sorted_keep_idx = keep_idx[cls_scores[keep_idx].sort(descending=True)[1]]

    # Step 5: TopK objectness proposals after NMS
    proposals = proposals[sorted_keep_idx[:FINAL_TOPK]]
    cls_scores = cls_scores[sorted_keep_idx[:FINAL_TOPK]]

    return proposals, cls_scores








########################################
########## ROI_Detector Utils ##########
########################################

def assign_targets_to_proposals(proposals, gt_boxes, gt_labels):
    '''
    This method assigns labels(-1, 0, ...) to each proposal as well as ground truth box from the gt_boxes list.
    labels:
        -1 : Ignore (low_threshold < iou < high_threshold)
         0 : Background (iou < low_threshold)
       ... : Foreground (high_threshold < iou)
    paramenters:
        :gt_boxes: [num_gt_boxes_of_an_image, 4]   #(N, 4)
        :proposals: [num_proposals_of_an_image, 4] #(M, 4)
    returns:
        :matched_labels: [num_proposasl_of_an_image, ]    #(M,)
        :matched_gt_boxes: [num_proposals_of_an_image, 4] #(M, 4)
    *Note: Returned matched_gt_boxes also assigns gt_box to ignore proposals which will be filtered later
    '''
    # Step 1: Calculate IoU matrix
    iou_matrix = compute_iou(gt_boxes, proposals)
    # Step 2: For each proposal, get the GT box index with the highest IoU
    best_iou, best_gt_idx = iou_matrix.max(dim=0) # [num_proposals_of_an_iamge,] = (M,)
    # Step 3: Background and Ignore idx
    background = (best_iou >= ROI_LOW_IOU_THRESHOLD) & (best_iou < ROI_HIGH_IOU_THRESHOLD)
    ignore = (best_ious < ROI_LOW_IOU_THRESHOLD)
    best_gt_idx[background] = -1
    best_gt_idx[ignore] = -2
    # Step 4: matched gt_boxes for all proposals
    # This includes backgrounds/ignores by assigning the gt_box at index 0 through clamping -1 and -2, 
    # This is not correct but it's okay as we will filter them out when we actually use the boxes
    matched_gt_boxes = gt_boxes[best_gt_idx.clamp(min=0)]
    # Step 5: matched labels for all proposals
    labels = gt_labels[best_gt_idx.clamp(min=0)].to(dtype=torch.int32)
    labels[background] = 0
    labels[ignore] = -1

    return matched_labels, matched_gt_boxes








###############################################
########## Faster-RCNN Wrapper Utils ##########
###############################################

def normalize_resize_image_and_boxes(image, bboxes, image_mean, image_std, min_size=600, max_size=1000):
    dtype, device = image.dtype, image.device

    # Normalize
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    image = (image - mean[:, None, None]) / std[:, None, None] # Broadcasting for image shape [3, H, W]

    # Find scaling factor such taht the smaller dimension (either height or width) is scaled up to 600, and
    # the larger dimension (the other axis) does not exceed 1000
    h, w = image.shape[-2:] # Store image shape before change
    image_shape = torch.tensor(image.shape[-2:])
    smaller_dim = torch.min(image_shape).to(dtype=torch.float32)
    larger_dim = torch.max(image_shape).to(dtype=torch.float32)
    scaling_factor = torch.min((min_size / smaller_dim), (max_size / larger_dim))

    # Resize image based on scale computed
    image = torch.nn.functional.interpolate(
        image,
        size=None,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )

    if bboxes is not None:
        # Resize boxes by
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=bboxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
            for s, s_orig in zip(image.shape[-2:], (h, w))
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = bboxes.unbind(2)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
    return image, bboxes