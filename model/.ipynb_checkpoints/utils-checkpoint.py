import torch
import numpy as np

##### Global Variables #####
TOPK = 12000
############################

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
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width, height = image_shape[-2:]
    # 

def sample_proposals(proposals, cls_scores, image_shape):
    '''
    This method samples positive / negative proposals from transformed anchor boxes
    1. Pre NMS topK filtering
    2. Make proposals valid by clamping coordinates(0, width/height)
    2. Small Boxes filtering based on width and height
    3. NMS
    4. Post NMS topK filtering
    params:
        :proposals: [num_anchors, 4]
        :cls_scores: [num_anchors, 1]
    return:
        :sampled_proposals, sampled_scores: [num_sampled_proposals, 4], [num_sampled_proposals]
    '''
    # Step 1: Pre NMS topK
    cls_scores = cls_scores.reshape(-1)    # [num_anchors, ]
    cls_scores = torch.sigmoid(cls_scores) # All scores are distributed between 0 and 1
    _, topk_idx = torch.topk(cls_scores, min(TOPK, len(cls_scores)))
    proposals = proposals[topk_idx]
    cls_scores[topk_idx]
    
    # Step 2: Clamp boxes to fit in image
    proposals = clamp_boxes(proposals, image_shape)